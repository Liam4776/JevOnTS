"""FordB 测试集完整批量分类脚本。"""
from __future__ import annotations
import ast, csv, http.client, json, math, os, sys, time, urllib.error, urllib.request
from pathlib import Path
from typing import Any

API_ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-1.13.0'
BATCH_SIZE = 5
TIMEOUT = 180
MAX_RETRIES = 6

class JevAPIError(RuntimeError):
    pass

def _api_key(task_dir: Path) -> str:
    env_key = os.environ.get("JEV_API_KEY", "").strip()
    if env_key:
        if env_key == "YOUR API KEY":
            raise ValueError("Set JEV_API_KEY to a real API key")
        return env_key
    key_file = task_dir.parents[1] / 'api_key.json'
    if not key_file.is_file():
        raise RuntimeError(f'API 密钥文件不存在：{key_file}')
    try:
        data = json.loads(key_file.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'无法读取 API 密钥文件：{error}') from error
    key = data.get('api_key') if isinstance(data, dict) else None
    if not isinstance(key, str) or not key.strip() or key.strip() == "YOUR API KEY":
        raise RuntimeError(f'API 密钥文件缺少有效的 api_key：{key_file}')
    return key.strip()

def _read_arff(path: Path, length: int, class_names: dict[str, str]):
    data, active = [], False
    with path.open(encoding='utf-8-sig', newline='') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            if not active:
                active = s.lower() == '@data'
                continue
            data.append(line)
    if not active:
        raise ValueError(f'ARFF 文件缺少 @data：{path}')
    sequences, labels = [], []
    for i, row in enumerate(csv.reader(data), 1):
        if len(row) != length + 1:
            raise ValueError(f'第 {i} 行列数为 {len(row)}，应为 {length + 1}。')
        label = row[-1].strip().strip("'\"")
        if label not in class_names:
            raise ValueError(f'第 {i} 行标签 {label!r} 不在 {tuple(class_names)} 中。')
        seq = []
        for raw in row[:-1]:
            value = raw.strip().strip("'\"")
            if value.lower() in ('?', 'nan', ''):
                seq.append(None)
            else:
                number = float(value)
                seq.append(number if math.isfinite(number) else None)
        sequences.append(seq)
        labels.append(label)
    if not sequences:
        raise ValueError('测试文件中没有样本。')
    return sequences, labels

def _payload(cfg: dict[str, Any], sequences):
    samples, questions = [], {}
    for i, seq in enumerate(sequences, 1):
        samples.append({'sample_id': i, cfg['measurement']: list(seq)})
        qid = f'sample_{i:03d}'
        questions[qid] = {
            'type': 'choice',
            'instructions': f'Classify only sample_id {i} in state.samples using state.task_instructions and state.class_definitions.',
            'criteria': cfg['criteria'],
        }
    return {'state': {'task_instructions': cfg['prompt'], 'class_definitions': cfg['criteria'], 'samples': samples}, 'model': MODEL, 'questions': questions}

def _request(cfg, key, seqs):
    body = json.dumps(_payload(cfg, seqs), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(API_ENDPOINT, data=body, method='POST', headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'Accept': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                answers = json.loads(r.read().decode())['answers']
            out = []
            for i in range(1, len(seqs) + 1):
                a = answers[f'sample_{i:03d}']
                label = str(a['choice']).strip()
                conf = float(a['confidence'])
                score = float(a['probabilities'][cfg['positive']])
                if label not in cfg['names'] or not 0 <= conf <= 1 or not 0 <= score <= 1:
                    raise JevAPIError(f'样本 {i} 的响应类别或概率无效。')
                out.append((label, conf, score))
            return out
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors='replace')[:1000]
            if e.code == 400 and 'max_tokens_exceeded' in text:
                raise JevAPIError(f'max_tokens_exceeded: 当前批次 {len(seqs)} 条。') from e
            if (e.code == 429 or e.code >= 500) and attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt)); continue
            raise JevAPIError(f'TypeSafe API HTTP {e.code}: {text}') from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            if attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt)); continue
            raise JevAPIError(f'网络重试失败：{e}') from e
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            raise JevAPIError(f'API 响应格式异常：{e}') from e
    raise JevAPIError('Jev 请求未能获得响应。')

def _predict(cfg, key, seqs):
    try:
        return _request(cfg, key, seqs)
    except JevAPIError as e:
        if 'max_tokens_exceeded' not in str(e) or len(seqs) <= 1:
            raise
        mid = len(seqs) // 2
        print(f'批次 {len(seqs)} 超过 token 限制，拆分为 {mid} 和 {len(seqs)-mid} 条。', file=sys.stderr)
        return _predict(cfg, key, seqs[:mid]) + _predict(cfg, key, seqs[mid:])

def _auc(labels, scores, positive):
    ranked = sorted(zip(scores, labels))
    p = sum(y == positive for y in labels); n = len(labels) - p
    if not p or not n:
        return None
    rank_sum = 0.0; i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y == positive for _, y in ranked[i:j])
        i = j
    return (rank_sum - p * (p + 1) / 2) / (p * n)

def run(cfg: dict[str, Any]):
    folder = Path(__file__).resolve().parent
    test_file = folder / f"{cfg['task']}_TEST.arff"
    output_dir = folder / f"jev_{cfg['slug']}_output"
    output_file = output_dir / f"{cfg['task']}_TEST_jev_predictions.csv"
    try:
        key = _api_key(folder)
        sequences, labels = _read_arff(test_file, cfg['length'], cfg['names'])
    except (OSError, ValueError, RuntimeError) as e:
        print(f'配置或输入错误：{e}', file=sys.stderr); return 2
    print(f"处理 {len(sequences)} 条 {cfg['task']} 测试序列，每批最多 {BATCH_SIZE} 条。", file=sys.stderr)
    results = []
    try:
        for start in range(0, len(sequences), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sequences))
            results.extend(_predict(cfg, key, sequences[start:end]))
            print(f'已完成 {end}/{len(sequences)} 条。', file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8-sig', newline='') as f:
            fields = ['predicted_class', 'predicted_name', 'confidence', cfg['score_field'], 'true_label', 'true_name']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for (pred, conf, score), actual in zip(results, labels):
                writer.writerow({'predicted_class': pred, 'predicted_name': cfg['names'][pred], 'confidence': f'{conf:.8f}', cfg['score_field']: f'{score:.8f}', 'true_label': actual, 'true_name': cfg['names'][actual]})
    except (OSError, JevAPIError) as e:
        print(f'分类中止：{e}', file=sys.stderr); return 1
    predictions = [x[0] for x in results]; scores = [x[2] for x in results]
    acc = sum(a == p for a, p in zip(labels, predictions)) / len(labels)
    auc = _auc(labels, scores, cfg['positive'])
    metrics_file = output_dir / f"{cfg['task']}_TEST_acc_auc.txt"
    auc_value = f"{auc:.4f}" if auc is not None else "N/A"
    try:
        metrics_file.write_text(f"ACC: {acc:.4f}\nAUC: {auc_value}\n", encoding="utf-8")
    except OSError as e:
        print(f"ACC/AUC 文件写入失败：{e}", file=sys.stderr)
        return 1
    print(f"ACC/AUC 已保存：{metrics_file}", file=sys.stderr)
    print(f'预测结果已保存：{output_file}', file=sys.stderr)
    print(f'ACC: {acc:.4f} ({acc*100:.2f}%)', file=sys.stderr)
    print(f"ROC AUC（{cfg['positive_name']} 为正类）: {auc:.4f}" if auc is not None else 'ROC AUC: N/A（评估集必须同时包含两个类别）', file=sys.stderr)
    return 0


CONFIG = {'task': 'FordB',
 'slug': 'fordb',
 'length': 500,
 'names': {'-1': 'SourceClassMinus1', '1': 'SourceClass1'},
 'positive': '1',
 'positive_name': 'SourceClass1',
 'score_field': 'probability_class_1',
 'measurement': 'engine_noise_measurements',
 'prompt': 'Classify this sequence of 500 engine-noise measurements from the FordB automotive '
           'subsystem task. The archive asks whether a subsystem symptom is present; test '
           'records were collected under noisier conditions. Return the matching source class '
           'label.',
 'criteria': {'-1': 'The engine-noise sequence has source class label -1.',
              '1': 'The engine-noise sequence has source class label 1.'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
