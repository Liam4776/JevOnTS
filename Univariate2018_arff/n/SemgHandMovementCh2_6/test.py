"""Standalone multiclass classifier for SemgHandMovementCh2 in the UEA archive."""
import ast
import csv
import http.client
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-1.13.0'
BATCH_SIZE = 15
TIMEOUT = 180
MAX_RETRIES = 6
CONFIG = {'task': 'SemgHandMovementCh2',
 'slug': 'semghandmovementch2',
 'length': 1500,
 'names': {'1': 'Cylindrical grasp',
           '2': 'Hook grasp',
           '3': 'Tip grasp',
           '4': 'Palmar grasp',
           '5': 'Spherical grasp',
           '6': 'Lateral grasp'},
 'criteria': {'1': 'The series belongs to Cylindrical grasp.',
              '2': 'The series belongs to Hook grasp.',
              '3': 'The series belongs to Tip grasp.',
              '4': 'The series belongs to Palmar grasp.',
              '5': 'The series belongs to Spherical grasp.',
              '6': 'The series belongs to Lateral grasp.'},
 'prompt': 'Classify this channel-2 sEMG power spectrum by the hand-grasp movement: cylindrical, '
           'hook, tip, palmar, spherical, or lateral.',
 'measurement': 'semg_spectrum_values'}

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

def _read_arff(path: Path, cfg: dict[str, Any]):
    rows = []
    active = False
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith('%'):
                continue
            if not active:
                if stripped.lower() == '@data':
                    active = True
                continue
            rows.append(line)
    if not active:
        raise ValueError(f'ARFF 文件缺少 @data：{path}')

    sequences, labels = [], []
    for row_number, row in enumerate(csv.reader(rows), 1):
        if len(row) != cfg['length'] + 1:
            raise ValueError(f'第 {row_number} 行列数为 {len(row)}，应为 {cfg["length"] + 1}。')
        label = row[-1].strip().strip("'\"")
        if label not in cfg['names']:
            raise ValueError(f'第 {row_number} 行标签 {label!r} 不在类别列表中。')
        values = []
        for raw in row[:-1]:
            value = raw.strip().strip("'\"")
            if value.lower() in ('?', 'nan', ''):
                values.append(None)
            else:
                number = float(value)
                values.append(number if math.isfinite(number) else None)
        sequences.append(values)
        labels.append(label)
    if not sequences:
        raise ValueError('测试文件中没有样本。')
    return sequences, labels

def _payload(cfg: dict[str, Any], sequences):
    samples, questions = [], {}
    for index, sequence in enumerate(sequences, 1):
        samples.append({'sample_id': index, cfg['measurement']: sequence})
        question_id = f'sample_{index:04d}'
        questions[question_id] = {
            'type': 'choice',
            'instructions': f'Classify sample_id {index} using state.task_instructions and state.class_definitions. Select the single best matching class.',
            'criteria': cfg['criteria'],
        }
    return {
        'state': {
            'task_instructions': cfg['prompt'],
            'class_definitions': cfg['criteria'],
            'samples': samples,
        },
        'model': MODEL,
        'questions': questions,
    }

def _request(cfg: dict[str, Any], key: str, sequences):
    body = json.dumps(_payload(cfg, sequences), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(
            API_ENDPOINT,
            data=body,
            method='POST',
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'Accept': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                answers = json.loads(response.read().decode('utf-8'))['answers']
            results = []
            for index in range(1, len(sequences) + 1):
                answer = answers[f'sample_{index:04d}']
                predicted = str(answer['choice']).strip()
                confidence = float(answer['confidence'])
                raw_probabilities = answer['probabilities']
                probabilities = {label: float(raw_probabilities[label]) for label in cfg['names']}
                if predicted not in cfg['names']:
                    raise JevAPIError(f'样本 {index} 的预测类别无效：{predicted!r}')
                if not 0 <= confidence <= 1 or any(not 0 <= score <= 1 for score in probabilities.values()):
                    raise JevAPIError(f'样本 {index} 的置信度或类别概率超出 [0, 1]。')
                results.append((predicted, confidence, probabilities))
            return results
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')[:1000]
            if error.code == 400 and 'max_tokens_exceeded' in detail:
                raise JevAPIError(f'max_tokens_exceeded: 当前批次 {len(sequences)} 条。') from error
            if (error.code == 429 or error.code >= 500) and attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise JevAPIError(f'TypeSafe API HTTP {error.code}: {detail}') from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            if attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise JevAPIError(f'网络重试失败：{error}') from error
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise JevAPIError(f'API 响应格式异常：{error}') from error
    raise JevAPIError('Jev 请求未能获得响应。')

def _predict(cfg: dict[str, Any], key: str, sequences):
    try:
        return _request(cfg, key, sequences)
    except JevAPIError as error:
        if 'max_tokens_exceeded' not in str(error) or len(sequences) <= 1:
            raise
        middle = len(sequences) // 2
        print(f'批次 {len(sequences)} 超过 token 限制，拆分为 {middle} 和 {len(sequences) - middle} 条。', file=sys.stderr)
        return _predict(cfg, key, sequences[:middle]) + _predict(cfg, key, sequences[middle:])

def _binary_auc(labels, scores, positive):
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(label == positive for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(label == positive for _, label in ranked[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

def _macro_ovr_auc(labels, probability_rows, class_labels):
    class_aucs = []
    for class_label in class_labels:
        scores = [row[class_label] for row in probability_rows]
        auc = _binary_auc(labels, scores, class_label)
        if auc is None:
            return None
        class_aucs.append(auc)
    return sum(class_aucs) / len(class_aucs)

def run(cfg: dict[str, Any]) -> int:
    task_dir = Path(__file__).resolve().parent
    test_file = task_dir / f"{cfg['task']}_TEST.arff"
    output_dir = task_dir / f"jev_{cfg['slug']}_output"
    prediction_file = output_dir / f"{cfg['task']}_TEST_jev_predictions.csv"
    metrics_file = output_dir / f"{cfg['task']}_TEST_acc_auc.txt"
    try:
        key = _api_key(task_dir)
        sequences, labels = _read_arff(test_file, cfg)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'配置或输入错误：{error}', file=sys.stderr)
        return 2

    print(f"处理 {len(sequences)} 条 {cfg['task']} 测试序列，每批最多 {BATCH_SIZE} 条。", file=sys.stderr)
    results = []
    try:
        for start in range(0, len(sequences), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sequences))
            results.extend(_predict(cfg, key, sequences[start:end]))
            print(f'已完成 {end}/{len(sequences)} 条。', file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        probability_fields = [f'probability_{label}' for label in cfg['names']]
        fields = ['predicted_class', 'predicted_name', 'confidence', *probability_fields, 'true_label', 'true_name']
        with prediction_file.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for (predicted, confidence, probabilities), actual in zip(results, labels):
                record = {
                    'predicted_class': predicted,
                    'predicted_name': cfg['names'][predicted],
                    'confidence': f'{confidence:.8f}',
                    'true_label': actual,
                    'true_name': cfg['names'][actual],
                }
                for label in cfg['names']:
                    record[f'probability_{label}'] = f'{probabilities[label]:.8f}'
                writer.writerow(record)
    except (OSError, JevAPIError) as error:
        print(f'分类中止：{error}', file=sys.stderr)
        return 1

    predictions = [result[0] for result in results]
    probability_rows = [result[2] for result in results]
    accuracy = sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels)
    auc = _macro_ovr_auc(labels, probability_rows, list(cfg['names']))
    auc_text = f'{auc:.4f}' if auc is not None else 'N/A'
    try:
        metrics_file.write_text(f'ACC: {accuracy:.4f}\nAUC: {auc_text}\n', encoding='utf-8')
    except OSError as error:
        print(f'ACC/AUC 文件写入失败：{error}', file=sys.stderr)
        return 1
    print(f'ACC/AUC 已保存：{metrics_file}', file=sys.stderr)
    print(f'预测结果已保存：{prediction_file}', file=sys.stderr)
    print(f'ACC: {accuracy:.4f} ({accuracy * 100:.2f}%)', file=sys.stderr)
    print(f'Macro one-vs-rest ROC AUC: {auc_text}', file=sys.stderr)
    return 0

if __name__ == '__main__':
    raise SystemExit(run(CONFIG))
