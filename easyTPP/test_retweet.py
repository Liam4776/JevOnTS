from __future__ import annotations

import os
import csv
import argparse
import random
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
BATCH_SIZE = 100
TIMEOUT = 180
MAX_RETRIES = 6


class JevAPIError(RuntimeError):
    pass


def _api_key() -> str:
    env_key = os.environ.get("JEV_API_KEY", "").strip()
    if env_key:
        if env_key == "YOUR API KEY":
            raise ValueError("Set JEV_API_KEY to a real API key")
        return env_key
    path = Path(__file__).resolve().parent / "api.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    key = data.get("api_key") if isinstance(data, dict) else None
    if not isinstance(key, str) or not key.strip() or key.strip() == "YOUR API KEY":
        raise ValueError("请在 api.json 中填写有效的 api_key")
    return key.strip()


def _read_sequences(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    data_path = Path(__file__).resolve().parent / cfg["data_file"]
    try:
        records = json.loads(data_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {data_path}: {error}") from error
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array in {data_path}")

    required = {
        "dim_process", "seq_idx", "seq_len", "time_since_start",
        "time_since_last_event", "type_event",
    }
    event_count = None
    sequences: list[dict[str, Any]] = []
    dims: set[int] = set()
    for row_no, record in enumerate(records, 1):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"Record {row_no} is missing required EasyTPP fields")
        dim = int(record["dim_process"])
        types = record["type_event"]
        gaps = record["time_since_last_event"]
        times = record["time_since_start"]
        length = int(record["seq_len"])
        if not (len(types) == len(gaps) == len(times) == length):
            raise ValueError(f"Record {row_no}: seq_len and event-array lengths differ")
        if length < 2:
            continue
        for event_type in types:
            if not isinstance(event_type, int) or not 0 <= event_type < dim:
                raise ValueError(f"Record {row_no}: invalid type_event value {event_type!r}")
        dims.add(dim)
        event_count = (event_count or 0) + length - 1
        sequences.append({
            "seq_idx": int(record["seq_idx"]),
            "types": types,
            "gaps": gaps,
            "times": times,
        })
    if len(dims) != 1:
        raise ValueError(f"Expected one dim_process value, found {sorted(dims)}")
    if not sequences or not event_count:
        raise ValueError("No next-event samples can be formed from this file")
    return sequences, dims.pop()


def iter_batches(sequences, batch_size):
    """每批来自不同序列；同一序列每批最多一个目标，防止答案泄漏。"""
    active = []
    pending = iter(sequences)
    sample_id = 0
    while True:
        while len(active) < batch_size:
            sequence = next(pending, None)
            if sequence is None:
                break
            active.append((sequence, 1))
        if not active:
            return
        samples, metadata, following = [], [], []
        for sequence, pos in active:
            sample_id += 1
            history = [
                {"type_event": int(sequence["types"][j]),
                 "time_since_start": float(sequence["times"][j]),
                 "time_since_last_event": float(sequence["gaps"][j])}
                for j in range(pos)
            ]
            samples.append({
                "sample_id": sample_id,
                "event_history": history,
                "target_time_since_start": float(sequence["times"][pos]),
                "target_time_since_last_event": float(sequence["gaps"][pos]),
            })
            metadata.append({"sample_id": sample_id, "seq_idx": sequence["seq_idx"],
                             "target_event_index": pos, "history_length": pos,
                             "target_time_since_start": sequence["times"][pos],
                             "target_time_since_last_event": sequence["gaps"][pos],
                             "true_label": str(sequence["types"][pos])})
            if pos + 1 < len(sequence["types"]):
                following.append((sequence, pos + 1))
        yield samples, metadata
        active = following


def _payload(cfg: dict[str, Any], samples: list[dict[str, Any]], dim_process: int) -> dict[str, Any]:
    criteria = {
        str(event_type): f"The next event has type_event ID {event_type}."
        for event_type in range(dim_process)
    }
    questions: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        question_id = f"sample_{sample_id:08d}"
        questions[question_id] = {
            "type": "choice",
            "instructions": (
                f"Predict the next event type for sample_id {sample_id} from its event_history and the known target event time. Treat every sample independently. "
                "Use state.task_instructions and state.class_definitions. "
                "Select exactly one event type."
            ),
            "criteria": criteria,
        }
    return {
        "state": {
            "task_instructions": cfg["prompt"],
            "class_definitions": criteria,
            "samples": samples,
        },
        "model": MODEL,
        "questions": questions,
    }


def _request(cfg: dict[str, Any], key: str, samples: list[dict[str, Any]], dim_process: int):
    body = json.dumps(
        _payload(cfg, samples, dim_process), ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(
            API_ENDPOINT, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                answers = json.loads(response.read().decode("utf-8"))["answers"]
            results = []
            for sample in samples:
                question_id = f"sample_{sample['sample_id']:08d}"
                answer = answers[question_id]
                predicted = str(answer["choice"]).strip()
                confidence = float(answer["confidence"])
                raw_probabilities = answer["probabilities"]
                probabilities = {
                    str(label): float(raw_probabilities[str(label)])
                    for label in range(dim_process)
                }
                if predicted not in probabilities:
                    raise JevAPIError(f"Invalid predicted event type: {predicted!r}")
                if not 0 <= confidence <= 1 or any(not 0 <= p <= 1 for p in probabilities.values()):
                    raise JevAPIError("API returned a confidence or probability outside [0, 1]")
                results.append((predicted, confidence, probabilities))
            return results
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1000]
            if error.code == 400 and "max_tokens_exceeded" in detail:
                raise JevAPIError("max_tokens_exceeded") from error
            if (error.code == 429 or error.code >= 500) and attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise JevAPIError(f"TypeSafe API HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            if attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt))
                continue
            raise JevAPIError(f"Network retries exhausted: {error}") from error
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise JevAPIError(f"Unexpected API response: {error}") from error
    raise JevAPIError("JEV request did not return a response")


def _predict(cfg: dict[str, Any], key: str, samples: list[dict[str, Any]], dim_process: int):
    try:
        return _request(cfg, key, samples, dim_process)
    except JevAPIError as error:
        if "max_tokens_exceeded" not in str(error) or len(samples) <= 1:
            raise
        middle = len(samples) // 2
        print(f"Splitting oversized batch of {len(samples)} into {middle} and {len(samples)-middle}.", file=sys.stderr)
        return (
            _predict(cfg, key, samples[:middle], dim_process)
            + _predict(cfg, key, samples[middle:], dim_process)
        )


def _binary_auc(labels: list[str], scores: list[float], positive: str):
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


def _macro_ovr_auc(labels: list[str], probability_rows: list[dict[str, float]], dim_process: int):
    aucs = []
    for event_type in range(dim_process):
        label = str(event_type)
        scores = [row[label] for row in probability_rows]
        auc = _binary_auc(labels, scores, label)
        if auc is None:
            return None
        aucs.append(auc)
    return sum(aucs) / len(aucs)


CONFIG = {'data_file': 'retweet_test.json', 'prompt': "Predict the type of the next retweet event from the observed history. Each event represents a retweet, and its type indicates the retweeter's influence group: small (fewer than 120 followers), medium (120-1,362 followers), or large (at least 1,363 followers). Use the order and timing of past events to select the next event type. Use only the sample's true historical event types and times plus its known target time. The target event type is hidden. Treat every sample as an independent sequence; do not use other samples as history. Return a type ID from 0 to 2 and probabilities for all 3 types. Numeric category IDs are identifiers; do not assume an unspecified mapping to semantic category names."}


def main():
    parser = argparse.ArgumentParser(description="抽样 retweet 序列，使用历史和目标时间预测事件类型")
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "jev_retweet_sampled_output")
    parser.add_argument("--dry-run", action="store_true", help="验证抽样和批次，不调用 API")
    args = parser.parse_args()
    if args.num_sequences < 1 or args.batch_size < 1:
        parser.error("num-sequences 和 batch-size 必须大于 0")
    sequences, dim = _read_sequences(CONFIG)
    if args.num_sequences > len(sequences):
        parser.error(f"仅有 {len(sequences)} 条有效序列")
    if len({s["seq_idx"] for s in sequences}) != len(sequences):
        parser.error("输入包含重复 seq_idx")
    selected = random.Random(args.seed).sample(sequences, args.num_sequences)
    total = sum(len(s["types"]) - 1 for s in selected)
    print(f"随机种子 {args.seed}；抽取 {len(selected)} 条序列；目标事件 {total}；每批最多 {args.batch_size} 条不同序列。")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "num_sequences": len(selected),
                "batch_size": args.batch_size, "total_targets": total,
                "data_file": CONFIG["data_file"],
                "sequences": [{"seq_idx": s["seq_idx"], "seq_len": len(s["types"])} for s in selected]}
    if args.dry_run:
        count = batches = 0
        for samples, metadata in iter_batches(selected, args.batch_size):
            assert len({m["seq_idx"] for m in metadata}) == len(metadata)
            assert len(samples) <= args.batch_size
            for sample, meta in zip(samples, metadata):
                assert len(sample["event_history"]) == meta["target_event_index"]
                assert "true_label" not in sample and "type_event" not in sample
            count += len(samples)
            batches += 1
        assert count == total
        (output / "retweet_dry_run_selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"验证通过：{batches} 批，{count} 个目标；未调用 API。")
        return 0
    key = _api_key()
    (output / "retweet_selected_sequences.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    predictions, labels, probabilities = [], [], []
    per_sequence = {}
    prediction_file = output / "retweet_next_event_predictions.csv"
    metrics_file = output / "retweet_next_event_metrics.txt"
    # 开始新运行后删除旧指标，避免中断时误用上一次完整运行的指标。
    metrics_file.unlink(missing_ok=True)
    fields = ["sample_id", "seq_idx", "target_event_index", "history_length",
              "target_time_since_start", "target_time_since_last_event", "predicted_type",
              "confidence", *[f"probability_{i}" for i in range(dim)], "true_type"]
    with prediction_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for samples, metadata in iter_batches(selected, args.batch_size):
            results = _predict(CONFIG, key, samples, dim)
            for meta, (predicted, confidence, probs) in zip(metadata, results):
                row = {k: v for k, v in meta.items() if k != "true_label"}
                row.update(predicted_type=predicted, confidence=f"{confidence:.8f}", true_type=meta["true_label"])
                row.update({f"probability_{i}": f"{probs[str(i)]:.8f}" for i in range(dim)})
                writer.writerow(row)
                predictions.append(predicted)
                labels.append(meta["true_label"])
                probabilities.append(probs)
                counts = per_sequence.setdefault(meta["seq_idx"], [0, 0])
                counts[0] += int(predicted == meta["true_label"])
                counts[1] += 1
            handle.flush()
            print(f"完成 {len(labels)}/{total}", flush=True)
    acc = sum(p == y for p, y in zip(predictions, labels)) / len(labels)
    auc = _macro_ovr_auc(labels, probabilities, dim)
    f1s = []
    class_lines = []
    for i in range(dim):
        label = str(i)
        tp = sum(p == label and y == label for p, y in zip(predictions, labels))
        fp = sum(p == label and y != label for p, y in zip(predictions, labels))
        fn = sum(p != label and y == label for p, y in zip(predictions, labels))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        class_lines.append(f"Class {i} support: {labels.count(label)}")
    text = (f"ACC: {acc:.4f}\nType error rate: {1-acc:.4f}\n"
            f"Macro one-vs-rest ROC AUC: {auc if auc is not None else 'N/A'}\n"
            f"Macro F1 (all {dim} classes): {sum(f1s)/dim:.4f}\n"
            f"Mean sequence ACC: {sum(c/n for c,n in per_sequence.values())/len(per_sequence):.4f}\n"
            f"Sequence count: {len(selected)}\nTarget count: {len(labels)}\nSeed: {args.seed}\n"
            + "\n".join(class_lines) + "\n")
    metrics_file.write_text(text, encoding="utf-8")
    print(text)
    print(f"输出目录：{output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        raise SystemExit(1)
