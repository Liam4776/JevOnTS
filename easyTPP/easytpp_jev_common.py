from __future__ import annotations

import os
import csv
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# API key 在此静态定义，修改下面的字符串即可。
API_KEY = os.environ.get("JEV_API_KEY", "YOUR API KEY")

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
    if not isinstance(API_KEY, str) or not API_KEY.strip() or API_KEY.strip() == "YOUR API KEY":
        raise RuntimeError("Fill in a valid API_KEY in easytpp_jev_common.py")
    return API_KEY.strip()


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
        })
    if len(dims) != 1:
        raise ValueError(f"Expected one dim_process value, found {sorted(dims)}")
    if not sequences or not event_count:
        raise ValueError("No next-event samples can be formed from this file")
    return sequences, dims.pop()


def _iter_sample_batches(sequences: list[dict[str, Any]], batch_size: int):
    samples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    sample_id = 0
    for sequence in sequences:
        types, gaps = sequence["types"], sequence["gaps"]
        for target_pos in range(1, len(types)):
            sample_id += 1
            history = [
                {
                    "type_event": int(types[pos]),
                    "time_since_last_event": float(gaps[pos]),
                }
                for pos in range(target_pos)
            ]
            samples.append({"sample_id": sample_id, "event_history": history})
            metadata.append({
                "seq_idx": sequence["seq_idx"],
                "target_event_index": target_pos,
                "true_label": str(types[target_pos]),
            })
            if len(samples) == batch_size:
                yield samples, metadata
                samples, metadata = [], []
    if samples:
        yield samples, metadata


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
                f"Predict the next event type for sample_id {sample_id} from its event_history. "
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


def run(cfg: dict[str, Any]) -> int:
    try:
        key = _api_key()
        sequences, dim_process = _read_sequences(cfg)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Configuration or input error: {error}", file=sys.stderr)
        return 2

    sample_count = sum(len(sequence["types"]) - 1 for sequence in sequences)

    print(
        f"Processing {sample_count} next-event samples from {cfg['data_file']} "
        f"with {dim_process} event types; batch size {BATCH_SIZE}.",
        file=sys.stderr,
    )
    task_dir = Path(__file__).resolve().parent
    output_dir = task_dir / "jev_easytpp_output"
    prediction_file = output_dir / f"{cfg['slug']}_next_event_predictions.csv"
    metrics_file = output_dir / f"{cfg['slug']}_next_event_metrics.txt"
    probability_fields = [f"probability_{event_type}" for event_type in range(dim_process)]
    fields = [
        "seq_idx", "target_event_index", "predicted_type", "confidence",
        *probability_fields, "true_type",
    ]
    predictions: list[str] = []
    labels: list[str] = []
    probability_rows: list[dict[str, float]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with prediction_file.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            completed = 0
            for batch, batch_metadata in _iter_sample_batches(sequences, BATCH_SIZE):
                batch_results = _predict(cfg, key, batch, dim_process)
                for offset, (predicted, confidence, probabilities) in enumerate(batch_results):
                    meta = batch_metadata[offset]
                    output = {
                        "seq_idx": meta["seq_idx"],
                        "target_event_index": meta["target_event_index"],
                        "predicted_type": predicted,
                        "confidence": f"{confidence:.8f}",
                        "true_type": meta["true_label"],
                    }
                    for event_type in range(dim_process):
                        output[f"probability_{event_type}"] = f"{probabilities[str(event_type)]:.8f}"
                    writer.writerow(output)
                    predictions.append(predicted)
                    labels.append(meta["true_label"])
                    probability_rows.append(probabilities)
                handle.flush()
                completed += len(batch)
                print(f"Completed {completed}/{sample_count}.", file=sys.stderr)
    except (OSError, JevAPIError) as error:
        print(f"Prediction stopped: {error}", file=sys.stderr)
        return 1

    accuracy = sum(label == pred for label, pred in zip(labels, predictions)) / len(labels)
    auc = _macro_ovr_auc(labels, probability_rows, dim_process)
    auc_text = f"{auc:.4f}" if auc is not None else "N/A"
    try:
        metrics_file.write_text(
            f"ACC: {accuracy:.4f}\nMacro one-vs-rest ROC AUC: {auc_text}\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"Cannot write metrics file: {error}", file=sys.stderr)
        return 1
    print(f"Predictions saved: {prediction_file}", file=sys.stderr)
    print(f"Metrics saved: {metrics_file}", file=sys.stderr)
    print(f"ACC: {accuracy:.4f}; Macro one-vs-rest ROC AUC: {auc_text}", file=sys.stderr)
    return 0
