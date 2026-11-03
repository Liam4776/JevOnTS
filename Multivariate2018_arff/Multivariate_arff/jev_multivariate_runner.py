from __future__ import annotations
import csv, http.client, json, math, os, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
BATCH_SIZE = 5
MAX_PARALLEL_BATCHES = 4
TIMEOUT = 180
MAX_RETRIES = 6
API_KEY_FILE = Path(__file__).resolve().parent / "api_key.json"
csv.field_size_limit(300_000_000)

class JevAPIError(RuntimeError):
    pass

def _api_keys(root: Path) -> list[str]:
    primary = os.environ.get("JEV_API_KEY")
    if not primary or not primary.strip():
        if not API_KEY_FILE.is_file():
            raise RuntimeError(f"API key JSON file not found: {API_KEY_FILE}")
        try:
            primary = json.loads(API_KEY_FILE.read_text(encoding="utf-8-sig")).get("api_key")
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            raise RuntimeError(f"Cannot read API key file {API_KEY_FILE}: {error}") from error
    if not isinstance(primary, str) or not primary.strip() or primary.strip() == "YOUR API KEY":
        raise RuntimeError(f"No valid api_key field in {API_KEY_FILE}")
    keys = [primary.strip()]
    fallback = os.environ.get("JEV_FALLBACK_API_KEY")
    if fallback and fallback.strip() and fallback.strip() not in keys:
        keys.append(fallback.strip())
    return keys

def _header(path: Path) -> list[str]:
    header = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.strip().lower() == "@data":
                for item in header:
                    match = re.match(r"^\s*@attribute\s+\S+\s+\{(.*)\}\s*$", item, re.I)
                    if match:
                        labels = [x.strip().strip("'\"") for x in match.group(1).split(",")]
                if not locals().get("labels"):
                    raise ValueError(f"No nominal class list in {path}")
                return labels
            header.append(line.rstrip("\r\n"))
    raise ValueError(f"Missing @data in {path}")

def _records(path: Path, class_labels: list[str]):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        active = False
        expected_shape = None
        sample_number = 0
        for line in stream:
            stripped = line.strip()
            if not active:
                if stripped.lower() == "@data":
                    active = True
                continue
            if not stripped or stripped.startswith("%"):
                continue
            row = next(csv.reader([line], delimiter=",", quotechar="'", strict=False))
            if len(row) < 2:
                raise ValueError(f"Malformed row {sample_number + 1}: missing class label")
            label = row[-1].strip().strip("'\"")
            if label not in class_labels:
                raise ValueError(f"Unexpected class {label!r} in row {sample_number + 1}")
            channels = []
            for dim, text in enumerate(row[0].split("\\n"), 1):
                values = []
                for token in text.split(","):
                    token = token.strip().strip("'\"")
                    if not token or token == "?" or token.lower() == "nan":
                        values.append(None)
                    else:
                        try:
                            value = float(token)
                        except ValueError as error:
                            raise ValueError(f"Invalid number at sample {sample_number+1}, dimension {dim}: {token!r}") from error
                        values.append(value if math.isfinite(value) else None)
                channels.append(values)
            lengths = {len(x) for x in channels}
            if not channels or not lengths or len(lengths) != 1 or not next(iter(lengths)):
                raise ValueError(f"Invalid channel lengths in sample {sample_number+1}: {sorted(lengths)}")
            shape = (len(channels), len(channels[0]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"Sample {sample_number+1} has shape {shape}, expected {expected_shape}")
            sample_number += 1
            yield channels, label, shape
        if not active or sample_number == 0:
            raise ValueError(f"No test samples found in {path}")

def _criteria(cfg, labels):
    descriptions = cfg.get("label_descriptions", {})
    names = cfg.get("label_names", {})
    return {label: descriptions.get(label, f"Class label {label} ({names.get(label, label)}); choose it when the complete multivariate pattern best matches this category.") for label in labels}

def _payload(cfg, batch, labels):
    criteria, samples, questions = _criteria(cfg, labels), [], {}
    for i, (channels, _, _) in enumerate(batch, 1):
        samples.append({"sample_id": i, "multivariate_measurements": {"dimensions": channels, "time_steps": len(channels[0])}})
        qid = f"sample_{i:03d}"
        questions[qid] = {"type": "choice", "instructions": f"Classify only sample_id {i} in state.samples using state.task_instructions and state.class_definitions.", "criteria": criteria}
    return {"state": {"task_instructions": cfg["prompt"], "class_definitions": criteria, "samples": samples}, "model": MODEL, "questions": questions}

def _numeric_alias(a, b):
    try:
        return float(str(a)) == float(str(b))
    except (ValueError, TypeError):
        return False

def _request(cfg, key, batch, labels):
    body = json.dumps(_payload(cfg, batch, labels), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(API_ENDPOINT, data=body, method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                answers = json.loads(response.read().decode("utf-8"))["answers"]
            results = []
            for i in range(1, len(batch) + 1):
                answer = answers[f"sample_{i:03d}"]
                predicted, confidence = str(answer["choice"]).strip(), float(answer["confidence"])
                raw_scores, scores = answer["probabilities"], {}
                if predicted not in labels or not 0 <= confidence <= 1:
                    raise JevAPIError(f"Invalid class or confidence for sample {i}")
                for label in labels:
                    key_match = label if label in raw_scores else next((k for k in raw_scores if _numeric_alias(k, label)), None)
                    if key_match is None:
                        raise JevAPIError(f"Missing probability for class {label!r}, sample {i}")
                    score = float(raw_scores[key_match])
                    if not 0 <= score <= 1:
                        raise JevAPIError(f"Invalid probability for class {label!r}, sample {i}")
                    scores[label] = score
                results.append((predicted, confidence, scores))
            return results
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1200]
            if error.code == 400 and "max_tokens_exceeded" in detail:
                raise JevAPIError(f"max_tokens_exceeded for batch size {len(batch)}") from error
            if (error.code == 429 or error.code >= 500) and attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt)); continue
            raise JevAPIError(f"TypeSafe API HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            if attempt < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt)); continue
            raise JevAPIError(f"Network retries exhausted: {error}") from error
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, JevAPIError): raise
            raise JevAPIError(f"Unexpected API response: {error}") from error
    raise JevAPIError("Jev request failed without a response")

def _predict(cfg, keys, batch, labels):
    token_error = None
    for index, key in enumerate(keys):
        try:
            return _request(cfg, key, batch, labels)
        except JevAPIError as error:
            if "max_tokens_exceeded" not in str(error):
                raise
            token_error = error
            if index + 1 < len(keys):
                print("Primary key reached its token limit; retrying with the fallback key.", file=sys.stderr)
    if len(batch) <= 1:
        raise token_error or JevAPIError("Token limit exceeded for a single sample.")
    middle = len(batch) // 2
    print(f"Token limit: splitting batch {len(batch)} into {middle} and {len(batch)-middle}.", file=sys.stderr)
    return _predict(cfg, keys, batch[:middle], labels) + _predict(cfg, keys, batch[middle:], labels)

def _auc(labels, scores, positive):
    ranked = sorted(zip(scores, labels)); p = sum(x == positive for x in labels); n = len(labels) - p
    if not p or not n: return None
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]: j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y == positive for _, y in ranked[i:j]); i = j
    return (rank_sum - p * (p + 1) / 2) / (p * n)

def _macro_auc(true_labels, score_rows, labels):
    values = []
    for label in labels:
        auc = _auc([label if x == label else "__other__" for x in true_labels], [row[label] for row in score_rows], label)
        if auc is not None: values.append(auc)
    return sum(values) / len(values) if values else None

def run(cfg: dict[str, Any]) -> int:
    root = Path(__file__).resolve().parent
    folder = root / cfg["task"]
    source = folder / f"{cfg['task']}_TEST.arff"
    out_dir = folder / f"jev_{cfg['slug']}_output"
    out_csv = out_dir / f"{cfg['task']}_TEST_jev_predictions.csv"
    try:
        keys = _api_keys(root); labels = _header(source)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Input/configuration error: {error}", file=sys.stderr); return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["predicted_class", "predicted_name", "confidence", *[f"probability::{x}" for x in labels], "true_label", "true_name"]
    true_labels, predicted_labels, score_rows = [], [], []
    try:
        with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
            source_records = iter(_records(source, labels)); completed = 0; shape = None
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
                while True:
                    pending = []
                    for _ in range(MAX_PARALLEL_BATCHES):
                        batch = []
                        try:
                            for _ in range(BATCH_SIZE): batch.append(next(source_records))
                        except StopIteration:
                            pass
                        if batch:
                            shape = batch[0][2] if shape is None else shape
                            pending.append((batch, executor.submit(_predict, cfg, keys, batch, labels)))
                    if not pending: break
                    for batch, future in pending:
                        answers = future.result()
                        for (predicted, confidence, scores), (_, actual, _) in zip(answers, batch):
                            writer.writerow({"predicted_class": predicted, "predicted_name": cfg.get("label_names", {}).get(predicted, predicted), "confidence": f"{confidence:.8f}", **{f"probability::{x}": f"{scores[x]:.8f}" for x in labels}, "true_label": actual, "true_name": cfg.get("label_names", {}).get(actual, actual)})
                            true_labels.append(actual); predicted_labels.append(predicted); score_rows.append(scores)
                        completed += len(batch); print(f"{cfg['task']}: completed {completed} samples; shape {shape}.", file=sys.stderr)
    except (OSError, ValueError, csv.Error, JevAPIError) as error:
        print(f"Classification stopped: {error}", file=sys.stderr); return 1
    acc = sum(a == p for a, p in zip(true_labels, predicted_labels)) / len(true_labels)
    auc = _macro_auc(true_labels, score_rows, labels)
    out_txt = out_dir / f"{cfg['task']}_TEST_acc_auc.txt"
    auc_text = f"{auc:.4f}" if auc is not None else "N/A"
    try: out_txt.write_text(f"ACC: {acc:.4f}\nAUC: {auc_text}\n", encoding="utf-8")
    except OSError as error:
        print(f"Could not write metrics file: {error}", file=sys.stderr); return 1
    print(f"Predictions: {out_csv}", file=sys.stderr); print(f"Metrics: {out_txt}", file=sys.stderr)
    return 0
