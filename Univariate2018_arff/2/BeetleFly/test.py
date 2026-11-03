"""使用 Jev 对 BeetleFly 测试集进行零样本批量分类。"""

from __future__ import annotations

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
from typing import Any, Sequence


# 输入和输出路径
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_FILE = SCRIPT_DIR / "BeetleFly_TEST.arff"
OUTPUT_DIR = SCRIPT_DIR / "jev_beetlefly_output"
OUTPUT_FILE = OUTPUT_DIR / "BeetleFly_TEST_jev_predictions.csv"

API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
SERIES_LENGTH = 512
BATCH_SIZE = 5
REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 6
MAX_ROWS: int | None = None


def load_api_key() -> str:
    env_key = os.environ.get("JEV_API_KEY", "").strip()
    if env_key:
        if env_key == "YOUR API KEY":
            raise ValueError("Set JEV_API_KEY to a real API key")
        return env_key
    key_file = SCRIPT_DIR.parents[1] / 'api_key.json'
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


API_KEY = "YOUR API KEY"


# BeetleFly sequences describe insect outlines; classify from the full shape profile.
CLASSIFICATION_INSTRUCTIONS = (
    "Determine whether this sequence represents the outline of a beetle or a fly. "
    "It contains 512 ordered values, each measuring the distance from a point on "
    "the outline boundary to the center of the outline. Use the shape profile "
    "across the complete sequence to choose the matching class."
)

# UCR label 1 corresponds to Beetle; label 2 corresponds to Fly.
CLASSIFICATION_CRITERIA = {
    "1": "The outline distance sequence belongs to a beetle.",
    "2": "The outline distance sequence belongs to a fly.",
}


class JevAPIError(RuntimeError):
    """Jev 请求或响应异常。"""


def iter_arff_data(path: Path):
    """逐行读取 ARFF 的 @data 部分，跳过头部、注释和空行。"""
    in_data = False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if not in_data:
                if stripped.lower() == "@data":
                    in_data = True
                continue
            yield line
    if not in_data:
        raise ValueError(f"ARFF 文件缺少 @data 标记：{path}")


def load_test_data(path: Path) -> tuple[list[list[float]], list[int]]:
    """读取 ARFF 测试集；标签在最后一列，只保存在本地。"""
    if not path.is_file():
        raise FileNotFoundError(f"测试文件不存在：{path}")

    sequences: list[list[float]] = []
    true_labels: list[int] = []
    for row_number, row in enumerate(csv.reader(iter_arff_data(path)), start=1):
        if len(row) != SERIES_LENGTH + 1:
            raise ValueError(
                f"第 {row_number} 行有 {len(row)} 列；"
                f"预期为 {SERIES_LENGTH} 个测量值和 1 个标签。"
            )
        try:
            label_value = float(row[-1].strip().strip("'\""))
            sequence = [float(value) for value in row[:-1]]
        except ValueError as error:
            raise ValueError(f"第 {row_number} 行包含非数值字段。") from error

        if label_value not in (1.0, 2.0):
            raise ValueError(f"第 {row_number} 行的标签必须为 1 或 2。")
        if any(not math.isfinite(value) for value in sequence):
            raise ValueError(f"第 {row_number} 行包含非有限测量值。")

        sequences.append(sequence)
        true_labels.append(int(label_value))

    if not sequences:
        raise ValueError("测试文件中没有样本。")
    return sequences, true_labels


def make_batch_payload(sequences: Sequence[Sequence[float]]) -> dict[str, Any]:
    """生成一批请求；只发送序列，不发送真实标签。"""
    if not sequences:
        raise ValueError("批次中没有样本。")

    samples = []
    questions = {}
    for sample_id, sequence in enumerate(sequences, start=1):
        if len(sequence) != SERIES_LENGTH:
            raise ValueError(
                f"样本 {sample_id} 有 {len(sequence)} 个测量值，"
                f"预期为 {SERIES_LENGTH} 个。"
            )
        samples.append(
            {"sample_id": sample_id, "outline_distance_sequence": list(sequence)}
        )
        question_id = f"sample_{sample_id:03d}"
        questions[question_id] = {
            "type": "choice",
            "instructions": (
                f"请只对 state.samples 中 sample_id 为 {sample_id} 的序列分类。"
                "遵循 state.task_instructions 和 state.class_definitions。"
            ),
            "criteria": CLASSIFICATION_CRITERIA,
        }

    return {
        "state": {
            "task_instructions": CLASSIFICATION_INSTRUCTIONS,
            "class_definitions": CLASSIFICATION_CRITERIA,
            "samples": samples,
        },
        "model": MODEL,
        "questions": questions,
    }


def retry_delay_seconds(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(30.0, 2.0**attempt)


def call_jev_batch(
    sequences: Sequence[Sequence[float]],
) -> list[tuple[int, float, float]]:
    """发送一批样本，返回预测标签、置信度和类别 2（Fly）的概率。"""
    encoded = json.dumps(
        make_batch_payload(sequences), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(
            API_ENDPOINT,
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "jev-beetlefly-zero-shot/1.0",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                response_json = json.loads(response.read().decode("utf-8"))
                answers = response_json["answers"]
                results = []
                for sample_id in range(1, len(sequences) + 1):
                    question_id = f"sample_{sample_id:03d}"
                    answer = answers[question_id]
                    predicted = str(answer["choice"]).strip()
                    confidence = float(answer["confidence"])
                    probability_fly = float(answer["probabilities"]["2"])
                    if predicted not in CLASSIFICATION_CRITERIA:
                        raise JevAPIError(
                            f"模型为 {question_id} 返回了无效类别：{predicted!r}"
                        )
                    if not 0.0 <= confidence <= 1.0:
                        raise JevAPIError(f"{question_id} 的置信度超出范围。")
                    if not 0.0 <= probability_fly <= 1.0:
                        raise JevAPIError(f"{question_id} 的类别 2 概率超出范围。")
                    results.append((int(predicted), confidence, probability_fly))
                return results

        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:1000]
            if error.code == 400 and "max_tokens_exceeded" in body:
                raise JevAPIError(
                    f"max_tokens_exceeded: 当前批次含 {len(sequences)} 条序列。"
                ) from error
            retryable = error.code == 429 or 500 <= error.code <= 599
            if retryable and attempt < MAX_RETRIES:
                time.sleep(retry_delay_seconds(error, attempt))
                continue
            raise JevAPIError(f"TypeSafe API 返回 HTTP {error.code}: {body}") from error

        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as error:
            if attempt < MAX_RETRIES:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise JevAPIError(f"网络请求重试后仍失败：{error}") from error

        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise JevAPIError(f"TypeSafe 响应格式异常：{error}") from error

    raise JevAPIError("Jev 请求未能获得响应。")


def predict_with_token_fallback(
    sequences: Sequence[Sequence[float]],
) -> list[tuple[int, float, float]]:
    """批次超过 token 限制时自动拆半，直到接口接受。"""
    try:
        return call_jev_batch(sequences)
    except JevAPIError as error:
        if "max_tokens_exceeded" not in str(error) or len(sequences) <= 1:
            raise
        midpoint = len(sequences) // 2
        print(
            f"批次 {len(sequences)} 条超过 token 限制，自动拆为 "
            f"{midpoint} 条和 {len(sequences) - midpoint} 条。",
            file=sys.stderr,
        )
        return predict_with_token_fallback(sequences[:midpoint]) + predict_with_token_fallback(
            sequences[midpoint:]
        )


def calculate_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """计算准确率。"""
    if not labels:
        raise ValueError("没有样本，无法计算 ACC。")
    return sum(a == p for a, p in zip(labels, predictions)) / len(labels)


def calculate_binary_auc(
    labels: Sequence[int], positive_scores: Sequence[float]
) -> float | None:
    """以类别 2（Fly）为正类，用概率计算 ROC AUC。"""
    ranked = sorted(zip(positive_scores, labels), key=lambda item: item[0])
    positives = sum(label == 2 for _, label in ranked)
    negatives = len(ranked) - positives
    if positives == 0 or negatives == 0:
        return None

    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(
            1 for _, label in ranked[index:end] if label == 2
        )
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main() -> int:
    global API_KEY
    try:
        API_KEY = load_api_key()
        if not API_KEY.strip():
            raise ValueError("API_KEY 为空。")
        sequences, true_labels = load_test_data(TEST_FILE)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"配置或输入错误：{error}", file=sys.stderr)
        return 2

    if MAX_ROWS is not None:
        sequences = sequences[:MAX_ROWS]
        true_labels = true_labels[:MAX_ROWS]

    if BATCH_SIZE < 1:
        print("BATCH_SIZE 必须大于 0。", file=sys.stderr)
        return 2

    print(
        f"将处理 {len(sequences)} 条 BeetleFly 测试序列，每批最多 {BATCH_SIZE} 条。",
        file=sys.stderr,
    )
    try:
        results = []
        for start in range(0, len(sequences), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sequences))
            results.extend(predict_with_token_fallback(sequences[start:end]))
            print(f"已完成 {end}/{len(sequences)} 条。", file=sys.stderr)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "predicted_class",
                    "predicted_name",
                    "confidence",
                    "probability_fly",
                    "true_label",
                    "true_name",
                ],
            )
            writer.writeheader()
            for (predicted, confidence, probability_fly), true_label in zip(
                results, true_labels
            ):
                writer.writerow(
                    {
                        "predicted_class": predicted,
                        "predicted_name": "Beetle" if predicted == 1 else "Fly",
                        "confidence": f"{confidence:.8f}",
                        "probability_fly": f"{probability_fly:.8f}",
                        "true_label": true_label,
                        "true_name": "Beetle" if true_label == 1 else "Fly",
                    }
                )
    except (OSError, JevAPIError) as error:
        print(f"分类中止：{error}", file=sys.stderr)
        return 1

    predictions = [result[0] for result in results]
    fly_probabilities = [result[2] for result in results]
    accuracy = calculate_accuracy(true_labels, predictions)
    auc = calculate_binary_auc(true_labels, fly_probabilities)
    metrics_file = OUTPUT_DIR / "BeetleFly_TEST_acc_auc.txt"
    auc_value = f"{auc:.4f}" if auc is not None else "N/A"
    try:
        metrics_file.write_text(f"ACC: {accuracy:.4f}\nAUC: {auc_value}\n", encoding="utf-8")
    except OSError as error:
        print(f"ACC/AUC 文件写入失败：{error}", file=sys.stderr)
        return 1
    print(f"ACC/AUC 已保存：{metrics_file}", file=sys.stderr)
    print(f"预测结果已保存：{OUTPUT_FILE}", file=sys.stderr)
    print(f"ACC: {accuracy:.4f} ({accuracy * 100:.2f}%)", file=sys.stderr)
    if auc is None:
        print("ROC AUC: N/A（评估数据必须同时包含两个类别）", file=sys.stderr)
    else:
        print(f"ROC AUC（Fly 为正类）: {auc:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
