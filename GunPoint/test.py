"""使用 TypeSafe Jev API 对 GunPoint 测试集做零样本分类。

输入文件每行包含真实标签和 150 个经过 z 标准化的手部 X 轴位置。
程序会将全部序列放进一次 API 请求；真实标签只保存在本地，不会发送给 API。

运行：python test.py

输出：jev_zero_shot_output/GunPoint_TEST_jev_predictions.csv
"""

from __future__ import annotations

import os
import csv
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# 用户配置
# ---------------------------------------------------------------------------

# API 密钥只影响能否调用服务，不直接决定分类准确率；不要公开或提交此文件。
API_KEY = os.environ.get("JEV_API_KEY", "YOUR API KEY")

# 输入、输出文件都以当前脚本所在目录为基准。
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_FILE = SCRIPT_DIR / "GunPoint_TEST.txt"
OUTPUT_DIR = SCRIPT_DIR / "jev_zero_shot_output"
OUTPUT_FILE = OUTPUT_DIR / "GunPoint_TEST_jev_predictions.csv"

API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
# 使用的模型版本会影响模型给出的分类结果。
MODEL = "jev-1.13.0"
# GunPoint 每条序列应有 150 个测量值；更改会影响读取和请求是否正确。
SERIES_LENGTH = 150
# 每次请求最多处理 50 条序列，避免单次请求超过 API token 限制。
BATCH_SIZE = 15
# 以下参数控制每批请求的超时和失败重试。
REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 6

# 设置整数可只处理前几条做连通性检查；None 表示处理全部测试样本。
MAX_ROWS: int | None = None


# ---------------------------------------------------------------------------
# 零样本分类提示词
# ---------------------------------------------------------------------------

# 提示词描述输入和类别，并允许模型分析序列本身可见的时序特征。
CLASSIFICATION_INSTRUCTIONS = (
    "Classify this complete univariate time series as class 1 or class 2. "
    "It contains 150 ordered measurements of the X-axis position of an actor's "
    "right-hand centroid during one action. Use the sequence and the class "
    "definitions below. Compare the full trajectory, including its shape, "
    "direction, range of movement, pauses, and changes over time. Choose the "
    "class that best matches the observed trajectory. If the evidence is "
    "ambiguous, select the more likely class and reflect the uncertainty in "
    "your confidence. Class numbers are identifiers only."
)

# 两个类别的文字定义会直接影响分类；返回的 1/2 应与数据集标签一致。
CLASSIFICATION_CRITERIA = {
    "1": (
        "Gun-Draw: The actor starts with the hand at their side, draws a replica "
        "gun from a hip-mounted holster, points it at a target for about one "
        "second, returns the gun to the holster, and brings the hand back to the side."
    ),
    "2": (
        "Point: The actor starts with the hand at their side, points at a target "
        "with the index finger for about one second, and brings the hand back to the side."
    ),
}


class JevAPIError(RuntimeError):
    """Raised when a Jev request cannot be completed safely."""


def load_test_data(path: Path) -> tuple[list[list[float]], list[int]]:
    """读取序列和本地真实标签；真实标签不会混入发给模型的请求。"""

    if not path.is_file():
        raise FileNotFoundError(f"Test file does not exist: {path}")

    sequences: list[list[float]] = []
    true_labels: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != SERIES_LENGTH + 1:
                raise ValueError(
                    f"Line {line_number} has {len(fields)} fields; expected "
                    f"one label plus {SERIES_LENGTH} measurements."
                )

            try:
                hidden_label = float(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number} has a non-numeric label."
                ) from error
            if hidden_label not in (1.0, 2.0):
                raise ValueError(
                    f"Line {line_number} label must be 1 or 2."
                )

            try:
                sequence = [float(value) for value in fields[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number} contains a non-numeric measurement."
                ) from error
            if any(not math.isfinite(value) for value in sequence):
                raise ValueError(
                    f"Line {line_number} contains a non-finite measurement."
                )

            sequences.append(sequence)
            true_labels.append(int(hidden_label))

    if not sequences:
        raise ValueError("The test file contains no samples.")
    return sequences, true_labels


def make_batch_payload(sequences: Sequence[Sequence[float]]) -> dict[str, Any]:
    """构造单次批量请求；只发送序列，不发送真实标签。"""
    if not sequences:
        raise ValueError("Cannot create a batch request without samples.")

    samples = []
    questions = {}
    choice_names = {"1": "类别 1（拔枪）", "2": "类别 2（指向）"}
    for sample_id, sequence in enumerate(sequences, start=1):
        if len(sequence) != SERIES_LENGTH:
            raise ValueError(
                f"Sample {sample_id} has {len(sequence)} measurements; "
                f"expected {SERIES_LENGTH}."
            )
        samples.append(
            {"sample_id": sample_id, "x_axis_sequence": list(sequence)}
        )
        question_id = f"sample_{sample_id:03d}"
        questions[question_id] = {
            "type": "choice",
            "instructions": (
                f"请只对 state.samples 中 sample_id 为 {sample_id} 的序列分类。"
                "请遵循 state.task_instructions 和 state.class_definitions。"
            ),
            "criteria": choice_names,
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
    """一次调用模型处理整批序列，逐题返回类别、置信度和类别 2 概率。"""

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
                "User-Agent": "jev-gunpoint-zero-shot/1.0",
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                response_json = json.loads(response.read().decode("utf-8"))
                answers = response_json["answers"]
                batch_results = []
                for sample_id in range(1, len(sequences) + 1):
                    question_id = f"sample_{sample_id:03d}"
                    answer = answers[question_id]
                    selected_class = str(answer["choice"]).strip()
                    confidence = float(answer["confidence"])
                    class_two_probability = float(answer["probabilities"]["2"])
                    if selected_class not in CLASSIFICATION_CRITERIA:
                        raise JevAPIError(
                            f"Jev returned an invalid class for {question_id}: "
                            f"{selected_class!r}"
                        )
                    if not 0.0 <= confidence <= 1.0:
                        raise JevAPIError(
                            f"Jev returned invalid confidence for {question_id}: "
                            f"{confidence}"
                        )
                    if not 0.0 <= class_two_probability <= 1.0:
                        raise JevAPIError(
                            f"Jev returned an invalid class-2 probability for "
                            f"{question_id}: {class_two_probability}"
                        )
                    batch_results.append(
                        (int(selected_class), confidence, class_two_probability)
                    )
                return batch_results

        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")[:1000]
            retryable = error.code == 429 or 500 <= error.code <= 599
            if retryable and attempt < MAX_RETRIES:
                time.sleep(retry_delay_seconds(error, attempt))
                continue
            raise JevAPIError(
                f"TypeSafe API returned HTTP {error.code}: {error_body}"
            ) from error

        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as error:
            if attempt < MAX_RETRIES:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise JevAPIError(
                f"Network request failed after retries: {error}"
            ) from error

        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise JevAPIError(
                f"Unexpected TypeSafe response shape: {error}"
            ) from error

    raise JevAPIError("Jev request failed without a response.")


def validate_configuration() -> None:
    """检查密钥和序列长度等运行配置。"""
    if not API_KEY.strip() or (API_KEY.startswith("PASTE_") or API_KEY == "YOUR API KEY"):
        raise ValueError(
            "Set API_KEY at the top of gunpoint_jev_zero_shot.py before running."
        )
    if SERIES_LENGTH != 150:
        raise ValueError("SERIES_LENGTH must remain 150 for GunPoint.")


def calculate_accuracy(true_labels: Sequence[int], predictions: Sequence[int]) -> float:
    """计算分类准确率（ACC）。"""
    if not true_labels:
        raise ValueError("Cannot calculate accuracy without samples.")
    correct = sum(actual == predicted for actual, predicted in zip(true_labels, predictions))
    return correct / len(true_labels)


def calculate_binary_auc(true_labels: Sequence[int], positive_scores: Sequence[float]) -> float | None:
    """按类别 2 为正类，用类别概率计算 ROC AUC；单类别时返回 None。"""
    positive_ranks = [
        (score, label == 2)
        for label, score in zip(true_labels, positive_scores)
    ]
    positives = sum(is_positive for _, is_positive in positive_ranks)
    negatives = len(positive_ranks) - positives
    if positives == 0 or negatives == 0:
        return None

    # AUC 等于正类样本在所有样本排序中的平均秩转换值；并列分数取平均秩。
    positive_ranks.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(positive_ranks):
        end = index + 1
        while end < len(positive_ranks) and positive_ranks[end][0] == positive_ranks[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(
            1 for _, is_positive in positive_ranks[index:end] if is_positive
        )
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main() -> int:
    """读取测试数据、一次性批量预测，并保存结果和本地真实标签。"""
    try:
        validate_configuration()
        sequences, true_labels = load_test_data(TEST_FILE)
    except (OSError, ValueError) as error:
        print(f"Configuration/input error: {error}", file=sys.stderr)
        return 2

    if MAX_ROWS is not None:
        sequences = sequences[:MAX_ROWS]
        true_labels = true_labels[:MAX_ROWS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Submitting {len(sequences)} GunPoint samples in batches of {BATCH_SIZE}...",
        file=sys.stderr,
    )
    try:
        batch_results = []
        for start in range(0, len(sequences), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sequences))
            batch_results.extend(call_jev_batch(sequences[start:end]))
            print(
                f"Completed {end}/{len(sequences)} samples.",
                file=sys.stderr,
            )

        predictions = [result[0] for result in batch_results]
        # ChoiceAnswer 提供各类别概率；直接使用类别 2 的概率计算 ROC AUC。
        positive_scores = [result[2] for result in batch_results]

        with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "predicted_class",
                    "confidence",
                    "probability_class_2",
                    "true_label",
                ],
            )
            writer.writeheader()

            for sample_position, (predicted_class, confidence, class_two_probability) in enumerate(
                batch_results, start=1
            ):
                writer.writerow(
                    {
                        "predicted_class": predicted_class,
                        "confidence": f"{confidence:.8f}",
                        "probability_class_2": f"{class_two_probability:.8f}",
                        "true_label": true_labels[sample_position - 1],
                    }
                )

    except (OSError, JevAPIError) as error:
        print(f"Classification stopped: {error}", file=sys.stderr)
        return 1

    print(f"Received {len(batch_results)} predictions.", file=sys.stderr)
    print(f"Finished. Predictions saved to: {OUTPUT_FILE}", file=sys.stderr)
    accuracy = calculate_accuracy(true_labels, predictions)
    auc = calculate_binary_auc(true_labels, positive_scores)
    print(f"ACC: {accuracy:.4f} ({accuracy * 100:.2f}%)", file=sys.stderr)
    if auc is None:
        print("ROC AUC: N/A (评估样本中必须同时包含类别 1 和类别 2)", file=sys.stderr)
    else:
        print(f"ROC AUC (类别 2 为正类): {auc:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
