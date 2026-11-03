"""汇总各数据集的 TEST ACC/AUC，仅使用 Python 标准库。

默认运行：python summarize_acc_auc.py
指定输出：python summarize_acc_auc.py --output results.csv
"""

import argparse
import csv
import re
from pathlib import Path


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def read_metrics(path):
    """读取 ACC: 数值、AUC: 数值，保留原有小数位数。"""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gb18030")
    metrics = {}
    for key in ("acc", "auc"):
        match = re.search(
            rf"^\s*{key}\s*[:：=]\s*({NUMBER})\s*$",
            content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        metrics[key] = match.group(1) if match else ""
    return metrics


def summarize(root, prefix, output):
    rows = []
    complete = 0
    # 数据集实际存放在 Multivariate_arff 子目录中。
    root = root / "Multivariate_arff" if (root / "Multivariate_arff").is_dir() else root
    for dataset_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith((".", "__")) or dataset_dir.name.lower() == "descriptions":
            continue
        dataset = dataset_dir.name
        # 例如 Coffee/jev_coffee_output/Coffee_TEST_acc_auc.txt
        result = (
            dataset_dir
            / f"{prefix}_{dataset.lower()}_output"
            / f"{dataset}_TEST_acc_auc.txt"
        )
        metrics = {"acc": "", "auc": ""}
        if not result.is_file():
            print(f"[缺失] {result}")
        else:
            try:
                metrics = read_metrics(result)
                if all(metrics.values()):
                    complete += 1
                else:
                    print(f"[格式异常] {result}：未找到完整的 ACC/AUC")
            except (OSError, UnicodeError) as exc:
                print(f"[读取失败] {result}：{exc}")
        rows.append({"dataset": dataset, **metrics})

    output.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM，便于 Excel 正确识别中文路径和数据集名称。
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "acc", "auc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入 {output}；共 {len(rows)} 个数据集，{complete} 个结果完整。")


def main():
    parser = argparse.ArgumentParser(description="按数据集汇总 TEST ACC/AUC")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="数据集根目录，默认为脚本所在目录")
    parser.add_argument("--prefix", default="jev", help="output 文件夹前缀，默认 jev")
    parser.add_argument("--output", type=Path, help="CSV 路径，默认根目录下的 <前缀>_acc_auc_summary.csv")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"数据集根目录不存在：{root}")
    output = args.output.resolve() if args.output else root / f"{args.prefix}_acc_auc_summary.csv"
    summarize(root, args.prefix, output)


if __name__ == "__main__":
    main()
