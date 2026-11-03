"""汇总 easyTPP 各 jev_<任务名>_sampled_output 的 ACC / AUC。

运行：python summarize_acc_auc.py
输出：脚本所在目录/jev_acc_auc_summary.csv
可选：--root 数据目录 --output CSV路径
仅使用 Python 标准库，ACC 取总体 ACC，不取 Mean sequence ACC。
"""
import argparse
import csv
import re
from pathlib import Path

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def read_metrics(path):
    text = path.read_text(encoding="utf-8-sig")
    values = []
    for label in ("ACC", "Macro one-vs-rest ROC AUC"):
        match = re.search(
            rf"^\s*{re.escape(label)}\s*[:：]\s*({NUMBER}|N/A|NaN)\s*$",
            text, re.MULTILINE | re.IGNORECASE,
        )
        values.append(match.group(1) if match else "")
    return values


def summarize(root, output):
    rows = []
    for folder in sorted(root.glob("jev_*_sampled_output"), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue
        dataset = folder.name[len("jev_"):-len("_sampled_output")]
        path = folder / f"{dataset}_next_event_metrics.txt"
        acc, auc = "", ""
        if not path.is_file():
            print(f"[缺失] {path}")
        else:
            try:
                acc, auc = read_metrics(path)
                if not acc or not auc:
                    print(f"[指标不完整] {path}")
            except (OSError, UnicodeError) as error:
                print(f"[读取失败] {path}: {error}")
        rows.append((dataset, acc, auc))
    if not rows:
        raise ValueError(f"未找到 jev_<任务名>_sampled_output 目录：{root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "acc", "auc"])
        writer.writerows(rows)
    print(f"已汇总 {len(rows)} 个任务：{output}")
    for row in rows:
        print(", ".join(row))


def main():
    parser = argparse.ArgumentParser(description="汇总 easyTPP 各任务的总体 ACC 和宏平均 AUC")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")
    output = args.output.resolve() if args.output else root / "jev_acc_auc_summary.csv"
    summarize(root, output)


if __name__ == "__main__":
    main()
