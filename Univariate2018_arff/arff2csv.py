import csv
import re
from pathlib import Path


# 输入文件夹：存放 ArrowHead 的 ARFF 文件
INPUT_DIR = Path(__file__).resolve().parent / "n" / "ArrowHead_3"

# 输出文件夹：转换后的 CSV 文件保存位置
OUTPUT_DIR = INPUT_DIR.parent / (INPUT_DIR.name + "_csv")

ATTRIBUTE_RE = re.compile(
    r"""^\s*@attribute\s+(?:'((?:\\.|[^'])*)'|"((?:\\.|[^"])*)"|(\S+))\s+(.+?)\s*$""",
    re.IGNORECASE,
)


def convert_arff_to_csv(arff_path: Path, csv_path: Path) -> None:
    column_names = []
    data_started = False

    with arff_path.open("r", encoding="utf-8-sig") as source:
        def data_lines():
            nonlocal data_started

            for line in source:
                stripped = line.strip()

                # 忽略空行和以 % 开头的注释
                if not stripped or stripped.startswith("%"):
                    continue

                if not data_started:
                    if stripped.lower() == "@data":
                        data_started = True
                    elif stripped.lower().startswith("@attribute"):
                        match = ATTRIBUTE_RE.match(stripped)
                        if not match:
                            raise ValueError(f"无法解析属性定义：{stripped}")

                        name = match.group(1) or match.group(2) or match.group(3)
                        column_names.append(name)
                    continue

                yield line

        with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)

            if not data_started:
                # 先读取数据行生成器，以解析 ARFF 文件头
                rows = csv.reader(data_lines())
            else:
                rows = csv.reader(data_lines())

            first_row = next(rows, None)  # Consume header before writing column names.
            if not data_started or not column_names:
                raise ValueError(f"Missing ARFF header: {arff_path}")
            writer.writerow(column_names)

            from itertools import chain
            for row_number, row in enumerate(chain([first_row], rows) if first_row is not None else [], start=1):
                if len(row) != len(column_names):
                    raise ValueError(
                        f"{arff_path.name} 第 {row_number} 条数据有 {len(row)} 列，"
                        f"但文件定义了 {len(column_names)} 列。"
                    )
                writer.writerow(row)

    print(f"已转换：{arff_path.name} -> {csv_path.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert univariate ARFF files to CSV")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir.parent / (input_dir.name + "_csv")
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    arff_files = list(input_dir.glob("*.arff"))
    if not arff_files:
        raise FileNotFoundError(f"输入文件夹中没有找到 ARFF 文件：{input_dir}")

    for arff_path in arff_files:
        csv_path = output_dir / f"{arff_path.stem}.csv"
        convert_arff_to_csv(arff_path, csv_path)

    print(f"全部完成。CSV 文件保存在：{output_dir}")


if __name__ == "__main__":
    main()