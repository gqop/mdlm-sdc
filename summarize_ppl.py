import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path


# 支援一般小數與科學記號，例如：
# 35.2366
# 1.23e-4
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_log(log_path: Path) -> dict[str, dict[int, float]]:
    """
    解析評估 log，回傳格式：

    {
        "openwebtext-split": {
            1: 35.236,
            2: 35.256,
            ...
        },
        ...
    }
    """
    results: dict[str, dict[int, float]] = defaultdict(dict)

    current_dataset: str | None = None
    current_seed: int | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # 例如：DATASET=openwebtext-split
            dataset_match = re.match(r"^DATASET=(.+)$", line)
            if dataset_match:
                current_dataset = dataset_match.group(1).strip()
                current_seed = None
                continue

            # 例如：SEED=1
            seed_match = re.match(r"^SEED=(\d+)$", line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                continue

            # 例如：
            # val/ppl          │     35.23662124270482     │
            ppl_match = re.search(
                rf"val/ppl.*?({FLOAT_PATTERN})",
                line,
            )

            if ppl_match:
                if current_dataset is None or current_seed is None:
                    print(
                        f"[警告] 第 {line_number} 行找到 val/ppl，"
                        "但尚未找到對應的 DATASET 或 SEED。"
                    )
                    continue

                ppl = float(ppl_match.group(1))
                results[current_dataset][current_seed] = ppl

    return results


def print_summary(results: dict[str, dict[int, float]]) -> None:
    if not results:
        print("找不到任何 DATASET、SEED 與 val/ppl 結果。")
        return

    rows = []

    for dataset, seed_results in results.items():
        sorted_results = sorted(seed_results.items())
        values = [value for _, value in sorted_results]

        mean_value = statistics.mean(values)
        std_value = (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        )

        seed_values = ", ".join(
            f"{seed}:{value:.3f}"
            for seed, value in sorted_results
        )

        summary = f"{mean_value:.3f} ± {std_value:.3f}"

        rows.append(
            (
                dataset,
                str(len(values)),
                seed_values,
                summary,
            )
        )

    # 根據實際內容自動計算欄位寬度
    dataset_width = max(
        len("DATASET"),
        max(len(row[0]) for row in rows),
    )

    seeds_width = max(
        len("Seeds"),
        max(len(row[1]) for row in rows),
    )

    values_width = max(
        len("各 Seed 的 val/ppl"),
        max(len(row[2]) for row in rows),
    )

    summary_width = max(
        len("平均值 ± 標準差"),
        max(len(row[3]) for row in rows),
    )

    separator_length = (
        dataset_width
        + seeds_width
        + values_width
        + summary_width
        + 6
    )

    print("=" * separator_length)

    print(
        f"{'DATASET':<{dataset_width}}  "
        f"{'Seeds':<{seeds_width}}  "
        f"{'各 Seed 的 val/ppl':<{values_width}}  "
        f"{'平均值 ± 標準差':<{summary_width}}"
    )

    print("=" * separator_length)

    for dataset, seeds, seed_values, summary in rows:
        print(
            f"{dataset:<{dataset_width}}  "
            f"{seeds:<{seeds_width}}  "
            f"{seed_values:<{values_width}}  "
            f"{summary:<{summary_width}}"
        )

    print("=" * separator_length)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="依 DATASET 與 SEED 統計 val/ppl 的平均值與標準差。"
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="評估 log 檔案路徑",
    )
    args = parser.parse_args()

    if not args.log_file.is_file():
        raise FileNotFoundError(f"找不到檔案：{args.log_file}")

    results = parse_log(args.log_file)
    print_summary(results)


if __name__ == "__main__":
    main()