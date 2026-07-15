import argparse
import re
from pathlib import Path


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_sample_eval_log(log_path: Path) -> list[dict]:
    results = []
    current_result = None

    step_pattern = re.compile(
        r"Running sample_eval with sampling\.steps=(\d+)"
    )
    gen_ppl_pattern = re.compile(
        rf"Generative perplexity:\s*"
        rf"(?:tensor\()?"
        rf"({FLOAT_PATTERN})"
    )
    entropy_pattern = re.compile(
        rf"Generation entropy:\s*({FLOAT_PATTERN})"
    )
    distinct4_pattern = re.compile(
        rf"Distinct-4:\s*({FLOAT_PATTERN})"
    )

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            step_match = step_pattern.search(line)

            if step_match:
                # 儲存上一組 sampling.steps 的結果
                if current_result is not None:
                    results.append(current_result)

                current_result = {
                    "sampling_steps": int(step_match.group(1)),
                    "generative_perplexity": None,
                    "generation_entropy": None,
                    "distinct_4": None,
                }
                continue

            if current_result is None:
                continue

            gen_ppl_match = gen_ppl_pattern.search(line)
            if gen_ppl_match:
                current_result["generative_perplexity"] = float(
                    gen_ppl_match.group(1)
                )
                continue

            entropy_match = entropy_pattern.search(line)
            if entropy_match:
                current_result["generation_entropy"] = float(
                    entropy_match.group(1)
                )
                continue

            distinct4_match = distinct4_pattern.search(line)
            if distinct4_match:
                current_result["distinct_4"] = float(
                    distinct4_match.group(1)
                )

    # 儲存最後一組結果
    if current_result is not None:
        results.append(current_result)

    return results


def format_value(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_results(results: list[dict]) -> None:
    if not results:
        print("找不到任何 sample_eval 結果。")
        return

    results = sorted(
        results,
        key=lambda item: item["sampling_steps"],
    )

    headers = [
        "Sampling Steps",
        "Generative Perplexity",
        "Generation Entropy",
        "Distinct-4",
    ]

    rows = []

    for result in results:
        rows.append([
            str(result["sampling_steps"]),
            format_value(result["generative_perplexity"]),
            format_value(result["generation_entropy"]),
            format_value(result["distinct_4"]),
        ])

    widths = []

    for column_index, header in enumerate(headers):
        widths.append(
            max(
                len(header),
                max(len(row[column_index]) for row in rows),
            )
        )

    separator = "+-" + "-+-".join(
        "-" * width for width in widths
    ) + "-+"

    print(separator)

    print(
        "| "
        + " | ".join(
            f"{header:<{width}}"
            for header, width in zip(headers, widths)
        )
        + " |"
    )

    print(separator)

    for row in rows:
        print(
            "| "
            + " | ".join(
                f"{value:>{width}}"
                for value, width in zip(row, widths)
            )
            + " |"
        )

    print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "依 sampling.steps 擷取 Generative perplexity、"
            "Generation entropy 與 Distinct-4。"
        )
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="sample_eval log 檔案路徑",
    )

    args = parser.parse_args()

    if not args.log_file.is_file():
        raise FileNotFoundError(
            f"找不到 log 檔案：{args.log_file}"
        )

    results = parse_sample_eval_log(args.log_file)
    print_results(results)


if __name__ == "__main__":
    main()