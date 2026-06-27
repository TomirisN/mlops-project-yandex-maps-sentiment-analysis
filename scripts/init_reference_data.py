"""Создаёт reference_sample.csv для drift-мониторинга из основного датасета."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate reference sample for drift detection")
    parser.add_argument(
        "--input",
        default="data/raw/geo-reviews-dataset-2023.csv",
        help="Path to full dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="data/reference/reference_sample.csv",
        help="Output reference sample path",
    )
    parser.add_argument("--nrows", type=int, default=50000, help="Rows to read from input")
    parser.add_argument("--sample-size", type=int, default=500, help="Reference sample size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}. Run `dvc pull` first.")

    df = pd.read_csv(input_path, nrows=args.nrows)
    df = df.dropna(subset=["text", "rating"])
    df = df[df["rating"] >= 1]
    df["rating"] = df["rating"].astype(int)

    sample = df.sample(n=min(args.sample_size, len(df)), random_state=42)
    sample = sample[["text", "rating"]].copy()
    sample["confidence"] = 0.75

    sample.to_csv(output_path, index=False)
    print(f"Reference sample saved: {output_path} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
