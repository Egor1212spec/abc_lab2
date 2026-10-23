import os
from pathlib import Path

import pandas as pd


def clean_frame(frame, target_column):
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()
        cleaned[column] = cleaned[column].replace("", pd.NA)
    cleaned = cleaned.dropna(how="all").dropna().reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No rows remain after removing blank or incomplete rows.")
    if target_column not in cleaned.columns:
        raise ValueError("Target column {!r} is missing. Available columns: {}".format(target_column, ", ".join(map(str, cleaned.columns))))
    if cleaned[target_column].nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two classes after cleaning.")
    return cleaned


def clean_csv(raw_path, clean_path, target_column):
    raw_path = Path(raw_path)
    clean_path = Path(clean_path)
    if not raw_path.is_file():
        raise FileNotFoundError("Raw CSV not found at {}. Run download_dataset.py or place a raw CSV there first.".format(raw_path))
    frame = pd.read_csv(raw_path, skip_blank_lines=False)
    cleaned = clean_frame(frame, target_column)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    return cleaned


def main():
    data_dir = Path(os.getenv("DATA_DIR", "/home/DATA"))
    raw_path = Path(os.getenv("RAW_DATA_PATH", str(data_dir / "raw_data.csv")))
    clean_path = Path(os.getenv("CLEAN_DATA_PATH", str(data_dir / "clean_data.csv")))
    target_column = os.getenv("TARGET_COLUMN", "species").strip()
    try:
        cleaned = clean_csv(raw_path, clean_path, target_column)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    print("Cleaned {} rows and saved {}.".format(len(cleaned), clean_path))


if __name__ == "__main__":
    main()
