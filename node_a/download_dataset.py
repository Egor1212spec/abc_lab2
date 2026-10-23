import argparse
import os
from pathlib import Path

from sklearn.datasets import load_iris


def export_iris(raw_path, target_column, overwrite=False):
    raw_path = Path(raw_path)
    if not target_column.strip():
        raise ValueError("TARGET_COLUMN must not be empty.")
    iris = load_iris(as_frame=True)
    if target_column in iris.data.columns:
        raise ValueError("TARGET_COLUMN {!r} conflicts with an Iris feature column.".format(target_column))
    if raw_path.exists() and not overwrite:
        raise FileExistsError("Raw CSV already exists at {}. Pass --overwrite to replace it.".format(raw_path))
    frame = iris.data.copy()
    frame[target_column] = iris.target.map(lambda value: iris.target_names[int(value)])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)
    return frame


def main():
    data_dir = Path(os.getenv("DATA_DIR", "/home/DATA"))
    raw_path = Path(os.getenv("RAW_DATA_PATH", str(data_dir / "raw_data.csv")))
    target_column = os.getenv("TARGET_COLUMN", "species").strip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        frame = export_iris(raw_path, target_column, args.overwrite)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    print("Exported {} Iris rows to {} (target: {}).".format(len(frame), raw_path, target_column))


if __name__ == "__main__":
    main()
