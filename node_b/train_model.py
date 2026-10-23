from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def train_model(
    data_path: Path,
    artifact_dir: Path,
    target_column: str = "species",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Transfer clean_data.csv to {data_path} before training.")
    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    data = pd.read_csv(data_path, skip_blank_lines=False)
    if data.empty:
        raise ValueError("The input CSV contains no rows.")
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} is missing. Set TARGET_COLUMN.")
    if data.isna().any().any():
        raise ValueError("The CSV contains missing values. Run clean_data.py on node A first.")
    if any(
        data[column].astype("string").str.strip().eq("").any()
        for column in data.select_dtypes(include=["object", "string"]).columns
    ):
        raise ValueError("The CSV contains blank values. Run clean_data.py on node A first.")
    if len(data.columns) < 2:
        raise ValueError("At least one feature column is required.")
    if data[target_column].nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")
    if data[target_column].value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for a stratified split.")

    try:
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data[target_column],
        )
    except ValueError as exc:
        raise ValueError(f"Cannot split the dataset into train and test parts: {exc}") from exc

    features = train_data.drop(columns=[target_column])
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", "passthrough", numeric_columns))
    if categorical_columns:
        transformers.append((
            "categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns,
        ))
    model = Pipeline([
        ("features", ColumnTransformer(transformers=transformers)),
        ("classifier", DecisionTreeClassifier(random_state=random_state)),
    ])
    model.fit(features, train_data[target_column])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    test_path = artifact_dir / "test_data.csv"
    with TemporaryDirectory(dir=artifact_dir) as temporary_dir:
        temporary_test = Path(temporary_dir) / "test_data.csv"
        temporary_model = Path(temporary_dir) / "model.joblib"
        test_data.to_csv(temporary_test, index=False)
        bundle = {
            "format_version": 1,
            "model": model,
            "target_column": target_column,
            "feature_columns": features.columns.tolist(),
            "data_columns": data.columns.tolist(),
            "data_dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "test_indices": test_data.index.tolist(),
            "test_sha256": hashlib.sha256(temporary_test.read_bytes()).hexdigest(),
            "test_size": test_size,
            "random_state": random_state,
        }
        joblib.dump(bundle, temporary_model, compress=3)
        temporary_test.replace(test_path)
        temporary_model.replace(model_path)

    return {
        "status": "trained",
        "target_column": target_column,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "model_path": str(model_path),
        "test_data_path": str(test_path),
    }


def main() -> None:
    data_path = Path(os.getenv("DATA_PATH", "/home/DATA/clean_data.csv"))
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", "/home/MODEL"))
    result = train_model(
        data_path,
        artifact_dir,
        target_column=os.getenv("TARGET_COLUMN", "species").strip(),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
