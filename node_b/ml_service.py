from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify
from sklearn.metrics import accuracy_score


def _json_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def predict_saved(artifact_dir: Path) -> dict[str, Any]:
    model_path = artifact_dir / "model.joblib"
    test_path = artifact_dir / "test_data.csv"
    if not model_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Saved model or test data is missing. Run train_model.py manually first.")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Unsupported model file. Run train_model.py again.")
    test_bytes = test_path.read_bytes()
    if hashlib.sha256(test_bytes).hexdigest() != bundle["test_sha256"]:
        raise ValueError("The test CSV does not match the saved model. Run train_model.py again.")

    test_data = pd.read_csv(BytesIO(test_bytes), dtype=bundle["data_dtypes"])
    if test_data.columns.tolist() != bundle["data_columns"]:
        raise ValueError("The test CSV columns do not match the saved model.")
    if len(test_data) != bundle["test_rows"] or len(test_data) != len(bundle["test_indices"]):
        raise ValueError("The test CSV row count does not match the saved model.")
    if test_data.isna().any().any():
        raise ValueError("The test CSV contains missing values.")

    predictions = bundle["model"].predict(test_data[bundle["feature_columns"]])
    return {
        "status": "success",
        "target_column": bundle["target_column"],
        "train_rows": bundle["train_rows"],
        "test_rows": bundle["test_rows"],
        "accuracy": round(float(accuracy_score(test_data[bundle["target_column"]], predictions)), 4),
        "predictions": [
            {"row_index": int(row_index), "prediction": _json_value(prediction)}
            for row_index, prediction in zip(bundle["test_indices"], predictions)
        ],
    }


def create_app(artifact_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    saved_dir = artifact_dir if artifact_dir is not None else Path(os.getenv("ARTIFACT_DIR", "/home/MODEL"))

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "model_ready": (saved_dir / "model.joblib").is_file(),
            "test_data_ready": (saved_dir / "test_data.csv").is_file(),
        })

    @app.get("/predict")
    def predict():
        try:
            return jsonify(predict_saved(saved_dir))
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 503
        except (ValueError, KeyError, TypeError, EOFError, UnpicklingError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 422

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
