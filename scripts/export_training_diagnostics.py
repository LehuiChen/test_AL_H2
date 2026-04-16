from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.io_utils import read_json, timestamp_string, write_csv_rows, write_json


MAIN_PREDICTION_FIELDS = [
    "sample_id",
    "split",
    "y_true",
    "y_pred",
    "residual",
    "abs_error",
    "reference_energy",
    "predicted_energy_main",
    "uncertainty",
    "true_gradient_norm",
    "pred_gradient_norm",
    "gradient_rmse",
    "true_force_norm",
    "pred_force_norm",
    "force_error_norm",
]

AUX_PREDICTION_FIELDS = [
    "sample_id",
    "split",
    "y_true",
    "y_pred",
    "residual",
    "abs_error",
    "reference_energy",
    "predicted_energy_main",
    "predicted_energy_aux",
    "uncertainty",
]


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def ensure_history_placeholder(path: Path, model_key: str) -> dict:
    payload = safe_read_json(path)
    if payload:
        return payload
    payload = {
        "available": False,
        "model": model_key,
        "reason": "当前运行记录中没有结构化 epoch history。",
    }
    write_json(path, payload)
    return payload


def ensure_split_placeholder(path: Path) -> dict:
    payload = safe_read_json(path)
    if payload:
        return payload
    payload = {
        "num_subtrain": 0,
        "num_validation": 0,
        "subtrain_sample_ids": [],
        "validation_sample_ids": [],
        "rows": [],
        "reason": "当前运行记录中没有训练划分信息。",
    }
    write_json(path, payload)
    return payload


def ensure_prediction_placeholder(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    write_csv_rows(path, [], fieldnames=fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总训练阶段诊断产物，方便 notebook 和结果验收。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]

    training_summary_path = Path(paths_cfg["models_dir"]) / config["training"].get("summary_filename", "training_summary.json")
    training_state_path = Path(paths_cfg["models_dir"]) / config["training"].get("state_filename", "training_state.json")
    train_main_status_path = Path(paths_cfg["models_dir"]) / "train_main_status.json"
    train_aux_status_path = Path(paths_cfg["models_dir"]) / "train_aux_status.json"

    training_split_path = Path(paths_cfg.get("training_split_file", Path(paths_cfg["models_dir"]) / "training_split.json"))
    main_predictions_path = Path(paths_cfg.get("train_main_predictions_file", Path(paths_cfg["models_dir"]) / "train_main_predictions.csv"))
    aux_predictions_path = Path(paths_cfg.get("train_aux_predictions_file", Path(paths_cfg["models_dir"]) / "train_aux_predictions.csv"))
    main_history_path = Path(paths_cfg.get("train_main_history_file", Path(paths_cfg["models_dir"]) / "train_main_history.json"))
    aux_history_path = Path(paths_cfg.get("train_aux_history_file", Path(paths_cfg["models_dir"]) / "train_aux_history.json"))
    diagnostics_path = Path(paths_cfg.get("training_diagnostics_file", Path(paths_cfg["models_dir"]) / "training_diagnostics.json"))

    training_summary = safe_read_json(training_summary_path)
    training_state = safe_read_json(training_state_path)
    train_main_status = safe_read_json(train_main_status_path)
    train_aux_status = safe_read_json(train_aux_status_path)

    training_split = ensure_split_placeholder(training_split_path)
    ensure_prediction_placeholder(main_predictions_path, MAIN_PREDICTION_FIELDS)
    ensure_prediction_placeholder(aux_predictions_path, AUX_PREDICTION_FIELDS)
    main_history = ensure_history_placeholder(main_history_path, "main_model")
    aux_history = ensure_history_placeholder(aux_history_path, "aux_model")

    diagnostics = {
        "generated_at": timestamp_string(),
        "config_file": str(Path(config["config_path"]).resolve()),
        "models_dir": str(Path(paths_cfg["models_dir"]).resolve()),
        "results_dir": str(Path(paths_cfg["results_dir"]).resolve()),
        "training_summary_file": str(training_summary_path.resolve()),
        "training_state_file": str(training_state_path.resolve()),
        "train_main_status_file": str(train_main_status_path.resolve()),
        "train_aux_status_file": str(train_aux_status_path.resolve()),
        "training_split_file": str(training_split_path.resolve()),
        "train_main_predictions_file": str(main_predictions_path.resolve()),
        "train_aux_predictions_file": str(aux_predictions_path.resolve()),
        "train_main_history_file": str(main_history_path.resolve()),
        "train_aux_history_file": str(aux_history_path.resolve()),
        "status": {
            "train_main_success": bool(train_main_status.get("success", False)),
            "train_aux_success": bool(train_aux_status.get("success", False)),
        },
        "artifacts": {
            "training_split": {
                "exists": training_split_path.exists(),
                "num_subtrain": training_split.get("num_subtrain", 0),
                "num_validation": training_split.get("num_validation", 0),
            },
            "train_main_predictions": {
                "exists": main_predictions_path.exists(),
                "num_rows": count_csv_rows(main_predictions_path),
            },
            "train_aux_predictions": {
                "exists": aux_predictions_path.exists(),
                "num_rows": count_csv_rows(aux_predictions_path),
            },
            "train_main_history": {
                "exists": main_history_path.exists(),
                "available": bool(main_history.get("available", False)),
                "reason": main_history.get("reason"),
            },
            "train_aux_history": {
                "exists": aux_history_path.exists(),
                "available": bool(aux_history.get("available", False)),
                "reason": aux_history.get("reason"),
            },
        },
        "training_summary": training_summary,
        "training_state": training_state,
        "analysis_defaults": {
            "preferred_prediction_file": str(main_predictions_path.resolve()),
            "preferred_history_file": str(main_history_path.resolve()),
            "aux_prediction_file": str(aux_predictions_path.resolve()),
            "aux_history_file": str(aux_history_path.resolve()),
            "cumulative_manifest_file": str(Path(paths_cfg.get("cumulative_labeled_manifest", "")).resolve()) if paths_cfg.get("cumulative_labeled_manifest") else None,
            "h2_seed_summary_file": str(Path(paths_cfg.get("h2_seed_summary_file", "")).resolve()) if paths_cfg.get("h2_seed_summary_file") else None,
            "active_learning_history_file": str(Path(paths_cfg.get("active_learning_round_history_file", "")).resolve()) if paths_cfg.get("active_learning_round_history_file") else None,
        },
    }

    write_json(diagnostics_path, diagnostics)
    print(f"训练诊断产物汇总完成：{diagnostics_path}")


if __name__ == "__main__":
    main()
