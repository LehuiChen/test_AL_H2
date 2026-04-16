from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.geometry import load_manifest
from minimal_adl.io_utils import read_json, timestamp_string, write_json


StageCheck = Callable[[], bool]
StageRun = Callable[[], dict[str, Any]]


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def label_success(label_file: Path) -> bool:
    payload = safe_read_json(label_file)
    return bool(payload.get("success", False))


def manifest_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_manifest(path)


def manifest_sample_ids(path: Path) -> list[str]:
    return [str(item["sample_id"]) for item in manifest_entries(path)]


def count_successful_labels(labels_root: Path, sample_ids: list[str]) -> int:
    return sum(1 for sample_id in sample_ids if label_success(labels_root / sample_id / "label.json"))


def cumulative_contains_all(cumulative_manifest_path: Path, sample_ids: list[str]) -> bool:
    if not cumulative_manifest_path.exists():
        return False
    cumulative_ids = {str(item["sample_id"]) for item in manifest_entries(cumulative_manifest_path)}
    return set(sample_ids).issubset(cumulative_ids)


def run_python_script(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def resolve_required_environment_checks(report: dict[str, Any]) -> list[str]:
    configured = report.get("required_checks")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    return ["yaml", "mlatom", "pyh5md", "joblib", "sklearn", "torch", "torchani", "g16"]


def main() -> None:
    stage_names = [
        "check_environment",
        "prepare_h2_seed",
        "sample_round_000_initial_conditions",
        "labels_round_000",
        "build_training_dataset",
        "train_main_model",
        "train_aux_model",
        "export_training_diagnostics",
        "run_md_sampling_round_001",
        "select_round_001_frames",
    ]

    parser = argparse.ArgumentParser(description="运行 H2 直接学习主动学习第一轮流水线。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--from-stage", choices=stage_names, default=None)
    parser.add_argument("--to-stage", choices=stage_names, default=None)
    parser.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="即使输出存在也重跑。")
    parser.add_argument("--submit-mode-labels", choices=["local", "pbs"], default="pbs")
    parser.add_argument("--submit-mode-train", choices=["local", "pbs"], default="pbs")
    parser.add_argument("--submit-mode-md", choices=["local", "pbs"], default="pbs")
    parser.add_argument("--md-num-initial-conditions", type=int, default=None)
    parser.add_argument("--md-maximum-propagation-time", type=float, default=None)
    parser.add_argument("--md-time-step", type=float, default=None)
    parser.add_argument("--md-save-interval-steps", type=int, default=None)
    parser.add_argument("--device", default=None, help="训练或 MD 设备覆盖，例如 cpu 或 cuda。")
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(config["project_root"]).resolve()
    config_path = Path(config["config_path"]).resolve()
    paths_cfg = config["paths"]
    results_dir = Path(paths_cfg["results_dir"]).resolve()
    models_dir = Path(paths_cfg["models_dir"]).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    h2_seed_json_path = Path(paths_cfg["h2_seed_json"]).resolve()
    h2_seed_xyz_path = Path(paths_cfg["h2_seed_xyz"]).resolve()
    h2_seed_summary_path = Path(paths_cfg["h2_seed_summary_file"]).resolve()
    round_000_manifest_path = results_dir / "round_000_initial_conditions_manifest.json"
    round_000_labels_status_path = results_dir / "round_000_labels_status.json"
    cumulative_manifest_path = Path(paths_cfg["cumulative_labeled_manifest"]).resolve()
    direct_npz_path = Path(paths_cfg["direct_dataset_npz"]).resolve()
    direct_metadata_path = Path(paths_cfg["direct_dataset_metadata"]).resolve()
    train_main_status_path = models_dir / "train_main_status.json"
    train_aux_status_path = models_dir / "train_aux_status.json"
    training_diagnostics_path = Path(paths_cfg.get("training_diagnostics_file", models_dir / "training_diagnostics.json")).resolve()
    round_001_md_status_path = results_dir / "round_001_md_sampling_status.json"
    round_001_md_frame_manifest_path = results_dir / "round_001_md_frame_manifest.json"
    round_001_selection_summary_path = results_dir / "round_001_selection_summary.json"
    round_001_selection_manifest_path = results_dir / "round_001_selected_manifest.json"
    active_learning_history_path = Path(paths_cfg.get("active_learning_round_history_file", results_dir / "active_learning_round_history.json")).resolve()
    check_environment_report_path = Path(paths_cfg.get("check_environment_report", results_dir / "check_environment_latest.json")).resolve()
    pipeline_run_summary_path = Path(paths_cfg.get("pipeline_run_summary", results_dir / "pipeline_run_summary.json")).resolve()

    start_index = 0 if args.from_stage is None else stage_names.index(args.from_stage)
    end_index = len(stage_names) - 1 if args.to_stage is None else stage_names.index(args.to_stage)
    if start_index > end_index:
        raise ValueError("from-stage 不能位于 to-stage 之后。")
    selected_stages = stage_names[start_index : end_index + 1]

    pipeline_summary: dict[str, Any] = {
        "generated_at": timestamp_string(),
        "config_file": str(config_path),
        "selected_stages": selected_stages,
        "resume": args.resume,
        "force": args.force,
        "submit_mode_labels": args.submit_mode_labels,
        "submit_mode_train": args.submit_mode_train,
        "submit_mode_md": args.submit_mode_md,
        "device_override": args.device,
        "stages": [],
        "success": False,
    }

    def persist_summary() -> None:
        write_json(pipeline_run_summary_path, pipeline_summary)

    def round_000_sample_ids() -> list[str]:
        return manifest_sample_ids(round_000_manifest_path)

    def check_environment_complete() -> bool:
        report = safe_read_json(check_environment_report_path)
        checks = report.get("checks")
        if not isinstance(checks, dict) or not checks:
            return False
        required_names = resolve_required_environment_checks(report)
        return all(bool(checks.get(name, {}).get("ok", False)) for name in required_names)

    def prepare_h2_seed_complete() -> bool:
        summary = safe_read_json(h2_seed_summary_path)
        return h2_seed_json_path.exists() and h2_seed_xyz_path.exists() and summary.get("num_atoms") is not None

    def round_000_sampling_complete() -> bool:
        expected_count = int(config.get("sampling", {}).get("initial_conditions_initial", 0))
        sample_ids = round_000_sample_ids()
        return bool(sample_ids) and len(sample_ids) == expected_count

    def labels_round_000_complete() -> bool:
        sample_ids = round_000_sample_ids()
        if not sample_ids:
            return False
        gaussian_root = Path(paths_cfg["gaussian_labels_dir"]).resolve()
        gaussian_ok = count_successful_labels(gaussian_root, sample_ids) == len(sample_ids)
        return gaussian_ok and cumulative_contains_all(cumulative_manifest_path, sample_ids)

    def direct_dataset_complete() -> bool:
        metadata = safe_read_json(direct_metadata_path)
        cumulative_ids = manifest_sample_ids(cumulative_manifest_path)
        return direct_npz_path.exists() and metadata.get("num_samples", 0) == len(cumulative_ids) > 0

    def train_main_complete() -> bool:
        status = safe_read_json(train_main_status_path)
        model_file = status.get("main_model_file")
        return bool(status.get("success", False)) and bool(model_file) and Path(model_file).exists()

    def train_aux_complete() -> bool:
        status = safe_read_json(train_aux_status_path)
        model_file = status.get("aux_model_file")
        return bool(status.get("success", False)) and bool(model_file) and Path(model_file).exists()

    def diagnostics_complete() -> bool:
        diagnostics = safe_read_json(training_diagnostics_path)
        return bool(diagnostics.get("artifacts"))

    def md_sampling_complete() -> bool:
        status = safe_read_json(round_001_md_status_path)
        if not bool(status.get("success", False)) or not round_001_md_frame_manifest_path.exists():
            return False
        requested_initial_conditions = int(
            args.md_num_initial_conditions
            or config.get("sampling", {}).get("initial_conditions_per_round", 100)
        )
        trajectory_mode = str(config.get("sampling", {}).get("md", {}).get("trajectory_mode", "bidirectional")).strip().lower()
        trajectory_multiplier = 2 if trajectory_mode == "bidirectional" else 1
        expected_trajectories = requested_initial_conditions * trajectory_multiplier
        return int(status.get("num_trajectories", -1)) == expected_trajectories

    def selection_complete() -> bool:
        if not (
            round_001_selection_summary_path.exists()
            and round_001_selection_manifest_path.exists()
            and active_learning_history_path.exists()
        ):
            return False
        summary = safe_read_json(round_001_selection_summary_path)
        if int(summary.get("round_index", -1)) != 1:
            return False
        selected_count = int(summary.get("selected_count", -1))
        manifest_count = len(load_manifest(round_001_selection_manifest_path))
        return selected_count == manifest_count

    def run_labels_round_000() -> dict[str, Any]:
        try:
            run_python_script(
                [
                    sys.executable,
                    str((project_root / "scripts" / "run_target_labels.py").resolve()),
                    "--config",
                    str(config_path),
                    "--manifest",
                    str(round_000_manifest_path),
                    "--submit-mode",
                    args.submit_mode_labels,
                    *([] if not args.force else ["--force"]),
                ],
                cwd=project_root,
            )
            run_python_script(
                [
                    sys.executable,
                    str((project_root / "scripts" / "update_cumulative_manifest.py").resolve()),
                    "--config",
                    str(config_path),
                    "--manifest",
                    str(round_000_manifest_path),
                ],
                cwd=project_root,
            )
        except Exception as exc:
            write_json(
                round_000_labels_status_path,
                {
                    "success": False,
                    "manifest_file": str(round_000_manifest_path.resolve()),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

        payload = {
            "success": True,
            "manifest_file": str(round_000_manifest_path.resolve()),
            "sample_ids": round_000_sample_ids(),
            "gaussian_labels_dir": str(Path(paths_cfg["gaussian_labels_dir"]).resolve()),
            "cumulative_manifest_file": str(cumulative_manifest_path.resolve()),
            "updated_at": timestamp_string(),
        }
        write_json(round_000_labels_status_path, payload)
        return payload

    stages: dict[str, dict[str, Any]] = {
        "check_environment": {
            "is_complete": check_environment_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "check_environment.py").resolve()),
                        "--config",
                        str(config_path),
                        "--json-output",
                        str(check_environment_report_path),
                        "--strict",
                    ],
                    cwd=project_root,
                ),
                {"report_file": str(check_environment_report_path)},
            )[1],
        },
        "prepare_h2_seed": {
            "is_complete": prepare_h2_seed_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "prepare_h2_seed.py").resolve()),
                        "--config",
                        str(config_path),
                    ],
                    cwd=project_root,
                ),
                {
                    "h2_seed_json": str(h2_seed_json_path),
                    "h2_seed_xyz": str(h2_seed_xyz_path),
                    "h2_seed_summary_file": str(h2_seed_summary_path),
                },
            )[1],
        },
        "sample_round_000_initial_conditions": {
            "is_complete": round_000_sampling_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "sample_h2_initial_conditions.py").resolve()),
                        "--config",
                        str(config_path),
                        "--round-index",
                        "0",
                    ],
                    cwd=project_root,
                ),
                {"round_000_initial_conditions_manifest": str(round_000_manifest_path)},
            )[1],
        },
        "labels_round_000": {
            "is_complete": labels_round_000_complete,
            "run": run_labels_round_000,
        },
        "build_training_dataset": {
            "is_complete": direct_dataset_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "build_training_dataset.py").resolve()),
                        "--config",
                        str(config_path),
                        "--manifest",
                        str(cumulative_manifest_path),
                    ],
                    cwd=project_root,
                ),
                {
                    "direct_dataset_npz": str(direct_npz_path),
                    "direct_dataset_metadata": str(direct_metadata_path),
                },
            )[1],
        },
        "train_main_model": {
            "is_complete": train_main_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "train_main_model.py").resolve()),
                        "--config",
                        str(config_path),
                        "--submit-mode",
                        args.submit_mode_train,
                        *([] if args.device is None else ["--device", args.device]),
                    ],
                    cwd=project_root,
                ),
                {"status_file": str(train_main_status_path)},
            )[1],
        },
        "train_aux_model": {
            "is_complete": train_aux_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "train_aux_model.py").resolve()),
                        "--config",
                        str(config_path),
                        "--submit-mode",
                        args.submit_mode_train,
                        *([] if args.device is None else ["--device", args.device]),
                    ],
                    cwd=project_root,
                ),
                {"status_file": str(train_aux_status_path)},
            )[1],
        },
        "export_training_diagnostics": {
            "is_complete": diagnostics_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "export_training_diagnostics.py").resolve()),
                        "--config",
                        str(config_path),
                    ],
                    cwd=project_root,
                ),
                {"training_diagnostics_file": str(training_diagnostics_path)},
            )[1],
        },
        "run_md_sampling_round_001": {
            "is_complete": md_sampling_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "run_md_sampling.py").resolve()),
                        "--config",
                        str(config_path),
                        "--round-index",
                        "1",
                        "--submit-mode",
                        args.submit_mode_md,
                        *([] if args.md_num_initial_conditions is None else ["--num-initial-conditions", str(args.md_num_initial_conditions)]),
                        *([] if args.md_maximum_propagation_time is None else ["--maximum-propagation-time", str(args.md_maximum_propagation_time)]),
                        *([] if args.md_time_step is None else ["--time-step", str(args.md_time_step)]),
                        *([] if args.md_save_interval_steps is None else ["--save-interval-steps", str(args.md_save_interval_steps)]),
                        *([] if args.device is None else ["--device", args.device]),
                    ],
                    cwd=project_root,
                ),
                {
                    "md_sampling_status_file": str(round_001_md_status_path),
                    "frame_manifest_file": str(round_001_md_frame_manifest_path),
                },
            )[1],
        },
        "select_round_001_frames": {
            "is_complete": selection_complete,
            "run": lambda: (
                run_python_script(
                    [
                        sys.executable,
                        str((project_root / "scripts" / "select_md_frames.py").resolve()),
                        "--config",
                        str(config_path),
                        "--round-index",
                        "1",
                    ],
                    cwd=project_root,
                ),
                {
                    "selection_summary_file": str(round_001_selection_summary_path),
                    "selection_manifest_file": str(round_001_selection_manifest_path),
                    "active_learning_round_history_file": str(active_learning_history_path),
                },
            )[1],
        },
    }

    persist_summary()

    must_rerun_remaining = False
    for stage_name in selected_stages:
        stage_def = stages[stage_name]
        stage_record: dict[str, Any] = {"stage": stage_name, "started_at": timestamp_string()}
        try:
            if args.resume and not args.force and not must_rerun_remaining and stage_def["is_complete"]():
                stage_record["status"] = "skipped"
                stage_record["reason"] = "已有输出满足当前阶段。"
            else:
                outputs = stage_def["run"]()
                stage_record["status"] = "completed"
                stage_record["outputs"] = outputs
                must_rerun_remaining = True
            stage_record["finished_at"] = timestamp_string()
            pipeline_summary["stages"].append(stage_record)
            persist_summary()
        except Exception as exc:
            stage_record["status"] = "failed"
            stage_record["finished_at"] = timestamp_string()
            stage_record["error_type"] = type(exc).__name__
            stage_record["error_message"] = str(exc)
            stage_record["traceback"] = traceback.format_exc()
            pipeline_summary["stages"].append(stage_record)
            pipeline_summary["success"] = False
            persist_summary()
            raise

    pipeline_summary["success"] = True
    pipeline_summary["finished_at"] = timestamp_string()
    persist_summary()
    print(f"First-round H2 pipeline completed: {pipeline_run_summary_path}")


if __name__ == "__main__":
    main()
