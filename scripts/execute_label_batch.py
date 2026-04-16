from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.geometry import load_manifest
from minimal_adl.io_utils import ensure_dir, read_json, timestamp_string, write_json


def _label_file_is_success(label_file: Path) -> bool:
    if not label_file.exists():
        return False
    try:
        payload = read_json(label_file)
    except Exception:
        return False
    return bool(payload.get("success", False))


def _run_single_sample(
    *,
    entry: dict[str, Any],
    project_root: Path,
    config_path: Path,
    method_key: str,
    labels_root: Path,
    python_command: str,
    execute_script_path: Path,
    force: bool,
) -> dict[str, Any]:
    sample_id = str(entry["sample_id"])
    geometry_file = Path(entry["geometry_file"])
    if not geometry_file.is_absolute():
        geometry_file = (project_root / geometry_file).resolve()
    else:
        geometry_file = geometry_file.resolve()

    output_dir = ensure_dir(labels_root / sample_id)
    label_file = output_dir / "label.json"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    if not force and _label_file_is_success(label_file):
        return {
            "sample_id": sample_id,
            "status": "skipped_existing_success",
            "success": True,
            "job_dir": str(output_dir.resolve()),
        }

    command = [
        python_command,
        str(execute_script_path),
        "--config",
        str(config_path),
        "--geometry",
        str(geometry_file),
        "--method-key",
        method_key,
        "--output-dir",
        str(output_dir.resolve()),
    ]

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=project_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )

    payload = None
    if label_file.exists():
        try:
            payload = read_json(label_file)
        except Exception:
            payload = None

    return {
        "sample_id": sample_id,
        "status": "completed" if result.returncode == 0 else "failed",
        "success": bool(payload and payload.get("success", False)),
        "returncode": result.returncode,
        "job_dir": str(output_dir.resolve()),
        "label_file": str(label_file.resolve()) if label_file.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在单个 PBS worker 内批量执行 target label 任务。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--manifest", required=True, help="当前 worker 负责的子 manifest。")
    parser.add_argument("--method-key", required=True, choices=["target"], help="方法键名。")
    parser.add_argument("--labels-root", required=True, help="统一 label 根目录。")
    parser.add_argument("--status-file", required=True, help="当前 worker 的汇总状态文件路径。")
    parser.add_argument("--local-parallelism", type=int, default=1, help="单个 worker 内并发样本数。")
    parser.add_argument("--worker-name", default="", help="可选 worker 名称。")
    parser.add_argument("--force", action="store_true", help="即使结果已存在也重新执行。")
    args = parser.parse_args()

    status_file = Path(args.status_file).resolve()

    try:
        config = load_config(args.config)
        project_root = Path(config["project_root"]).resolve()
        config_path = Path(config["config_path"]).resolve()
        labels_root = ensure_dir(Path(args.labels_root).resolve())
        execute_script_path = (project_root / "scripts" / "execute_label_job.py").resolve()
        python_command = str(config["cluster"].get("python_command", "python"))
        manifest_entries = load_manifest(args.manifest)
        local_parallelism = max(1, int(args.local_parallelism))

        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=local_parallelism) as executor:
            futures = [
                executor.submit(
                    _run_single_sample,
                    entry=entry,
                    project_root=project_root,
                    config_path=config_path,
                    method_key=args.method_key,
                    labels_root=labels_root,
                    python_command=python_command,
                    execute_script_path=execute_script_path,
                    force=args.force,
                )
                for entry in manifest_entries
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda item: item["sample_id"])
        num_success = sum(1 for item in results if item.get("success"))
        num_failed = sum(1 for item in results if not item.get("success"))
        summary = {
            "success": num_failed == 0,
            "worker_name": args.worker_name or status_file.parent.name,
            "method_key": args.method_key,
            "manifest": str(Path(args.manifest).resolve()),
            "labels_root": str(labels_root.resolve()),
            "local_parallelism": local_parallelism,
            "num_samples": len(results),
            "num_success": num_success,
            "num_failed": num_failed,
            "finished_at": timestamp_string(),
            "results": results,
        }
        write_json(status_file, summary)

        if num_failed:
            raise SystemExit(1)
    except Exception as exc:
        write_json(
            status_file,
            {
                "success": False,
                "worker_name": args.worker_name or status_file.parent.name,
                "method_key": args.method_key,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "finished_at": timestamp_string(),
            },
        )
        raise


if __name__ == "__main__":
    main()
