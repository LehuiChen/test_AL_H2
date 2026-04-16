from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import get_method_config
from .geometry import load_manifest, write_manifest
from .io_utils import ensure_dir, read_json, write_json
from .pbs import build_pbs_script, build_shell_command, submit_job, wait_for_status_files, write_pbs_script


def _label_file_is_success(label_file: Path) -> bool:
    if not label_file.exists():
        return False
    try:
        payload = read_json(label_file)
    except Exception:
        return False
    return bool(payload.get("success", False))


def _resolve_labels_root(config: dict[str, Any], method_key: str) -> Path:
    if method_key != "target":
        raise ValueError(f"H2 直接学习工作流只支持 target 标注，收到：{method_key}")
    return ensure_dir(Path(config["paths"]["gaussian_labels_dir"]))


def _validate_worker_status_files(status_files: list[Path], *, method_key: str) -> None:
    failures: list[str] = []
    for status_file in status_files:
        try:
            payload = read_json(status_file)
        except Exception as exc:
            failures.append(f"{status_file}: unreadable status file ({type(exc).__name__}: {exc})")
            continue
        if bool(payload.get("success", False)):
            continue
        worker_name = payload.get("worker_name", status_file.parent.name)
        num_failed = payload.get("num_failed")
        if isinstance(num_failed, int):
            failures.append(f"{worker_name}: {num_failed} sample(s) failed")
            continue
        error_type = payload.get("error_type", "UnknownError")
        error_message = payload.get("error_message", "worker batch reported failure without an error_message")
        failures.append(f"{worker_name}: {error_type}: {error_message}")

    if failures:
        raise RuntimeError(f"{method_key} label worker batches reported failures: {'; '.join(failures)}")


def _build_single_label_command(
    *,
    python_command: str,
    script_path: Path,
    config_path: Path,
    geometry_file: Path,
    method_key: str,
    output_dir: Path,
) -> list[str]:
    return [
        python_command,
        str(script_path),
        "--config",
        str(config_path),
        "--geometry",
        str(geometry_file),
        "--method-key",
        method_key,
        "--output-dir",
        str(output_dir),
    ]


def _prepare_manifest_entries(
    *,
    config: dict[str, Any],
    manifest_path: str | Path,
    method_key: str,
    force: bool,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    project_root = Path(config["project_root"]).resolve()
    labels_root = _resolve_labels_root(config, method_key)
    manifest_entries = load_manifest(manifest_path)

    pending_entries: list[dict[str, Any]] = []
    submitted_jobs: list[dict[str, Any]] = []

    for entry in manifest_entries:
        sample_id = str(entry["sample_id"])
        geometry_file = Path(entry["geometry_file"])
        if not geometry_file.is_absolute():
            geometry_file = (project_root / geometry_file).resolve()
        else:
            geometry_file = geometry_file.resolve()

        job_dir = ensure_dir(labels_root / sample_id)
        label_file = job_dir / "label.json"
        status_file = job_dir / "status.json"

        if not force and _label_file_is_success(label_file):
            submitted_jobs.append(
                {
                    "sample_id": sample_id,
                    "method_key": method_key,
                    "status": "skipped_existing_success",
                    "job_dir": str(job_dir.resolve()),
                }
            )
            continue

        write_json(job_dir / "job_meta.json", {"sample_id": sample_id, "geometry_file": str(geometry_file)})
        resolved_entry = dict(entry)
        resolved_entry["sample_id"] = sample_id
        resolved_entry["geometry_file"] = str(geometry_file)
        pending_entries.append(
            {
                "sample_id": sample_id,
                "geometry_file": geometry_file,
                "job_dir": job_dir.resolve(),
                "status_file": status_file.resolve(),
                "manifest_entry": resolved_entry,
            }
        )

    return labels_root, pending_entries, submitted_jobs


def _launch_local_jobs(
    *,
    project_root: Path,
    config_path: Path,
    python_command: str,
    script_path: Path,
    method_key: str,
    pending_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    submitted_jobs: list[dict[str, Any]] = []
    for item in pending_entries:
        local_cmd = _build_single_label_command(
            python_command=python_command,
            script_path=script_path,
            config_path=config_path,
            geometry_file=item["geometry_file"],
            method_key=method_key,
            output_dir=item["job_dir"],
        )
        subprocess.run(local_cmd, check=True, cwd=project_root)
        submitted_jobs.append(
            {
                "sample_id": item["sample_id"],
                "method_key": method_key,
                "status": "ran_locally",
                "job_dir": str(item["job_dir"]),
            }
        )
    return submitted_jobs


def _launch_per_sample_pbs_jobs(
    *,
    cluster_config: dict[str, Any],
    project_root: Path,
    config_path: Path,
    python_command: str,
    script_path: Path,
    method_key: str,
    pending_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    submitted_jobs: list[dict[str, Any]] = []
    status_files: list[Path] = []

    for item in pending_entries:
        if item["status_file"].exists():
            item["status_file"].unlink()
        local_cmd = _build_single_label_command(
            python_command=python_command,
            script_path=script_path,
            config_path=config_path,
            geometry_file=item["geometry_file"],
            method_key=method_key,
            output_dir=item["job_dir"],
        )
        pbs_text = build_pbs_script(
            job_name=f"{method_key[:3]}_{item['sample_id'][-8:]}",
            workdir=project_root,
            command=build_shell_command(local_cmd),
            stdout_path=item["job_dir"] / "stdout.log",
            stderr_path=item["job_dir"] / "stderr.log",
            cluster_config=cluster_config,
            method_key=method_key,
        )
        script_file = write_pbs_script(item["job_dir"] / "job.pbs", pbs_text)
        job_id = submit_job(script_file, submit_command=cluster_config.get("submit_command", "qsub"))
        submitted_jobs.append(
            {
                "sample_id": item["sample_id"],
                "method_key": method_key,
                "status": "submitted",
                "job_id": job_id,
                "job_dir": str(item["job_dir"]),
            }
        )
        status_files.append(item["status_file"])

    return submitted_jobs, status_files


def _distribute_entries_round_robin(pending_entries: list[dict[str, Any]], worker_count: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    for index, entry in enumerate(pending_entries):
        chunks[index % worker_count].append(entry)
    return [chunk for chunk in chunks if chunk]


def _launch_worker_pbs_jobs(
    *,
    cluster_config: dict[str, Any],
    project_root: Path,
    config_path: Path,
    python_command: str,
    method_key: str,
    labels_root: Path,
    pending_entries: list[dict[str, Any]],
    force: bool,
) -> tuple[list[dict[str, Any]], list[Path]]:
    resources = cluster_config.get("resources_by_method", {}).get(method_key, {})
    requested_worker_count = max(1, int(resources.get("worker_count", 1)))
    local_parallelism = max(1, int(resources.get("local_parallelism", 1)))
    worker_count = min(requested_worker_count, len(pending_entries))
    if worker_count == 0:
        return [], []

    jobs_root = ensure_dir(labels_root / "jobs")
    batch_script_path = (project_root / "scripts" / "execute_label_batch.py").resolve()
    worker_status_files: list[Path] = []
    submitted_jobs: list[dict[str, Any]] = []

    for worker_index, chunk in enumerate(_distribute_entries_round_robin(pending_entries, worker_count)):
        worker_dir = ensure_dir(jobs_root / f"worker_{worker_index:03d}")
        worker_manifest_path = worker_dir / "worker_manifest.json"
        worker_status_path = worker_dir / "batch_status.json"

        if worker_status_path.exists():
            worker_status_path.unlink()

        worker_manifest_entries = [dict(item["manifest_entry"]) for item in chunk]
        write_manifest(worker_manifest_entries, worker_manifest_path)

        local_cmd = [
            python_command,
            str(batch_script_path),
            "--config",
            str(config_path),
            "--manifest",
            str(worker_manifest_path.resolve()),
            "--method-key",
            method_key,
            "--labels-root",
            str(labels_root.resolve()),
            "--status-file",
            str(worker_status_path.resolve()),
            "--local-parallelism",
            str(local_parallelism),
            "--worker-name",
            worker_dir.name,
        ]
        if force:
            local_cmd.append("--force")

        pbs_text = build_pbs_script(
            job_name=f"{method_key[:3]}w_{worker_index:03d}",
            workdir=project_root,
            command=build_shell_command(local_cmd),
            stdout_path=worker_dir / "stdout.log",
            stderr_path=worker_dir / "stderr.log",
            cluster_config=cluster_config,
            method_key=method_key,
        )
        script_file = write_pbs_script(worker_dir / "job.pbs", pbs_text)
        job_id = submit_job(script_file, submit_command=cluster_config.get("submit_command", "qsub"))

        write_json(
            worker_dir / "job_meta.json",
            {
                "worker_name": worker_dir.name,
                "method_key": method_key,
                "job_id": job_id,
                "local_parallelism": local_parallelism,
                "num_samples": len(chunk),
                "sample_ids": [item["sample_id"] for item in chunk],
                "worker_manifest": str(worker_manifest_path.resolve()),
            },
        )

        for item in chunk:
            submitted_jobs.append(
                {
                    "sample_id": item["sample_id"],
                    "method_key": method_key,
                    "status": "queued_in_worker",
                    "job_id": job_id,
                    "worker_name": worker_dir.name,
                    "job_dir": str(item["job_dir"]),
                }
            )

        worker_status_files.append(worker_status_path)

    return submitted_jobs, worker_status_files


def launch_label_jobs(
    *,
    config: dict,
    manifest_path: str | Path,
    method_key: str,
    submit_mode: str,
    wait: bool,
    force: bool,
) -> list[dict]:
    get_method_config(config, method_key)
    cluster_config = config["cluster"]
    project_root = Path(config["project_root"]).resolve()
    config_path = Path(config["config_path"]).resolve()
    python_command = cluster_config.get("python_command", "python")
    script_path = (project_root / "scripts" / "execute_label_job.py").resolve()

    labels_root, pending_entries, submitted_jobs = _prepare_manifest_entries(
        config=config,
        manifest_path=manifest_path,
        method_key=method_key,
        force=force,
    )

    if submit_mode == "local":
        submitted_jobs.extend(
            _launch_local_jobs(
                project_root=project_root,
                config_path=config_path,
                python_command=python_command,
                script_path=script_path,
                method_key=method_key,
                pending_entries=pending_entries,
            )
        )
        return submitted_jobs

    if submit_mode != "pbs":
        raise ValueError("submit_mode 只能是 `local` 或 `pbs`。")

    resources = cluster_config.get("resources_by_method", {}).get(method_key, {})
    submission_strategy = str(resources.get("submission_strategy", "worker")).strip().lower()

    if submission_strategy == "worker":
        new_jobs, status_files = _launch_worker_pbs_jobs(
            cluster_config=cluster_config,
            project_root=project_root,
            config_path=config_path,
            python_command=python_command,
            method_key=method_key,
            labels_root=labels_root,
            pending_entries=pending_entries,
            force=force,
        )
    elif submission_strategy == "per-sample":
        new_jobs, status_files = _launch_per_sample_pbs_jobs(
            cluster_config=cluster_config,
            project_root=project_root,
            config_path=config_path,
            python_command=python_command,
            script_path=script_path,
            method_key=method_key,
            pending_entries=pending_entries,
        )
    else:
        raise ValueError(f"不支持的 label 提交策略：{submission_strategy}")

    submitted_jobs.extend(new_jobs)

    if wait and status_files:
        wait_for_status_files(
            status_files,
            timeout_seconds=int(cluster_config.get("poll_timeout_seconds", 86400)),
            poll_interval_seconds=int(cluster_config.get("poll_interval_seconds", 30)),
        )
        _validate_worker_status_files(status_files, method_key=method_key)

    return submitted_jobs
