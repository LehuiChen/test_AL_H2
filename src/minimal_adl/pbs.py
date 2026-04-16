from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .io_utils import ensure_dir, read_json


def _normalize_shell_lines(block: str | Iterable[str] | None) -> list[str]:
    """把环境块统一转换成命令列表。"""

    if block is None:
        return []
    if isinstance(block, str):
        return [line.strip() for line in block.splitlines() if line.strip()]

    normalized: list[str] = []
    for line in block:
        if isinstance(line, dict):
            raise TypeError(
                "PBS 环境块中的命令必须是字符串。如果命令里包含冒号，请把整条命令放进引号。"
            )
        text = str(line).strip()
        if text:
            normalized.append(text)
    return normalized


def build_shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_pbs_script(
    *,
    job_name: str,
    workdir: str | Path,
    command: str,
    stdout_path: str | Path,
    stderr_path: str | Path,
    cluster_config: dict,
    method_key: str,
) -> str:
    workdir = Path(workdir).resolve()
    stdout_path = Path(stdout_path).resolve()
    stderr_path = Path(stderr_path).resolve()

    resources = cluster_config.get("resources_by_method", {}).get(method_key, {})
    queue = resources.get("queue", cluster_config.get("queue", "default"))
    nodes = int(resources.get("nodes", cluster_config.get("nodes", 1)))
    ppn = int(resources.get("ppn", cluster_config.get("ppn", 1)))
    walltime = resources.get("walltime", cluster_config.get("walltime", "01:00:00"))
    extra_pbs_lines = _normalize_shell_lines(resources.get("extra_pbs_lines"))
    env_blocks = cluster_config.get("env_blocks", {})
    cleanup_blocks = cluster_config.get("cleanup_blocks", {})
    setup_lines = _normalize_shell_lines(env_blocks.get(method_key))
    cleanup_lines = _normalize_shell_lines(cleanup_blocks.get(method_key))

    if not setup_lines:
        conda_init = cluster_config.get("conda_init", "source ~/.bashrc")
        conda_env = cluster_config.get("conda_env", "ADL_env")
        setup_lines = [conda_init, f"conda activate {conda_env}"]

    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -q {queue}",
        f"#PBS -l nodes={nodes}:ppn={ppn}",
        f"#PBS -l walltime={walltime}",
        f"#PBS -o {stdout_path}",
        f"#PBS -e {stderr_path}",
        *extra_pbs_lines,
        "",
        "set -euo pipefail",
        f"cd {workdir}",
        "",
    ]

    if cleanup_lines:
        lines.extend([
            "cleanup_job() {",
            *[f"  {line}" for line in cleanup_lines],
            "}",
            "trap cleanup_job EXIT",
            "",
        ])

    lines.extend(setup_lines)
    lines.extend(["", command, ""])
    return "\n".join(lines)


def submit_job(script_path: str | Path, submit_command: str = "qsub") -> str:
    script_path = Path(script_path).resolve()
    result = subprocess.run(
        [submit_command, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def wait_for_status_files(
    status_files: list[Path],
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    start_time = time.time()
    pending = set(Path(item).resolve() for item in status_files)

    while pending:
        finished = {item for item in pending if item.exists()}
        pending -= finished
        if not pending:
            return
        if time.time() - start_time > timeout_seconds:
            pending_text = "\n".join(str(item) for item in sorted(pending))
            raise TimeoutError(f"等待作业结果超时，以下状态文件仍未出现：\n{pending_text}")
        time.sleep(poll_interval_seconds)


def write_pbs_script(script_path: str | Path, content: str) -> Path:
    script_path = Path(script_path)
    ensure_dir(script_path.parent)
    script_path.write_text(content, encoding="utf-8")
    return script_path


def launch_python_job(
    *,
    config: dict,
    job_key: str,
    submit_mode: str,
    wait: bool,
    script_path: str | Path,
    script_args: list[str],
    output_dir: str | Path,
    status_file: str | Path,
    job_name: str,
) -> dict[str, str]:
    project_root = Path(config["project_root"]).resolve()
    cluster_config = config["cluster"]
    python_command = cluster_config.get("python_command", "python")
    output_dir = ensure_dir(output_dir)
    status_file = Path(status_file).resolve()

    if status_file.exists():
        status_file.unlink()

    command_parts = [python_command, str(Path(script_path).resolve()), *script_args]

    if submit_mode == "local":
        subprocess.run(command_parts, check=True, cwd=project_root)
        return {
            "status": "ran_locally",
            "job_name": job_name,
            "status_file": str(status_file),
        }

    if submit_mode != "pbs":
        raise ValueError("submit_mode 只能是 `local` 或 `pbs`。")

    pbs_text = build_pbs_script(
        job_name=job_name,
        workdir=project_root,
        command=build_shell_command(command_parts),
        stdout_path=Path(output_dir) / "stdout.log",
        stderr_path=Path(output_dir) / "stderr.log",
        cluster_config=cluster_config,
        method_key=job_key,
    )
    script_file = write_pbs_script(Path(output_dir) / "job.pbs", pbs_text)
    job_id = submit_job(script_file, submit_command=cluster_config.get("submit_command", "qsub"))

    if wait:
        wait_for_status_files(
            [status_file],
            timeout_seconds=int(cluster_config.get("poll_timeout_seconds", 86400)),
            poll_interval_seconds=int(cluster_config.get("poll_interval_seconds", 30)),
        )
        try:
            status_payload = read_json(status_file)
        except Exception as exc:
            raise RuntimeError(f"PBS job `{job_name}` 已结束，但无法读取状态文件：{status_file}") from exc
        if not bool(status_payload.get("success", False)):
            error_type = status_payload.get("error_type", "UnknownError")
            error_message = status_payload.get("error_message", "PBS job reported failure without an error_message.")
            raise RuntimeError(f"PBS job `{job_name}` failed: {error_type}: {error_message}")

    return {
        "status": "completed" if wait else "submitted",
        "job_name": job_name,
        "job_id": job_id,
        "status_file": str(status_file),
        "job_script": str(script_file),
    }
