#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Config must be a YAML mapping: %s" % path)
    return payload


def _resolve_config(config_arg: str) -> Path:
    path = Path(config_arg)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _resolve_run_dir(config: Dict[str, Any], mode: str, override: str) -> Path:
    if override:
        return Path(override).resolve()
    run_cfg = config.get("run", {})
    runs_dir = Path(run_cfg.get("runs_dir", "runs"))
    if not runs_dir.is_absolute():
        runs_dir = (PROJECT_ROOT / runs_dir).resolve()
    if mode == "smoke":
        name = str(run_cfg.get("smoke_name") or "h2_ani_smoke")
    else:
        name = str(run_cfg.get("full_name") or "h2_ani_full")
    return (runs_dir / name).resolve()


def _effective_submit_mode(labels: str, train: str, md: str) -> str:
    unique = {labels, train, md}
    if len(unique) == 1:
        return unique.pop()
    if "pbs" in unique:
        return "pbs"
    return "local"


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADL 风格一键入口：ANI-H2 直接主动学习流程。"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--resume", dest="resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--submit-mode-labels", choices=("local", "pbs"), default="pbs")
    parser.add_argument("--submit-mode-train", choices=("local", "pbs"), default="pbs")
    parser.add_argument("--submit-mode-md", choices=("local", "pbs"), default="pbs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument("--wait", dest="wait", action="store_true")
    wait_group.add_argument("--no-wait", dest="wait", action="store_false")
    parser.set_defaults(wait=False)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(list(argv or []))
    config_path = _resolve_config(args.config)
    config = _load_yaml(config_path)
    run_dir = _resolve_run_dir(config, args.mode, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    submit_mode = _effective_submit_mode(
        labels=args.submit_mode_labels,
        train=args.submit_mode_train,
        md=args.submit_mode_md,
    )

    command = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "run_h2_al.py").resolve()),
        "--config",
        str(config_path),
        "--mode",
        args.mode,
        "--submit-mode",
        submit_mode,
        "--run-dir",
        str(run_dir),
    ]
    if args.force:
        command.append("--force-recompute-freq")
    if args.debug:
        command.append("--debug")
    if args.dry_run:
        command.append("--dry-run")
    command.append("--wait" if args.wait else "--no-wait")

    started_at = _timestamp()
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)

    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

    submission_path = run_dir / "submission.json"
    status_path = run_dir / "status.json"
    al_info_path = run_dir / "al_info.json"
    submission = _safe_load_json(submission_path)
    status = _safe_load_json(status_path)

    summary = {
        "generated_at": _timestamp(),
        "started_at": started_at,
        "config_file": str(config_path),
        "mode": args.mode,
        "resume": bool(args.resume),
        "force": bool(args.force),
        "submit_modes": {
            "labels": args.submit_mode_labels,
            "train": args.submit_mode_train,
            "md": args.submit_mode_md,
            "effective": submit_mode,
        },
        "wait": bool(args.wait),
        "dry_run": bool(args.dry_run),
        "run_dir": str(run_dir),
        "command": command,
        "returncode": proc.returncode,
        "submission_file": str(submission_path),
        "submission": submission if submission else None,
        "status_file": str(status_path),
        "status": status if status else None,
        "al_info_file": str(al_info_path),
        "al_info_exists": al_info_path.exists(),
        "success": proc.returncode == 0,
    }
    if submission.get("job_id"):
        summary["job_id"] = submission.get("job_id")
    if status.get("success") is not None:
        summary["status_success"] = bool(status.get("success"))

    summary_path = run_dir / "active_learning_experiment_summary.json"
    _write_json(summary_path, summary)
    print("实验汇总文件:", summary_path)

    if submit_mode == "pbs" and not args.wait and proc.returncode == 0:
        job_id = submission.get("job_id")
        if job_id:
            print("已提交作业 job_id:", job_id)

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
