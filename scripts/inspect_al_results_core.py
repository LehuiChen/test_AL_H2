#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: pathlib.Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_curve_csv(rounds: List[Dict[str, Any]], target: pathlib.Path) -> None:
    lines = ["round_index,selected_count,uncertain_ratio,converged"]
    for item in rounds:
        lines.append(
            "%s,%s,%s,%s"
            % (
                item.get("round_index", ""),
                item.get("selected_count", ""),
                item.get("uncertain_ratio", ""),
                item.get("converged", ""),
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trend_label(values: List[int]) -> str:
    if len(values) < 2:
        return "insufficient data"
    if values[-1] < values[0]:
        return "decreasing"
    if values[-1] == values[0]:
        return "flat"
    return "increasing"


def _evaluate_acceptance(
    *,
    results_dir: pathlib.Path,
    experiment_obj: Optional[Dict[str, Any]],
    history_obj: Optional[Dict[str, Any]],
    min_new_points: int,
) -> Tuple[bool, List[str]]:
    checks: List[str] = []
    ok = True

    experiment_path = results_dir / "active_learning_experiment_summary.json"
    if experiment_path.exists():
        checks.append("PASS: active_learning_experiment_summary.json exists.")
    else:
        checks.append("FAIL: active_learning_experiment_summary.json is missing.")
        ok = False

    if isinstance(experiment_obj, dict):
        if bool(experiment_obj.get("success", False)):
            checks.append("PASS: experiment summary reports success.")
        else:
            checks.append("WARN: experiment summary does not report success yet.")

    rounds = []
    if isinstance(history_obj, dict):
        rounds = history_obj.get("rounds", []) or []
    if not rounds:
        checks.append("FAIL: active_learning_round_history.json has no rounds.")
        return False, checks

    selected_counts = [int(item["selected_count"]) for item in rounds if isinstance(item.get("selected_count"), int)]
    if not selected_counts:
        checks.append("FAIL: cannot parse selected_count from round history.")
        return False, checks

    checks.append(f"INFO: selected_count trend is {_trend_label(selected_counts)} ({selected_counts}).")
    last_round = rounds[-1]
    if bool(last_round.get("converged", False)) or selected_counts[-1] < min_new_points:
        checks.append(
            "PASS: stop condition reached (converged=%s, last selected_count=%d, min_new_points=%d)."
            % (bool(last_round.get("converged", False)), selected_counts[-1], min_new_points)
        )
    else:
        checks.append(
            "FAIL: stop condition not reached (converged=%s, last selected_count=%d, min_new_points=%d)."
            % (bool(last_round.get("converged", False)), selected_counts[-1], min_new_points)
        )
        ok = False

    return ok, checks


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect H2 AL results under results/.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-dir", default="", help="兼容旧参数；如果传入则等同 results-dir。")
    parser.add_argument("--min-new-points", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(list(argv or []))
    base_dir = pathlib.Path(args.run_dir or args.results_dir).resolve()

    experiment_path = base_dir / "active_learning_experiment_summary.json"
    history_path = base_dir / "active_learning_round_history.json"
    experiment_obj_raw = _load_json(experiment_path)
    history_obj_raw = _load_json(history_path)
    experiment_obj = experiment_obj_raw if isinstance(experiment_obj_raw, dict) else None
    history_obj = history_obj_raw if isinstance(history_obj_raw, dict) else None
    rounds = history_obj.get("rounds", []) if history_obj else []

    curve_csv = base_dir / "new_points_curve.csv"
    _write_curve_csv(rounds, curve_csv)

    print("Results directory:", base_dir)
    print("Experiment summary:", experiment_path)
    print("Round history:", history_path)
    print("Curve CSV:", curve_csv)

    ok, checks = _evaluate_acceptance(
        results_dir=base_dir,
        experiment_obj=experiment_obj,
        history_obj=history_obj,
        min_new_points=args.min_new_points,
    )
    for line in checks:
        print(line)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
