#!/usr/bin/env python3
"""Inspect and summarize H2 active learning run outputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and "--legacy-runner" not in sys.argv:
    from inspect_al_results_core import main as _inspect_main

    raise SystemExit(_inspect_main(sys.argv[1:]))


def _load_json(path: pathlib.Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_new_points_from_history(history: List[Dict[str, Any]]) -> List[int]:
    values: List[int] = []
    for item in history:
        value = item.get("new_points")
        if isinstance(value, int):
            values.append(value)
    return values


def _write_curve_csv(history: List[Dict[str, Any]], target: pathlib.Path) -> None:
    lines = ["cycle,new_points,uq_threshold,excess"]
    for item in history:
        cycle = item.get("cycle", "")
        new_points = item.get("new_points", "")
        uq_threshold = item.get("uq_threshold", "")
        excess = item.get("excess", "")
        lines.append(f"{cycle},{new_points},{uq_threshold},{excess}")
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
    run_dir: pathlib.Path,
    history: List[Dict[str, Any]],
    min_new_points: int,
) -> Tuple[bool, List[str]]:
    checks: List[str] = []
    ok = True

    al_info_path = run_dir / "al_info.json"
    if al_info_path.exists():
        checks.append("PASS: al_info.json exists.")
    else:
        checks.append("FAIL: al_info.json is missing.")
        ok = False

    if history:
        checks.append("PASS: al_iteration_history.json contains iterations.")
    else:
        checks.append("FAIL: no iteration history found.")
        ok = False
        return ok, checks

    new_points = _extract_new_points_from_history(history)
    if new_points:
        trend = _trend_label(new_points)
        checks.append(f"INFO: new_points trend is {trend} ({new_points}).")
        if new_points[-1] < min_new_points:
            checks.append(
                "PASS: stop condition reached (last new_points=%d < min_new_points=%d)."
                % (new_points[-1], min_new_points)
            )
        else:
            checks.append(
                "FAIL: stop condition not reached (last new_points=%d, min_new_points=%d)."
                % (new_points[-1], min_new_points)
            )
            ok = False
    else:
        checks.append("FAIL: cannot read new_points from history.")
        ok = False

    return ok, checks


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect H2 AL run results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-new-points", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    run_dir = pathlib.Path(args.run_dir).resolve()

    history_path = run_dir / "al_iteration_history.json"
    history_obj = _load_json(history_path)
    history: List[Dict[str, Any]] = history_obj if isinstance(history_obj, list) else []

    csv_path = run_dir / "new_points_curve.csv"
    _write_curve_csv(history, csv_path)

    print("Run directory:", run_dir)
    print("History file:", history_path)
    print("Curve CSV:", csv_path)

    ok, checks = _evaluate_acceptance(run_dir, history, args.min_new_points)
    for line in checks:
        print(line)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
