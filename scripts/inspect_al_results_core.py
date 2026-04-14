#!/usr/bin/env python3
"""Inspect ANI H2 AL outputs with al_info.json-first logic."""

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


def _find_lists_of_dicts(obj: Any) -> List[List[Dict[str, Any]]]:
    found: List[List[Dict[str, Any]]] = []
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj):
            found.append(obj)
        for item in obj:
            found.extend(_find_lists_of_dicts(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            found.extend(_find_lists_of_dicts(value))
    return found


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_new_points(record: Dict[str, Any]) -> Optional[int]:
    for key in (
        "new_points",
        "n_new_points",
        "num_new_points",
        "number_of_new_points",
        "selected_count",
        "selected_points",
    ):
        if key in record:
            return _coerce_int(record.get(key))
    return None


def _extract_history_from_al_info(al_info_obj: Any) -> List[Dict[str, Any]]:
    candidates = _find_lists_of_dicts(al_info_obj)
    if not candidates:
        return []

    def score(items: List[Dict[str, Any]]) -> Tuple[int, int]:
        has_new_points = sum(1 for item in items if _extract_new_points(item) is not None)
        return has_new_points, len(items)

    best = sorted(candidates, key=score, reverse=True)[0]
    history: List[Dict[str, Any]] = []
    for index, item in enumerate(best, start=1):
        cycle = (
            _coerce_int(item.get("cycle"))
            or _coerce_int(item.get("iteration"))
            or _coerce_int(item.get("round"))
            or index
        )
        history.append(
            {
                "cycle": cycle,
                "new_points": _extract_new_points(item),
                "uq_threshold": item.get("uq_threshold"),
                "excess": item.get("excess"),
            }
        )
    return history


def _write_curve_csv(history: List[Dict[str, Any]], target: pathlib.Path) -> None:
    lines = ["cycle,new_points,uq_threshold,excess"]
    for item in history:
        lines.append(
            "%s,%s,%s,%s"
            % (
                item.get("cycle", ""),
                item.get("new_points", ""),
                item.get("uq_threshold", ""),
                item.get("excess", ""),
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
    run_dir: pathlib.Path,
    history: List[Dict[str, Any]],
    min_new_points: int,
    status_obj: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    checks: List[str] = []
    ok = True

    al_info_path = run_dir / "al_info.json"
    if al_info_path.exists():
        checks.append("PASS: al_info.json exists.")
    else:
        checks.append("FAIL: al_info.json is missing.")
        ok = False

    if status_obj is not None:
        if bool(status_obj.get("success", False)):
            checks.append("PASS: status.json reports success.")
        else:
            checks.append("FAIL: status.json reports failure.")
            ok = False

    if not history:
        checks.append("WARN: iteration history not found in al_info/history files.")
        return ok, checks

    new_points = [int(item["new_points"]) for item in history if isinstance(item.get("new_points"), int)]
    if not new_points:
        checks.append("WARN: history exists but no parseable new_points values.")
        return ok, checks

    trend = _trend_label(new_points)
    checks.append("INFO: new_points trend is %s (%s)." % (trend, new_points))
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

    return ok, checks


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ANI H2 AL run results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-new-points", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(list(argv or []))
    run_dir = pathlib.Path(args.run_dir).resolve()

    al_info_path = run_dir / "al_info.json"
    history_path = run_dir / "al_iteration_history.json"
    status_path = run_dir / "status.json"

    al_info_obj = _load_json(al_info_path)
    history_obj = _load_json(history_path)
    status_obj_raw = _load_json(status_path)
    status_obj = status_obj_raw if isinstance(status_obj_raw, dict) else None

    history: List[Dict[str, Any]] = []
    if isinstance(history_obj, list) and all(isinstance(item, dict) for item in history_obj):
        history = history_obj
    elif al_info_obj is not None:
        history = _extract_history_from_al_info(al_info_obj)

    curve_csv = run_dir / "new_points_curve.csv"
    _write_curve_csv(history, curve_csv)

    print("Run directory:", run_dir)
    print("AL info file:", al_info_path)
    print("Status file:", status_path)
    print("History file:", history_path)
    print("Curve CSV:", curve_csv)

    ok, checks = _evaluate_acceptance(
        run_dir=run_dir,
        history=history,
        min_new_points=args.min_new_points,
        status_obj=status_obj,
    )
    for line in checks:
        print(line)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
