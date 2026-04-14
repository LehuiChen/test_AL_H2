#!/usr/bin/env python3
"""H2 active learning reproduction with MLatom + Gaussian (g16)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and "--legacy-runner" not in sys.argv:
    from h2_al_runner import main as _h2_al_main

    raise SystemExit(_h2_al_main(sys.argv[1:]))


def _best_effort_call(func: Any, kwargs: Dict[str, Any]) -> Any:
    """Call a function while removing unsupported keyword arguments on the fly."""
    call_kwargs = dict(kwargs)
    while True:
        try:
            return func(**call_kwargs)
        except TypeError as exc:
            message = str(exc)
            marker = "unexpected keyword argument '"
            if marker not in message:
                raise
            start = message.find(marker) + len(marker)
            end = message.find("'", start)
            if end <= start:
                raise
            bad_key = message[start:end]
            if bad_key not in call_kwargs:
                raise
            call_kwargs.pop(bad_key)


def _import_mlatom() -> Any:
    try:
        import mlatom as ml  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Cannot import mlatom. Activate the target virtual environment first."
        ) from exc
    return ml


def _resolve_path(path_value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_json(path: pathlib.Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_molecules(dataset_json_path: pathlib.Path) -> Optional[int]:
    obj = _load_json(dataset_json_path)
    if obj is None:
        return None
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for key in ("molecules", "entries", "data", "points"):
            value = obj.get(key)
            if isinstance(value, list):
                return len(value)
        if "ids" in obj and isinstance(obj["ids"], list):
            return len(obj["ids"])
    return None


def _get_molecule_class(ml: Any) -> Any:
    if hasattr(ml, "data") and hasattr(ml.data, "molecule"):
        return ml.data.molecule
    if hasattr(ml, "molecule"):
        return ml.molecule
    raise RuntimeError("Cannot find molecule class in mlatom module.")


def _build_reference_method(ml: Any, method: str, qmprog: str) -> Any:
    candidates = [
        {"method": method, "program": qmprog},
        {"method": method, "qmprog": qmprog},
        {"method": method},
    ]
    last_error: Optional[Exception] = None
    for kwargs in candidates:
        try:
            return _best_effort_call(ml.models.methods, kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to create reference method. Check refmethod/qmprog and MLatom version."
    ) from last_error


def _build_al_model(ml: Any) -> Any:
    candidates = [
        {
            "ml_program": "KREG_API",
            "hyperparameter_optimization": {"optimization_algorithm": "grid"},
        },
        {
            "program": "KREG_API",
            "hyperparameter_optimization": {"optimization_algorithm": "grid"},
        },
        {"hyperparameter_optimization": {"optimization_algorithm": "grid"}},
        {},
    ]
    last_error: Optional[Exception] = None
    for kwargs in candidates:
        try:
            return _best_effort_call(ml.models.mlatomf, kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Failed to create AL model.") from last_error


def _load_or_compute_eq_molecule(
    ml: Any,
    reference_method: Any,
    xyz_path: pathlib.Path,
    freq_json_path: pathlib.Path,
    force_recompute: bool,
) -> Tuple[Any, str]:
    molecule_cls = _get_molecule_class(ml)

    if not force_recompute and freq_json_path.exists():
        try:
            if hasattr(molecule_cls, "load"):
                eqmol = _best_effort_call(
                    molecule_cls.load,
                    {"filename": str(freq_json_path), "format": "json"},
                )
            else:
                raise RuntimeError("molecule.load() unavailable in this MLatom version.")
            return eqmol, "loaded"
        except Exception:
            pass

    if not xyz_path.exists():
        raise FileNotFoundError("Missing XYZ input file: %s" % xyz_path)
    if not hasattr(molecule_cls, "from_xyz_file"):
        raise RuntimeError("molecule.from_xyz_file() is unavailable.")

    molecule = _best_effort_call(molecule_cls.from_xyz_file, {"filename": str(xyz_path)})
    eqmol = _best_effort_call(reference_method.optimize_geometry, {"molecule": molecule})
    _best_effort_call(reference_method.predict, {"molecule": eqmol, "calculate_hessian": True})

    if hasattr(eqmol, "dump"):
        try:
            _best_effort_call(eqmol.dump, {"filename": str(freq_json_path), "format": "json"})
        except Exception:
            _best_effort_call(eqmol.dump, {"filename": str(freq_json_path)})

    return eqmol, "computed"


def _run_single_al_cycle(
    ml: Any,
    al_model: Any,
    reference_method: Any,
    eq_molecule: Any,
    sampled_points: int,
    init_train_set_size: int,
    min_new_points: int,
    uq_threshold: float,
    label_nthreads: int,
    initial_temperature: float,
    maximum_propagation_time: float,
    time_step: float,
    debug: bool,
) -> None:
    initcond_sampler_kwargs = {
        "molecule": eq_molecule,
        "number_of_initial_conditions": sampled_points,
        "initial_temperature": initial_temperature,
    }
    initdata_sampler_kwargs = {
        "molecule": eq_molecule,
        "number_of_initial_conditions": init_train_set_size,
        "initial_temperature": initial_temperature,
    }
    sampler_kwargs = {
        "initcond_sampler": "wigner",
        "initcond_sampler_kwargs": initcond_sampler_kwargs,
        "maximum_propagation_time": maximum_propagation_time,
        "time_step": time_step,
        "uq_threshold": uq_threshold,
    }

    al_kwargs = {
        "al_model": al_model,
        "reference_method": reference_method,
        "initcond_sampler": "wigner",
        "initcond_sampler_kwargs": initcond_sampler_kwargs,
        "maximum_propagation_time": maximum_propagation_time,
        "time_step": time_step,
        "model_predict_kwargs": {
            "calculate_energy": True,
            "calculate_energy_gradients": True,
        },
        "initial_points_refinement": "cross-validation",
        "initdata_sampler": "wigner",
        "initdata_sampler_kwargs": initdata_sampler_kwargs,
        "min_new_points": min_new_points,
        "retrain_each_iteration": False,
        "debug": debug,
        "sampler_kwargs": sampler_kwargs,
        "label_nthreads": label_nthreads,
        "max_iterations": 1,
    }
    _best_effort_call(ml.al, al_kwargs)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the H2 active learning workflow with MLatom and Gaussian. "
            "Implements smoke/full modes and adaptive uq_threshold control."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--run-dir", default="", help="Output directory for AL run artifacts.")
    parser.add_argument("--xyz", default=str(root / "inputs" / "h2.xyz"))
    parser.add_argument("--freq-json", default=str(root / "inputs" / "h2_freq.json"))
    parser.add_argument("--refmethod", default="B3LYP/6-31G*")
    parser.add_argument("--qmprog", default="gaussian")
    parser.add_argument("--init-train-set-size", type=int, default=5)
    parser.add_argument("--max-sampled-points", type=int, default=0)
    parser.add_argument("--min-new-points", type=int, default=5)
    parser.add_argument("--max-excess", type=float, default=0.25)
    parser.add_argument("--min-excess", type=float, default=0.20)
    parser.add_argument("--uq-threshold", type=float, default=1.0e-6)
    parser.add_argument("--threshold-expand-factor", type=float, default=1.25)
    parser.add_argument("--threshold-shrink-factor", type=float, default=0.85)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--label-nthreads", type=int, default=4)
    parser.add_argument("--initial-temperature", type=float, default=300.0)
    parser.add_argument("--maximum-propagation-time", type=float, default=5.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--force-recompute-freq", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    script_root = pathlib.Path(__file__).resolve().parents[1]
    default_run_name = "h2_smoke" if args.mode == "smoke" else "h2_full"
    run_dir = (
        _resolve_path(args.run_dir, pathlib.Path.cwd())
        if args.run_dir
        else (script_root / "runs" / default_run_name)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    sampled_points = args.max_sampled_points
    if sampled_points <= 0:
        sampled_points = 50 if args.mode == "smoke" else 1000

    xyz_path = _resolve_path(args.xyz, pathlib.Path.cwd())
    freq_json_path = _resolve_path(args.freq_json, pathlib.Path.cwd())

    config = {
        "mode": args.mode,
        "run_dir": str(run_dir),
        "xyz": str(xyz_path),
        "freq_json": str(freq_json_path),
        "refmethod": args.refmethod,
        "qmprog": args.qmprog,
        "init_train_set_size": args.init_train_set_size,
        "max_sampled_points": sampled_points,
        "min_new_points": args.min_new_points,
        "max_excess": args.max_excess,
        "min_excess": args.min_excess,
        "uq_threshold_initial": args.uq_threshold,
        "threshold_expand_factor": args.threshold_expand_factor,
        "threshold_shrink_factor": args.threshold_shrink_factor,
        "max_cycles": args.max_cycles,
        "label_nthreads": args.label_nthreads,
        "initial_temperature": args.initial_temperature,
        "maximum_propagation_time": args.maximum_propagation_time,
        "time_step": args.time_step,
        "force_recompute_freq": args.force_recompute_freq,
    }
    _write_json(run_dir / "run_config.json", config)

    if args.dry_run:
        print("Dry run only. Configuration written to:", run_dir / "run_config.json")
        return 0

    ml = _import_mlatom()
    reference_method = _build_reference_method(ml, args.refmethod, args.qmprog)
    eq_molecule, source = _load_or_compute_eq_molecule(
        ml=ml,
        reference_method=reference_method,
        xyz_path=xyz_path,
        freq_json_path=freq_json_path,
        force_recompute=args.force_recompute_freq,
    )
    al_model = _build_al_model(ml)

    history: List[Dict[str, Any]] = []
    uq_threshold = args.uq_threshold
    old_cwd = pathlib.Path.cwd()
    labeled_db_path = run_dir / "labeled_db.json"

    try:
        os.chdir(run_dir)

        previous_count = _count_molecules(labeled_db_path)

        for cycle in range(1, args.max_cycles + 1):
            cycle_start = time.time()
            _run_single_al_cycle(
                ml=ml,
                al_model=al_model,
                reference_method=reference_method,
                eq_molecule=eq_molecule,
                sampled_points=sampled_points,
                init_train_set_size=args.init_train_set_size,
                min_new_points=args.min_new_points,
                uq_threshold=uq_threshold,
                label_nthreads=args.label_nthreads,
                initial_temperature=args.initial_temperature,
                maximum_propagation_time=args.maximum_propagation_time,
                time_step=args.time_step,
                debug=args.debug,
            )

            current_count = _count_molecules(labeled_db_path)
            new_points: Optional[int] = None
            if current_count is not None:
                baseline = previous_count if previous_count is not None else 0
                new_points = max(current_count - baseline, 0)

            cycle_info: Dict[str, Any] = {
                "cycle": cycle,
                "uq_threshold": uq_threshold,
                "labeled_count_before": previous_count,
                "labeled_count_after": current_count,
                "new_points": new_points,
                "elapsed_seconds": round(time.time() - cycle_start, 3),
                "eq_molecule_source": source,
            }

            if new_points is not None:
                excess = float(new_points) / float(sampled_points)
                cycle_info["excess"] = excess
                if excess > args.max_excess:
                    uq_threshold *= args.threshold_expand_factor
                elif excess < args.min_excess:
                    uq_threshold *= args.threshold_shrink_factor
                uq_threshold = max(uq_threshold, 1.0e-14)

            history.append(cycle_info)
            _write_json(run_dir / "al_iteration_history.json", history)

            print(
                "[cycle %d] new_points=%s uq_threshold=%.3e"
                % (cycle, str(new_points), cycle_info["uq_threshold"])
            )

            previous_count = current_count

            if new_points is None:
                print("Could not infer newly labeled points; stopping for safety.")
                break
            if new_points < args.min_new_points:
                print(
                    "Stop condition reached: new_points (%d) < min_new_points (%d)."
                    % (new_points, args.min_new_points)
                )
                break

    finally:
        os.chdir(old_cwd)

    print("Run directory:", run_dir)
    print("Config file:", run_dir / "run_config.json")
    print("Iteration history:", run_dir / "al_iteration_history.json")
    print("Expected AL info file:", run_dir / "al_info.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
