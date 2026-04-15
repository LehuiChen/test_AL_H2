#!/usr/bin/env python3
"""ANI-only H2 active learning runner with single-job PBS submission."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: pathlib.Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required. Install it in your PBS runtime env: `python3 -m pip install pyyaml`."
        ) from exc
    return yaml


def _best_effort_call(func: Any, kwargs: Dict[str, Any]) -> Any:
    """Call with kwargs and strip unsupported keyword arguments automatically."""
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


def _normalize_shell_lines(block: Any) -> List[str]:
    if block is None:
        return []
    if isinstance(block, str):
        return [line.strip() for line in block.splitlines() if line.strip()]
    if isinstance(block, list):
        lines: List[str] = []
        for item in block:
            text = str(item).strip()
            if text:
                lines.append(text)
        return lines
    return []


def _resolve_repo_path(value: str, project_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(str(value))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _require_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError("Invalid integer for `%s`: %r" % (key, value)) from exc


def _require_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError("Invalid float for `%s`: %r" % (key, value)) from exc


def _load_config(config_path: str, project_root: pathlib.Path) -> Dict[str, Any]:
    yaml = _import_yaml()
    path = pathlib.Path(config_path)
    if not path.is_absolute():
        path = (project_root / path).resolve()

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Config must be a YAML mapping: %s" % path)

    payload.setdefault("project", {})
    payload.setdefault("run", {})
    payload.setdefault("system", {})
    payload.setdefault("reference", {})
    payload.setdefault("training", {})
    payload.setdefault("active_learning", {})
    payload.setdefault("uncertainty", {})
    payload.setdefault("cluster", {})
    payload["_config_file"] = str(path)
    payload["_project_root"] = str(project_root.resolve())
    return payload


def _default_run_name(config: Dict[str, Any], mode: str) -> str:
    run_cfg = config.get("run", {})
    name_key = "smoke_name" if mode == "smoke" else "full_name"
    if run_cfg.get(name_key):
        return str(run_cfg[name_key])
    prefix = str(run_cfg.get("name_prefix", "h2_ani"))
    return "%s_%s" % (prefix, mode)


def _resolve_run_dir(args: argparse.Namespace, config: Dict[str, Any], project_root: pathlib.Path) -> pathlib.Path:
    if args.run_dir:
        return pathlib.Path(args.run_dir).resolve()
    run_cfg = config.get("run", {})
    runs_dir = _resolve_repo_path(str(run_cfg.get("runs_dir", "runs")), project_root)
    return (runs_dir / _default_run_name(config, args.mode)).resolve()


def _max_sampled_points(config: Dict[str, Any], mode: str) -> int:
    al_cfg = config.get("active_learning", {})
    raw_value = al_cfg.get("max_sampled_points", {})
    if isinstance(raw_value, dict):
        mode_value = raw_value.get(mode)
    else:
        mode_value = raw_value
    if mode_value is None:
        mode_value = 50 if mode == "smoke" else 1000
    return _require_int(mode_value, "active_learning.max_sampled_points")


def _import_mlatom() -> Any:
    try:
        import mlatom as ml  # type: ignore
    except Exception as exc:
        raise RuntimeError("Cannot import mlatom. Activate your ANI runtime env before running.") from exc
    return ml


def _new_molecule(ml: Any) -> Any:
    if hasattr(ml, "data") and hasattr(ml.data, "molecule"):
        return ml.data.molecule()
    if hasattr(ml, "molecule"):
        return ml.molecule()
    raise RuntimeError("Cannot locate molecule class in current mlatom version.")


def _load_molecule_file(ml: Any, path: pathlib.Path, fmt: str) -> Any:
    molecule = _new_molecule(ml)
    try:
        molecule.load(str(path), format=fmt)
    except TypeError:
        molecule.load(str(path), fmt)
    return molecule


def _build_reference_method(ml: Any, reference_cfg: Dict[str, Any]) -> Any:
    method = str(reference_cfg.get("refmethod", "B3LYP/6-31G*"))
    qmprog = reference_cfg.get("qmprog", "gaussian")
    nthreads = _require_int(reference_cfg.get("nthreads", 8), "reference.nthreads")

    candidates = [
        {"method": method, "program": qmprog, "nthreads": nthreads, "save_files_in_current_directory": False},
        {"method": method, "qmprog": qmprog, "nthreads": nthreads, "save_files_in_current_directory": False},
        {"method": method, "program": qmprog, "nthreads": nthreads},
        {"method": method, "program": qmprog},
        {"method": method},
    ]
    last_error: Optional[Exception] = None
    for kwargs in candidates:
        try:
            return _best_effort_call(ml.models.methods, kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to construct reference method `%s` with qmprog `%s`." % (method, qmprog)
    ) from last_error


def _resolve_optimized_molecule(result: Any) -> Any:
    if hasattr(result, "optimized_molecule"):
        return result.optimized_molecule
    return result


def _load_or_compute_eq_molecule(
    ml: Any,
    reference_method: Any,
    xyz_path: pathlib.Path,
    freq_json_path: pathlib.Path,
    force_recompute: bool,
) -> Tuple[Any, str]:
    if (not force_recompute) and freq_json_path.exists():
        try:
            return _load_molecule_file(ml, freq_json_path, "json"), "loaded"
        except Exception:
            pass

    if not xyz_path.exists():
        raise FileNotFoundError("Missing system.xyz file: %s" % xyz_path)

    molecule = _load_molecule_file(ml, xyz_path, "xyz")
    optimization_result = _best_effort_call(reference_method.optimize_geometry, {"molecule": molecule})
    eq_molecule = _resolve_optimized_molecule(optimization_result)
    _best_effort_call(reference_method.predict, {"molecule": eq_molecule, "calculate_hessian": True})

    freq_json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _best_effort_call(eq_molecule.dump, {"filename": str(freq_json_path), "format": "json"})
    except Exception:
        _best_effort_call(eq_molecule.dump, {"filename": str(freq_json_path)})

    return eq_molecule, "computed"


def _build_ani_model_object(ml: Any, model_file: pathlib.Path, device: Optional[str]) -> Optional[Any]:
    constructors: List[Any] = []
    if hasattr(ml, "models"):
        if hasattr(ml.models, "ani"):
            constructors.append(ml.models.ani)
        if hasattr(ml.models, "torchani"):
            constructors.append(ml.models.torchani)
    if not constructors:
        return None

    kwargs_candidates = [
        {"model_file": str(model_file), "device": device},
        {"model_file": str(model_file)},
        {"device": device},
        {},
    ]

    for constructor in constructors:
        for kwargs in kwargs_candidates:
            cleaned = {k: v for k, v in kwargs.items() if v is not None}
            try:
                return _best_effort_call(constructor, cleaned)
            except Exception:
                continue
    return None


def _run_ani_al(
    ml: Any,
    eq_molecule: Any,
    reference_method: Any,
    run_dir: pathlib.Path,
    al_cfg: Dict[str, Any],
    uncertainty_cfg: Dict[str, Any],
    sampled_points: int,
    device: Optional[str],
    debug: bool,
) -> Dict[str, Any]:
    init_train_set_size = _require_int(al_cfg.get("init_train_set_size", 5), "active_learning.init_train_set_size")
    min_new_points = _require_int(al_cfg.get("min_new_points", 5), "active_learning.min_new_points")
    max_excess = _require_float(al_cfg.get("max_excess", 0.25), "active_learning.max_excess")
    min_excess = _require_float(al_cfg.get("min_excess", 0.20), "active_learning.min_excess")
    initial_temperature = _require_float(al_cfg.get("initial_temperature", 300.0), "active_learning.initial_temperature")
    max_prop_time = _require_float(
        al_cfg.get("maximum_propagation_time", 5.0), "active_learning.maximum_propagation_time"
    )
    time_step = _require_float(al_cfg.get("time_step", 0.1), "active_learning.time_step")
    label_nthreads = _require_int(al_cfg.get("label_nthreads", 4), "active_learning.label_nthreads")
    initial_points_refinement = str(al_cfg.get("initial_points_refinement", "cross-validation"))
    committee_size = _require_int(uncertainty_cfg.get("committee_size", 2), "uncertainty.committee_size")
    uq_metric = str(uncertainty_cfg.get("metric", "energy_forces"))
    energy_weight = _require_float(uncertainty_cfg.get("energy_weight", 1.0), "uncertainty.energy_weight")
    force_weight = _require_float(uncertainty_cfg.get("force_weight", 1.0), "uncertainty.force_weight")
    uq_threshold_raw = uncertainty_cfg.get("uq_threshold")
    uq_threshold: Optional[float] = None
    if uq_threshold_raw is not None:
        uq_threshold = _require_float(uq_threshold_raw, "uncertainty.uq_threshold")
    committee_size = max(2, committee_size)

    if device is not None:
        device = str(device)

    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "ani_main_model.pt"
    aux_model_files = [model_dir / ("ani_aux_model_%02d.pt" % idx) for idx in range(1, committee_size)]
    model_kwargs: Dict[str, Any] = {"model_file": str(model_file)}
    if device:
        model_kwargs["device"] = device
    aux_model_kwargs_list: List[Dict[str, Any]] = []
    for aux_path in aux_model_files:
        aux_kwargs: Dict[str, Any] = {"model_file": str(aux_path)}
        if device:
            aux_kwargs["device"] = device
        aux_model_kwargs_list.append(aux_kwargs)
    primary_aux_model_kwargs = aux_model_kwargs_list[0] if aux_model_kwargs_list else dict(model_kwargs)

    initdata_sampler_kwargs = {
        "molecule": eq_molecule,
        "number_of_initial_conditions": init_train_set_size,
        "initial_temperature": initial_temperature,
    }
    sampling_pool_kwargs = {
        "molecule": eq_molecule,
        "number_of_initial_conditions": sampled_points,
        "initial_temperature": initial_temperature,
    }
    sampler_kwargs: Dict[str, Any] = {
        "initcond_sampler": "wigner",
        "initcond_sampler_kwargs": sampling_pool_kwargs,
        "maximum_propagation_time": max_prop_time,
        "time_step": time_step,
        "max_excess": max_excess,
        "min_excess": min_excess,
        "uq_metric": uq_metric,
        "energy_weight": energy_weight,
        "force_weight": force_weight,
    }
    if uq_threshold is not None:
        sampler_kwargs["uq_threshold"] = uq_threshold

    base_kwargs: Dict[str, Any] = {
        "reference_method": reference_method,
        "label_nthreads": label_nthreads,
        "initdata_sampler": "wigner",
        "initdata_sampler_kwargs": initdata_sampler_kwargs,
        "initial_points_refinement": initial_points_refinement,
        "sampler": "batch_md",
        "sampler_kwargs": sampler_kwargs,
        "max_sampled_points": sampled_points,
        "min_new_points": min_new_points,
        "max_excess": max_excess,
        "min_excess": min_excess,
        "uq_metric": uq_metric,
        "debug": debug,
        "molecule": eq_molecule,
        "model_predict_kwargs": {
            "calculate_energy": True,
            "calculate_energy_gradients": True,
        },
    }
    if uq_threshold is not None:
        base_kwargs["uq_threshold"] = uq_threshold

    kwargs_variants: List[Tuple[str, Dict[str, Any]]] = [
        (
            "ml_model_main_aux_kwargs",
            dict(
                base_kwargs,
                ml_model="ANI",
                ml_model_kwargs=model_kwargs,
                ml_model_aux="ANI",
                ml_model_aux_kwargs=primary_aux_model_kwargs,
                committee_size=committee_size,
            ),
        ),
        (
            "ml_model_uq_kwargs",
            dict(
                base_kwargs,
                ml_model="ANI",
                ml_model_kwargs=model_kwargs,
                ml_model_uq="ANI",
                ml_model_uq_kwargs=primary_aux_model_kwargs,
                committee_size=committee_size,
            ),
        ),
        (
            "ml_models_list",
            dict(
                base_kwargs,
                ml_models=["ANI", "ANI"],
                ml_models_kwargs=[model_kwargs, primary_aux_model_kwargs],
                committee_size=committee_size,
            ),
        ),
        (
            "ml_model_string_with_kwargs",
            dict(base_kwargs, ml_model="ANI", ml_model_kwargs=model_kwargs),
        ),
        (
            "ml_model_string_only",
            dict(base_kwargs, ml_model="ANI"),
        ),
    ]

    ani_model_obj = _build_ani_model_object(ml, model_file=model_file, device=device)
    aux_model_objs: List[Any] = []
    for aux_path in aux_model_files:
        aux_obj = _build_ani_model_object(ml, model_file=aux_path, device=device)
        if aux_obj is not None:
            aux_model_objs.append(aux_obj)
    if ani_model_obj is not None and aux_model_objs:
        kwargs_variants.extend(
            [
                (
                    "explicit_main_aux_model_objects",
                    dict(
                        base_kwargs,
                        al_model=ani_model_obj,
                        aux_model=aux_model_objs[0],
                        committee_size=committee_size,
                    ),
                ),
                (
                    "explicit_main_uq_model_objects",
                    dict(
                        base_kwargs,
                        al_model=ani_model_obj,
                        uq_model=aux_model_objs[0],
                        committee_size=committee_size,
                    ),
                ),
                (
                    "explicit_model_list_objects",
                    dict(
                        base_kwargs,
                        al_models=[ani_model_obj, aux_model_objs[0]],
                        committee_size=committee_size,
                    ),
                ),
            ]
        )
    elif ani_model_obj is not None:
        kwargs_variants.append(("explicit_ani_model_object", dict(base_kwargs, al_model=ani_model_obj)))

    entrypoints: List[Tuple[str, Any]] = []
    if hasattr(ml, "al"):
        entrypoints.append(("al", ml.al))
    if hasattr(ml, "active_learning"):
        entrypoints.append(("active_learning", ml.active_learning))
    if not entrypoints:
        raise RuntimeError("Current mlatom build has neither `ml.al` nor `ml.active_learning` entrypoint.")

    failures: List[str] = []
    for entrypoint_name, fn in entrypoints:
        for variant_name, kwargs in kwargs_variants:
            try:
                _best_effort_call(fn, kwargs)
                return {
                    "entrypoint": entrypoint_name,
                    "variant": variant_name,
                    "ani_model_file": str(model_file.resolve()),
                    "aux_model_files": [str(path.resolve()) for path in aux_model_files],
                    "committee_size": committee_size,
                    "uncertainty_metric": uq_metric,
                    "uq_threshold": uq_threshold,
                    "device": device,
                }
            except Exception as exc:
                failures.append("%s / %s: %s: %s" % (entrypoint_name, variant_name, type(exc).__name__, exc))

    raise RuntimeError(
        "Failed to execute ANI AL call. Tried variants:\n- %s" % "\n- ".join(failures[:10])
    )


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


def _extract_iteration_history(al_info_obj: Any) -> List[Dict[str, Any]]:
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


def _build_shell_command(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_pbs_script(
    *,
    cluster_cfg: Dict[str, Any],
    command: str,
    workdir: pathlib.Path,
    job_name: str,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> str:
    queue = str(cluster_cfg.get("queue", "GPU"))
    nodes = _require_int(cluster_cfg.get("nodes", 1), "cluster.nodes")
    ppn = _require_int(cluster_cfg.get("ppn", 24), "cluster.ppn")
    walltime = str(cluster_cfg.get("walltime", "72:00:00"))
    extra_pbs_lines = _normalize_shell_lines(cluster_cfg.get("extra_pbs_lines"))
    env_blocks = cluster_cfg.get("env_blocks", {})
    if not isinstance(env_blocks, dict):
        env_blocks = {}
    env_lines = _normalize_shell_lines(env_blocks.get("default"))
    if not env_lines:
        for key in ("train", "training", "md", "labels"):
            env_lines = _normalize_shell_lines(env_blocks.get(key))
            if env_lines:
                break
    if not env_lines:
        env_lines = ["source ~/.bashrc", "conda activate ADL_env"]
    cleanup_lines = _normalize_shell_lines(cluster_cfg.get("cleanup_lines"))

    lines = [
        "#!/bin/bash",
        "#PBS -N %s" % job_name,
        "#PBS -q %s" % queue,
        "#PBS -l nodes=%d:ppn=%d" % (nodes, ppn),
        "#PBS -l walltime=%s" % walltime,
        "#PBS -o %s" % stdout_path,
        "#PBS -e %s" % stderr_path,
    ]
    lines.extend(extra_pbs_lines)
    lines.extend(["", "set -euo pipefail", "cd %s" % workdir, ""])

    if cleanup_lines:
        lines.extend(["cleanup_job() {"])
        lines.extend(["  %s" % line for line in cleanup_lines])
        lines.extend(["}", "trap cleanup_job EXIT", ""])

    lines.extend(env_lines)
    lines.extend(["", command, ""])
    return "\n".join(lines)


def _wait_for_status_file(
    status_file: pathlib.Path,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> Dict[str, Any]:
    start = time.time()
    while True:
        if status_file.exists():
            payload = _load_json(status_file)
            if isinstance(payload, dict):
                return payload
            raise RuntimeError("Status file exists but is not valid JSON: %s" % status_file)
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timeout while waiting for status file: %s" % status_file)
        time.sleep(max(1, poll_interval_seconds))


def _run_local(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    run_dir: pathlib.Path,
    status_file: pathlib.Path,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)

    project_root = pathlib.Path(config["_project_root"]).resolve()
    system_cfg = config.get("system", {})
    reference_cfg = config.get("reference", {})
    al_cfg = config.get("active_learning", {})
    uncertainty_cfg = config.get("uncertainty", {})
    training_cfg = config.get("training", {})

    model_type = str(training_cfg.get("ml_model_type", "ANI")).strip().upper()
    if model_type != "ANI":
        raise RuntimeError("This runner is ANI-only. Set training.ml_model_type: ANI in config.")

    xyz_path = _resolve_repo_path(str(system_cfg.get("xyz", "inputs/h2.xyz")), project_root)
    freq_json_path = _resolve_repo_path(str(system_cfg.get("freq_json", "inputs/h2_freq.json")), project_root)
    sampled_points = _max_sampled_points(config, args.mode)

    runtime_config = {
        "generated_at": _timestamp(),
        "config_file": config["_config_file"],
        "mode": args.mode,
        "submit_mode": "local",
        "run_dir": str(run_dir),
        "system": {"xyz": str(xyz_path), "freq_json": str(freq_json_path)},
        "reference": {
            "refmethod": reference_cfg.get("refmethod", "B3LYP/6-31G*"),
            "qmprog": reference_cfg.get("qmprog", "gaussian"),
            "nthreads": reference_cfg.get("nthreads", 8),
        },
        "active_learning": {
            "init_train_set_size": al_cfg.get("init_train_set_size", 5),
            "max_sampled_points": sampled_points,
            "min_new_points": al_cfg.get("min_new_points", 5),
            "max_excess": al_cfg.get("max_excess", 0.25),
            "min_excess": al_cfg.get("min_excess", 0.20),
            "initial_temperature": al_cfg.get("initial_temperature", 300.0),
            "maximum_propagation_time": al_cfg.get("maximum_propagation_time", 5.0),
            "time_step": al_cfg.get("time_step", 0.1),
            "label_nthreads": al_cfg.get("label_nthreads", 4),
            "initial_points_refinement": al_cfg.get("initial_points_refinement", "cross-validation"),
        },
        "uncertainty": {
            "committee_size": uncertainty_cfg.get("committee_size", 2),
            "metric": uncertainty_cfg.get("metric", "energy_forces"),
            "uq_threshold": uncertainty_cfg.get("uq_threshold"),
            "energy_weight": uncertainty_cfg.get("energy_weight", 1.0),
            "force_weight": uncertainty_cfg.get("force_weight", 1.0),
        },
        "training": {"ml_model_type": model_type, "device": training_cfg.get("device", "cuda")},
    }
    _write_json(run_dir / "run_config.json", runtime_config)

    if args.dry_run:
        payload = {
            "success": True,
            "dry_run": True,
            "mode": args.mode,
            "run_dir": str(run_dir),
            "run_config_file": str((run_dir / "run_config.json").resolve()),
            "status_file": str(status_file.resolve()),
            "generated_at": _timestamp(),
        }
        _write_json(status_file, payload)
        print("Dry run complete. Config written to:", run_dir / "run_config.json")
        return 0

    old_cwd = pathlib.Path.cwd()
    try:
        ml = _import_mlatom()
        reference_method = _build_reference_method(ml, reference_cfg)
        eq_molecule, eq_source = _load_or_compute_eq_molecule(
            ml=ml,
            reference_method=reference_method,
            xyz_path=xyz_path,
            freq_json_path=freq_json_path,
            force_recompute=args.force_recompute_freq,
        )

        os.chdir(run_dir)
        al_meta = _run_ani_al(
            ml=ml,
            eq_molecule=eq_molecule,
            reference_method=reference_method,
            run_dir=run_dir,
            al_cfg=runtime_config["active_learning"],
            uncertainty_cfg=runtime_config["uncertainty"],
            sampled_points=sampled_points,
            device=training_cfg.get("device"),
            debug=args.debug,
        )

        al_info_path = run_dir / "al_info.json"
        al_info_obj = _load_json(al_info_path)
        history = _extract_iteration_history(al_info_obj)
        history_path = run_dir / "al_iteration_history.json"
        if history:
            _write_json(history_path, history)

        status_payload = {
            "success": True,
            "mode": args.mode,
            "run_dir": str(run_dir),
            "eq_molecule_source": eq_source,
            "al_entrypoint": al_meta.get("entrypoint"),
            "al_variant": al_meta.get("variant"),
            "al_info_file": str(al_info_path.resolve()),
            "al_info_exists": al_info_path.exists(),
            "iteration_history_file": str(history_path.resolve()) if history else None,
            "iteration_history_length": len(history),
            "uncertainty": {
                "committee_size": al_meta.get("committee_size"),
                "metric": al_meta.get("uncertainty_metric"),
                "uq_threshold": al_meta.get("uq_threshold"),
                "aux_model_files": al_meta.get("aux_model_files"),
            },
            "generated_at": _timestamp(),
        }
        _write_json(status_file, status_payload)

        print("AL run finished. Run directory:", run_dir)
        print("Status file:", status_file)
        print("AL info file:", al_info_path)
        return 0
    except Exception as exc:
        failure = {
            "success": False,
            "mode": args.mode,
            "run_dir": str(run_dir),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "generated_at": _timestamp(),
        }
        _write_json(status_file, failure)
        print("AL run failed. Status file:", status_file, file=sys.stderr)
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    finally:
        os.chdir(old_cwd)


def _run_pbs(
    *,
    args: argparse.Namespace,
    config: Dict[str, Any],
    run_dir: pathlib.Path,
    status_file: pathlib.Path,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    cluster_cfg = config.get("cluster", {})
    project_root = pathlib.Path(config["_project_root"]).resolve()
    python_command = str(cluster_cfg.get("python_command", "python3"))
    submit_command = str(cluster_cfg.get("submit_command", "qsub"))

    job_name = str(cluster_cfg.get("job_name", "h2_ani_al")) + "_" + args.mode
    job_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    job_dir = run_dir / "jobs" / ("%s_%s" % (args.mode, job_stamp))
    job_dir.mkdir(parents=True, exist_ok=True)
    script_file = job_dir / "job.pbs"
    stdout_file = job_dir / "stdout.log"
    stderr_file = job_dir / "stderr.log"

    if status_file.exists():
        try:
            status_file.unlink()
        except Exception:
            pass

    launcher = pathlib.Path(__file__).resolve().parent / "run_h2_al.py"
    command_parts = [
        python_command,
        str(launcher.resolve()),
        "--config",
        str(pathlib.Path(config["_config_file"]).resolve()),
        "--mode",
        args.mode,
        "--submit-mode",
        "local",
        "--run-dir",
        str(run_dir.resolve()),
        "--status-file",
        str(status_file.resolve()),
    ]
    if args.force_recompute_freq:
        command_parts.append("--force-recompute-freq")
    if args.debug:
        command_parts.append("--debug")

    pbs_text = _build_pbs_script(
        cluster_cfg=cluster_cfg,
        command=_build_shell_command(command_parts),
        workdir=project_root,
        job_name=job_name,
        stdout_path=stdout_file.resolve(),
        stderr_path=stderr_file.resolve(),
    )
    script_file.write_text(pbs_text, encoding="utf-8")

    submission_file = run_dir / "submission.json"
    submission_payload = {
        "generated_at": _timestamp(),
        "mode": args.mode,
        "run_dir": str(run_dir),
        "status_file": str(status_file.resolve()),
        "job_name": job_name,
        "job_script": str(script_file.resolve()),
        "stdout_file": str(stdout_file.resolve()),
        "stderr_file": str(stderr_file.resolve()),
        "submit_command": submit_command,
        "python_command": python_command,
        "submitted": False,
        "wait": bool(args.wait),
    }

    if args.dry_run:
        _write_json(submission_file, submission_payload)
        print("Dry-run PBS script generated:", script_file)
        print("Submission preview:", submission_file)
        return 0

    result = subprocess.run(
        [submit_command, str(script_file.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip()
    submission_payload["submitted"] = True
    submission_payload["job_id"] = job_id
    submission_payload["submit_stdout"] = result.stdout.strip()
    submission_payload["submit_stderr"] = result.stderr.strip()
    _write_json(submission_file, submission_payload)

    print("Submitted PBS job:", job_id)
    print("Submission file:", submission_file)
    print("Status file:", status_file)

    if not args.wait:
        return 0

    status_payload = _wait_for_status_file(
        status_file.resolve(),
        timeout_seconds=_require_int(cluster_cfg.get("poll_timeout_seconds", 172800), "cluster.poll_timeout_seconds"),
        poll_interval_seconds=_require_int(cluster_cfg.get("poll_interval_seconds", 30), "cluster.poll_interval_seconds"),
    )
    success = bool(status_payload.get("success", False))
    if success:
        print("PBS local run completed successfully.")
        return 0

    print("PBS local run completed with failure. See status file:", status_file, file=sys.stderr)
    return 1


def _parse_args(argv: List[str]) -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="ANI-only H2 AL runner with one-command PBS submission."
    )
    parser.add_argument("--config", default=str(repo_root / "configs" / "base.yaml"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--submit-mode", choices=("local", "pbs"), default="pbs")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force-recompute-freq", action="store_true")
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument("--wait", dest="wait", action="store_true")
    wait_group.add_argument("--no-wait", dest="wait", action="store_false")
    parser.set_defaults(wait=False)
    parser.add_argument("--status-file", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(list(argv or []))
    project_root = pathlib.Path(__file__).resolve().parents[1]
    config = _load_config(args.config, project_root)
    run_dir = _resolve_run_dir(args, config, project_root)
    status_file = pathlib.Path(args.status_file).resolve() if args.status_file else (run_dir / "status.json").resolve()

    if args.submit_mode == "local":
        return _run_local(args=args, config=config, run_dir=run_dir, status_file=status_file)
    return _run_pbs(args=args, config=config, run_dir=run_dir, status_file=status_file)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
