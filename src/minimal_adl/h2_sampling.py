from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .direct_training import create_direct_model_bundle, load_training_state
from .geometry import GeometryRecord, load_manifest, save_xyz, write_manifest
from .io_utils import ensure_dir, read_json, timestamp_string, write_json
from .round_history import rebuild_round_history


@dataclass
class FrameRecord:
    sample_id: str
    trajectory_id: str
    initcond_id: str
    direction: str
    round_index: int
    frame_index: int
    time_fs: float
    uq: float | None
    exceeds_threshold: bool
    predicted_total_energy: float | None
    symbols: list[str]
    atomic_numbers: list[int]
    coordinates: np.ndarray
    charge: int
    multiplicity: int
    trajectory_xyz_file: str
    trajectory_summary_file: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "initcond_id": self.initcond_id,
            "direction": self.direction,
            "round_index": self.round_index,
            "frame_index": self.frame_index,
            "time_fs": self.time_fs,
            "uncertainty": self.uq,
            "exceeds_threshold": self.exceeds_threshold,
            "predicted_total_energy": self.predicted_total_energy,
            "symbols": list(self.symbols),
            "atomic_numbers": list(self.atomic_numbers),
            "coordinates": np.asarray(self.coordinates, dtype=float).tolist(),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "source_kind": "md_frame",
            "trajectory_xyz_file": self.trajectory_xyz_file,
            "trajectory_summary_file": self.trajectory_summary_file,
        }


def _resolve_generate_initial_conditions(ml):
    if hasattr(ml, "generate_initial_conditions"):
        return ml.generate_initial_conditions
    initial_conditions_module = getattr(ml, "initial_conditions", None)
    if initial_conditions_module is not None and hasattr(initial_conditions_module, "generate_initial_conditions"):
        return initial_conditions_module.generate_initial_conditions
    raise RuntimeError("当前 MLatom 安装没有暴露 generate_initial_conditions。")


def _molecule_symbols(molecule) -> list[str]:
    return [str(atom.element_symbol) for atom in molecule.atoms]


def _molecule_atomic_numbers(molecule) -> list[int]:
    return [int(atom.atomic_number) for atom in molecule.atoms]


def _molecule_charge(molecule) -> int:
    return int(getattr(molecule, "charge", 0))


def _molecule_multiplicity(molecule) -> int:
    return int(getattr(molecule, "multiplicity", 1))


def _trajectory_step_uq(molecule) -> float | None:
    value = getattr(molecule, "uq", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _trajectory_step_energy(molecule) -> float | None:
    value = getattr(molecule, "energy", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _trajectory_step_coordinates(molecule) -> np.ndarray:
    return np.asarray(molecule.xyz_coordinates, dtype=float)


def _molecule_to_manifest_entry(
    *,
    sample_id: str,
    geometry_path: Path,
    molecule,
    project_root: Path,
    source: str,
    source_kind: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "geometry_file": str(geometry_path.resolve().relative_to(project_root.resolve())),
        "charge": _molecule_charge(molecule),
        "multiplicity": _molecule_multiplicity(molecule),
        "num_atoms": len(_molecule_symbols(molecule)),
        "source": source,
        "source_kind": source_kind,
        "metadata": metadata,
    }


def _save_initial_condition_molecules(
    *,
    molecular_database,
    round_index: int,
    output_dir: str | Path,
    project_root: Path,
    manifest_path: str | Path,
    source_kind: str,
    source_label: str,
    temperature: float | None,
    sampling_method: str,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    entries: list[dict[str, Any]] = []

    for index, molecule in enumerate(molecular_database):
        sample_id = f"r{round_index:03d}_i{index:04d}"
        molecule.id = sample_id
        geometry_path = output_dir / f"{sample_id}.json"
        molecule.dump(filename=str(geometry_path), format="json")
        entries.append(
            _molecule_to_manifest_entry(
                sample_id=sample_id,
                geometry_path=geometry_path,
                molecule=molecule,
                project_root=project_root,
                source=source_label,
                source_kind=source_kind,
                metadata={
                    "round_index": round_index,
                    "initcond_id": sample_id,
                    "sampling_method": sampling_method,
                    "initial_temperature": temperature,
                },
            )
        )

    write_manifest(entries, manifest_path)
    return {
        "round_index": round_index,
        "manifest_file": str(Path(manifest_path).resolve()),
        "num_initial_conditions": len(entries),
        "output_dir": str(output_dir.resolve()),
        "sample_ids": [entry["sample_id"] for entry in entries],
    }


def generate_h2_initial_conditions(
    *,
    config: dict[str, Any],
    round_index: int,
    number_of_initial_conditions: int,
    output_dir: str | Path,
    manifest_path: str | Path,
    seed_json_path: str | Path | None = None,
) -> dict[str, Any]:
    from .mlatom_bridge import import_mlatom

    ml = import_mlatom()
    seed_path = Path(seed_json_path or config["paths"]["h2_seed_json"]).resolve()
    seed_molecule = ml.data.molecule()
    seed_molecule.load(str(seed_path), format="json")

    sampling_cfg = config.get("sampling", {})
    generate_initial_conditions = _resolve_generate_initial_conditions(ml)
    initial_temperature = sampling_cfg.get("initial_temperature")
    initial_molecular_database = generate_initial_conditions(
        molecule=seed_molecule,
        generation_method=str(sampling_cfg.get("initial_condition_sampler", "harmonic-quantum-boltzmann")),
        number_of_initial_conditions=int(number_of_initial_conditions),
        initial_temperature=None if initial_temperature is None else float(initial_temperature),
        use_hessian=bool(sampling_cfg.get("use_hessian", False)),
        reaction_coordinate_momentum=bool(sampling_cfg.get("reaction_coordinate_momentum", False)),
        random_seed=int(sampling_cfg.get("random_seed", 20260416)) + int(round_index),
    )

    return _save_initial_condition_molecules(
        molecular_database=initial_molecular_database,
        round_index=round_index,
        output_dir=output_dir,
        project_root=Path(config["project_root"]).resolve(),
        manifest_path=manifest_path,
        source_kind="h2_initial_condition",
        source_label="harmonic_quantum_boltzmann_from_h2_seed",
        temperature=None if initial_temperature is None else float(initial_temperature),
        sampling_method=str(sampling_cfg.get("initial_condition_sampler", "harmonic-quantum-boltzmann")),
    )


def _load_initial_conditions_for_md(
    *,
    config: dict[str, Any],
    round_index: int,
    number_of_initial_conditions: int,
):
    from .mlatom_bridge import import_mlatom

    ml = import_mlatom()
    seed_molecule = ml.data.molecule()
    seed_molecule.load(str(Path(config["paths"]["h2_seed_json"]).resolve()), format="json")

    sampling_cfg = config.get("sampling", {})
    generate_initial_conditions = _resolve_generate_initial_conditions(ml)
    initial_temperature = sampling_cfg.get("initial_temperature")
    initial_molecular_database = generate_initial_conditions(
        molecule=seed_molecule,
        generation_method=str(sampling_cfg.get("initial_condition_sampler", "harmonic-quantum-boltzmann")),
        number_of_initial_conditions=int(number_of_initial_conditions),
        initial_temperature=None if initial_temperature is None else float(initial_temperature),
        use_hessian=bool(sampling_cfg.get("use_hessian", False)),
        reaction_coordinate_momentum=bool(sampling_cfg.get("reaction_coordinate_momentum", False)),
        random_seed=int(sampling_cfg.get("random_seed", 20260416)) + int(round_index),
    )

    reversed_initial_conditions = initial_molecular_database.copy()
    for molecule in reversed_initial_conditions:
        for atom in molecule:
            atom.xyz_velocities = -atom.xyz_velocities

    return ml, initial_molecular_database, reversed_initial_conditions


def _trajectory_prefix(output_dir: Path, trajectory_id: str) -> Path:
    return output_dir / trajectory_id


def _select_dumped_trajectory_steps(traj, dump_interval: int) -> list[Any]:
    selected_steps: list[Any] = []
    for step_position, step in enumerate(traj.steps):
        if step_position == 0:
            selected_steps.append(step)
            continue
        step_index = int(getattr(step, "step", len(selected_steps)))
        if step_index % dump_interval == 0:
            selected_steps.append(step)

    if traj.steps:
        last_step = traj.steps[-1]
        if not selected_steps or selected_steps[-1] is not last_step:
            selected_steps.append(last_step)
    return selected_steps


def _dump_trajectory_files(traj, trajectory_prefix: Path, dump_interval: int, ml) -> tuple[str, str, int]:
    trajectory_prefix.parent.mkdir(parents=True, exist_ok=True)
    dumped_steps = _select_dumped_trajectory_steps(traj, dump_interval)
    dumped_trajectory = ml.data.molecular_trajectory()
    dumped_trajectory.steps = list(dumped_steps)
    dumped_trajectory.dump(filename=str(trajectory_prefix), format="plain_text")
    xyz_file = trajectory_prefix.with_suffix(".xyz")
    vxyz_file = trajectory_prefix.with_suffix(".vxyz")
    return str(xyz_file.resolve()), str(vxyz_file.resolve()), len(dumped_steps)


def _build_uncertainty_stop_function(threshold: float | None):
    def _stop_function(*, mol, stop_state=None):
        uncertain = bool(getattr(mol, "uncertain", False))
        if not uncertain and threshold is not None:
            uq_value = _trajectory_step_uq(mol)
            uncertain = uq_value is not None and uq_value > float(threshold)
            if uncertain:
                try:
                    mol.uncertain = True
                except Exception:
                    pass
        return uncertain, stop_state

    return _stop_function


def run_md_sampling_round(
    *,
    config: dict[str, Any],
    round_index: int,
    number_of_initial_conditions: int,
    maximum_propagation_time: float | None = None,
    time_step: float | None = None,
    save_interval_steps: int | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    ml, initial_conditions, reversed_initial_conditions = _load_initial_conditions_for_md(
        config=config,
        round_index=round_index,
        number_of_initial_conditions=number_of_initial_conditions,
    )
    if device_override:
        config["training"]["device"] = device_override

    state = load_training_state(config)
    model = create_direct_model_bundle(config, config["paths"]["models_dir"])
    model.load_trained_models(config["paths"]["models_dir"], load_main=True, load_aux=True)
    model.uq_threshold = state.get("uq_threshold")
    if model.uq_threshold is None:
        raise ValueError("MD 阶段需要先得到 uq_threshold，才能在首个不确定点停止轨迹。")

    sampling_cfg = config.get("sampling", {})
    md_cfg = sampling_cfg.get("md", {})
    md_time_step = float(time_step if time_step is not None else md_cfg.get("time_step", 0.5))
    md_max_time = float(maximum_propagation_time if maximum_propagation_time is not None else md_cfg.get("maximum_propagation_time", 150.0))
    dump_interval = max(1, int(save_interval_steps if save_interval_steps is not None else md_cfg.get("save_interval_steps", 5)))
    ensemble = str(md_cfg.get("ensemble", "NVE"))
    trajectory_mode = str(md_cfg.get("trajectory_mode", "bidirectional")).strip().lower()
    if trajectory_mode not in {"bidirectional", "forward"}:
        raise ValueError(f"不支持的 trajectory_mode：{trajectory_mode}")

    stop_function = _build_uncertainty_stop_function(float(model.uq_threshold))
    results_dir = Path(config["paths"]["results_dir"]).resolve()
    trajectories_dir = ensure_dir(results_dir / "trajectories" / f"round_{round_index:03d}")
    frame_manifest_path = results_dir / f"round_{round_index:03d}_md_frame_manifest.json"
    trajectory_summary_path = results_dir / f"round_{round_index:03d}_md_trajectory_summary.json"

    trajectory_summaries: list[dict[str, Any]] = []
    frame_payloads: list[dict[str, Any]] = []
    direction_datasets = [("forward", initial_conditions)]
    if trajectory_mode == "bidirectional":
        direction_datasets.append(("backward", reversed_initial_conditions))

    for direction, molecular_database in direction_datasets:
        for init_index, initial_molecule in enumerate(molecular_database):
            initcond_id = f"r{round_index:03d}_i{init_index:04d}"
            trajectory_id = f"r{round_index:03d}_t{init_index:04d}_{direction}"
            trajectory_prefix = _trajectory_prefix(trajectories_dir, trajectory_id)

            dyn = ml.md(
                model=model,
                molecule_with_initial_conditions=initial_molecule,
                ensemble=ensemble,
                time_step=md_time_step,
                maximum_propagation_time=md_max_time,
                stop_function=stop_function,
            )
            traj = dyn.molecular_trajectory
            trajectory_xyz_file, trajectory_vxyz_file, dumped_num_steps = _dump_trajectory_files(traj, trajectory_prefix, dump_interval, ml)
            last_step = traj.steps[-1]
            last_molecule = last_step.molecule
            last_uncertainty = _trajectory_step_uq(last_molecule)
            stopped_by_uncertainty = bool(getattr(last_molecule, "uncertain", False))
            if not stopped_by_uncertainty and last_uncertainty is not None:
                stopped_by_uncertainty = last_uncertainty > float(model.uq_threshold)

            trajectory_summary = {
                "trajectory_id": trajectory_id,
                "round_index": round_index,
                "initcond_id": initcond_id,
                "direction": direction,
                "num_steps": len(traj.steps),
                "trajectory_xyz_file": trajectory_xyz_file,
                "trajectory_vxyz_file": trajectory_vxyz_file,
                "time_step_fs": md_time_step,
                "maximum_propagation_time_fs": md_max_time,
                "save_interval_steps": dump_interval,
                "dumped_num_steps": dumped_num_steps,
                "stopped_by_uncertainty": stopped_by_uncertainty,
                "last_time_fs": float(last_step.time),
                "last_frame_index": int(last_step.step),
                "last_uncertainty": last_uncertainty,
                "generated_at": timestamp_string(),
            }
            trajectory_summary_file = trajectory_prefix.with_suffix(".json")
            write_json(trajectory_summary_file, trajectory_summary)
            trajectory_summary["trajectory_summary_file"] = str(trajectory_summary_file.resolve())
            trajectory_summaries.append(trajectory_summary)

            if stopped_by_uncertainty:
                frame_record = FrameRecord(
                    sample_id=f"{trajectory_id}_stop",
                    trajectory_id=trajectory_id,
                    initcond_id=initcond_id,
                    direction=direction,
                    round_index=round_index,
                    frame_index=int(last_step.step),
                    time_fs=float(last_step.time),
                    uq=last_uncertainty,
                    exceeds_threshold=True,
                    predicted_total_energy=_trajectory_step_energy(last_molecule),
                    symbols=_molecule_symbols(last_molecule),
                    atomic_numbers=_molecule_atomic_numbers(last_molecule),
                    coordinates=_trajectory_step_coordinates(last_molecule),
                    charge=_molecule_charge(last_molecule),
                    multiplicity=_molecule_multiplicity(last_molecule),
                    trajectory_xyz_file=trajectory_xyz_file,
                    trajectory_summary_file=str(trajectory_summary_file.resolve()),
                )
                frame_payloads.append(frame_record.to_payload())

    payload = {
        "round_index": round_index,
        "uq_threshold": model.uq_threshold,
        "num_trajectories": len(trajectory_summaries),
        "num_frames": len(frame_payloads),
        "num_candidate_samples": len(frame_payloads),
        "num_samples": len(frame_payloads),
        "num_uncertain_samples": len(frame_payloads),
        "num_uncertain_trajectories": len(frame_payloads),
        "time_step_fs": md_time_step,
        "maximum_propagation_time_fs": md_max_time,
        "save_interval_steps": dump_interval,
        "ensemble": ensemble,
        "trajectory_mode": trajectory_mode,
        "sampling_logic": "stop_on_first_uncertain_point",
        "frame_manifest_file": str(frame_manifest_path.resolve()),
        "trajectory_summary_file": str(trajectory_summary_path.resolve()),
        "trajectories": trajectory_summaries,
        "frames": frame_payloads,
        "samples": frame_payloads,
        "generated_at": timestamp_string(),
    }
    write_json(frame_manifest_path, payload)
    write_json(trajectory_summary_path, {"round_index": round_index, "trajectories": trajectory_summaries})
    return payload


def _kabsch_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    ref_centered = ref - ref.mean(axis=0)
    cand_centered = cand - cand.mean(axis=0)
    covariance = cand_centered.T @ ref_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    aligned = cand_centered @ rotation
    return float(np.sqrt(np.mean(np.square(aligned - ref_centered))))


def _load_geometry_coordinates(geometry_path: Path) -> np.ndarray:
    if geometry_path.suffix.lower() == ".json":
        payload = read_json(geometry_path)
        return np.asarray([atom["xyz_coordinates"] for atom in payload.get("atoms", [])], dtype=float)

    raw_lines = [line.strip() for line in geometry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    natoms = int(raw_lines[0])
    coordinates = []
    for line in raw_lines[2 : 2 + natoms]:
        fields = line.split()
        coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return np.asarray(coordinates, dtype=float)


def _load_cumulative_coordinates(config: dict[str, Any]) -> list[np.ndarray]:
    cumulative_manifest_path = Path(config["paths"]["cumulative_labeled_manifest"]).resolve()
    if not cumulative_manifest_path.exists():
        return []

    coordinates: list[np.ndarray] = []
    project_root = Path(config["project_root"]).resolve()
    for entry in load_manifest(cumulative_manifest_path):
        geometry_path = Path(entry["geometry_file"])
        if not geometry_path.is_absolute():
            geometry_path = (project_root / geometry_path).resolve()
        coordinates.append(_load_geometry_coordinates(geometry_path))
    return coordinates


def select_md_frames(
    *,
    config: dict[str, Any],
    round_index: int,
    frame_manifest_path: str | Path | None = None,
    max_new_points: int | None = None,
    dedup_rmsd_threshold: float | None = None,
) -> dict[str, Any]:
    results_dir = Path(config["paths"]["results_dir"]).resolve()
    project_root = Path(config["project_root"]).resolve()
    frame_manifest_path = Path(frame_manifest_path or results_dir / f"round_{round_index:03d}_md_frame_manifest.json").resolve()
    payload = read_json(frame_manifest_path)
    frames = payload.get("samples") or payload.get("frames", [])
    threshold = payload.get("uq_threshold")

    selection_limit_raw = max_new_points if max_new_points is not None else config["active_learning"].get("max_new_points_per_round")
    selection_limit = None
    if selection_limit_raw is not None:
        selection_limit_value = int(selection_limit_raw)
        if selection_limit_value > 0:
            selection_limit = selection_limit_value

    rmsd_threshold = float(
        dedup_rmsd_threshold if dedup_rmsd_threshold is not None else config.get("sampling", {}).get("dedup_rmsd_threshold", 0.05)
    )
    total_trajectories = int(payload.get("num_trajectories", len(frames)))

    sorted_frames = sorted(frames, key=lambda item: float(item.get("uncertainty", -1.0) or -1.0), reverse=True)
    if threshold is None:
        uncertain_frames = sorted_frames
    else:
        uncertain_frames = [item for item in sorted_frames if item.get("uncertainty") is not None and float(item["uncertainty"]) > float(threshold)]

    history_coordinates = _load_cumulative_coordinates(config)
    selected_coordinates: list[np.ndarray] = []
    selected_frames: list[dict[str, Any]] = []
    rejected_due_to_history = 0
    rejected_due_to_current_round = 0

    selected_output_dir = ensure_dir(project_root / "data" / "raw" / f"round_{round_index:03d}_selected")
    manifest_entries: list[dict[str, Any]] = []

    for frame in uncertain_frames:
        if selection_limit is not None and len(selected_frames) >= selection_limit:
            break

        coordinates = np.asarray(frame["coordinates"], dtype=float)
        if any(_kabsch_rmsd(history_coords, coordinates) < rmsd_threshold for history_coords in history_coordinates):
            rejected_due_to_history += 1
            continue
        if any(_kabsch_rmsd(existing_coords, coordinates) < rmsd_threshold for existing_coords in selected_coordinates):
            rejected_due_to_current_round += 1
            continue

        sample_id = str(frame["sample_id"])
        geometry_record = GeometryRecord(
            sample_id=sample_id,
            symbols=[str(symbol) for symbol in frame["symbols"]],
            coordinates=coordinates,
            charge=int(frame.get("charge", 0)),
            multiplicity=int(frame.get("multiplicity", 1)),
            source="selected_from_md_uq",
            source_kind="md_frame",
            metadata={
                "round_index": round_index,
                "parent_trajectory_id": frame["trajectory_id"],
                "frame_index": int(frame["frame_index"]),
                "time_fs": float(frame["time_fs"]),
                "initcond_id": frame["initcond_id"],
                "uq_at_selection": frame.get("uncertainty"),
                "direction": frame.get("direction"),
                "trajectory_xyz_file": frame.get("trajectory_xyz_file"),
            },
        )
        geometry_path = selected_output_dir / f"{sample_id}.xyz"
        save_xyz(geometry_record, geometry_path, comment="selected from md trajectory")
        manifest_entries.append(geometry_record.to_manifest_entry(geometry_path, project_root))
        selected_frames.append(frame)
        selected_coordinates.append(coordinates)

    selection_manifest_path = results_dir / f"round_{round_index:03d}_selected_manifest.json"
    selection_summary_path = results_dir / f"round_{round_index:03d}_selection_summary.json"
    write_manifest(manifest_entries, selection_manifest_path)

    total = max(total_trajectories, 1)
    uncertain_ratio = len(uncertain_frames) / total
    summary = {
        "round_index": round_index,
        "uq_threshold": threshold,
        "num_trajectories": total_trajectories,
        "num_frame_samples": len(sorted_frames),
        "num_candidate_samples": len(sorted_frames),
        "num_pool_samples": len(sorted_frames),
        "num_uncertain_frames": len(uncertain_frames),
        "num_uncertain_samples": len(uncertain_frames),
        "num_uncertain_trajectories": len(uncertain_frames),
        "selection_limit": selection_limit,
        "selected_count": len(selected_frames),
        "selected_sample_ids": [entry["sample_id"] for entry in manifest_entries],
        "uncertain_ratio": uncertain_ratio,
        "converged": uncertain_ratio < float(config["active_learning"].get("convergence_ratio", 0.05))
        or len(selected_frames) < int(config["active_learning"].get("min_new_points", 5)),
        "rejected_due_to_history_rmsd": rejected_due_to_history,
        "rejected_due_to_current_round_rmsd": rejected_due_to_current_round,
        "selection_basis": "first_uncertain_md_point",
        "sampling_logic": payload.get("sampling_logic", "stop_on_first_uncertain_point"),
        "frame_manifest_file": str(frame_manifest_path.resolve()),
        "selection_manifest_file": str(selection_manifest_path.resolve()),
        "generated_at": timestamp_string(),
    }
    write_json(selection_summary_path, summary)

    history_path = Path(config["paths"].get("active_learning_round_history_file", results_dir / "active_learning_round_history.json")).resolve()
    write_json(history_path, rebuild_round_history(results_dir))
    return summary
