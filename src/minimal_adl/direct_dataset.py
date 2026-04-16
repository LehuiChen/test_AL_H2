from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import load_geometry, load_manifest
from .io_utils import read_json, write_json


def load_label_result(label_json_path: str | Path) -> dict[str, Any]:
    payload = read_json(label_json_path)
    if not payload.get("success", False):
        raise RuntimeError(f"标注任务失败：{label_json_path}")
    return payload


def build_direct_dataset(
    *,
    manifest_path: str | Path,
    gaussian_labels_dir: str | Path,
    npz_output_path: str | Path,
    metadata_output_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """把累计 Gaussian 标注结果组装成直接学习数据集。"""

    manifest_entries = load_manifest(manifest_path)
    gaussian_labels_dir = Path(gaussian_labels_dir).resolve()
    project_root_path = Path(project_root).resolve() if project_root is not None else Path(manifest_path).resolve().parents[2]

    sample_ids: list[str] = []
    atomic_numbers: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    target_energies: list[float] = []
    target_forces: list[np.ndarray] = []
    per_sample_metadata: list[dict[str, Any]] = []

    for entry in manifest_entries:
        sample_id = str(entry["sample_id"])
        geometry_file = Path(entry["geometry_file"])
        if not geometry_file.is_absolute():
            geometry_file = (project_root_path / geometry_file).resolve()
        else:
            geometry_file = geometry_file.resolve()

        target_result = load_label_result(gaussian_labels_dir / sample_id / "label.json")
        geometry = load_geometry(geometry_file)

        sample_ids.append(sample_id)
        atomic_numbers.append(geometry.atomic_numbers)
        coordinates.append(np.asarray(geometry.coordinates, dtype=float))
        target_energies.append(float(target_result["energy"]))
        target_forces.append(np.asarray(target_result["forces"], dtype=float))
        per_sample_metadata.append(
            {
                "sample_id": sample_id,
                "geometry_file": str(geometry_file),
                "charge": geometry.charge,
                "multiplicity": geometry.multiplicity,
                "source": entry.get("source", ""),
                "source_kind": entry.get("source_kind", ""),
                "manifest_metadata": entry.get("metadata", {}),
                "round_index": entry.get("metadata", {}).get("round_index"),
                "parent_trajectory_id": entry.get("metadata", {}).get("parent_trajectory_id"),
                "frame_index": entry.get("metadata", {}).get("frame_index"),
                "time_fs": entry.get("metadata", {}).get("time_fs"),
                "initcond_id": entry.get("metadata", {}).get("initcond_id"),
                "uq_at_selection": entry.get("metadata", {}).get("uq_at_selection"),
                "target_label_file": str((gaussian_labels_dir / sample_id / "label.json").resolve()),
            }
        )

    np.savez_compressed(
        npz_output_path,
        sample_ids=np.asarray(sample_ids),
        atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        coordinates=np.asarray(coordinates, dtype=float),
        E_target=np.asarray(target_energies, dtype=float),
        F_target=np.asarray(target_forces, dtype=float),
    )

    metadata = {
        "num_samples": len(sample_ids),
        "samples": per_sample_metadata,
        "notes": {
            "energy_definition": "Gaussian target energy",
            "force_definition": "Gaussian target force",
        },
    }
    write_json(metadata_output_path, metadata)
    return metadata
