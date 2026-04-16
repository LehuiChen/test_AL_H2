from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .geometry import load_geometry
from .io_utils import ensure_dir, read_json, write_json


def prepare_h2_seed(
    *,
    xyz_source_path: str | Path,
    frequency_json_source_path: str | Path,
    xyz_output_path: str | Path,
    json_output_path: str | Path,
    summary_output_path: str | Path,
) -> dict[str, Any]:
    """把 H2 平衡构型和频率文件标准化成项目内 seed 产物。"""

    xyz_source_path = Path(xyz_source_path).resolve()
    frequency_json_source_path = Path(frequency_json_source_path).resolve()
    xyz_output_path = Path(xyz_output_path).resolve()
    json_output_path = Path(json_output_path).resolve()
    summary_output_path = Path(summary_output_path).resolve()

    if not xyz_source_path.exists():
        raise FileNotFoundError(f"找不到 H2 xyz 文件：{xyz_source_path}")
    if not frequency_json_source_path.exists():
        raise FileNotFoundError(f"找不到 H2 频率 JSON 文件：{frequency_json_source_path}")

    ensure_dir(xyz_output_path.parent)
    ensure_dir(json_output_path.parent)
    ensure_dir(summary_output_path.parent)

    shutil.copyfile(xyz_source_path, xyz_output_path)
    shutil.copyfile(frequency_json_source_path, json_output_path)

    geometry = load_geometry(xyz_output_path)
    freq_payload = read_json(json_output_path)
    atoms = freq_payload.get("atoms", [])
    has_hessian = "hessian" in freq_payload
    normal_modes_per_atom = 0
    if atoms and isinstance(atoms[0], dict):
        normal_modes = atoms[0].get("normal_modes", [])
        if isinstance(normal_modes, list):
            normal_modes_per_atom = len(normal_modes)

    summary = {
        "xyz_source_file": str(xyz_source_path),
        "frequency_source_file": str(frequency_json_source_path),
        "h2_seed_xyz": str(xyz_output_path),
        "h2_seed_json": str(json_output_path),
        "num_atoms": len(geometry.symbols),
        "charge": geometry.charge,
        "multiplicity": geometry.multiplicity,
        "symbols": geometry.symbols,
        "has_hessian": has_hessian,
        "normal_modes_per_atom": normal_modes_per_atom,
    }
    write_json(summary_output_path, summary)
    return summary
