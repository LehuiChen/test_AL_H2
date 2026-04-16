from __future__ import annotations

import importlib
import inspect
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import load_manifest
from .io_utils import read_json, write_json


def _ensure_torch_load_compat() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return

    try:
        load_signature = inspect.signature(torch.load)
    except (TypeError, ValueError):
        return

    if getattr(torch.load, "_minimal_adl_weights_only_compat", False):
        return

    original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        if "weights_only" in load_signature.parameters:
            kwargs.setdefault("weights_only", False)
        else:
            kwargs.pop("weights_only", None)
        return original_torch_load(*args, **kwargs)

    _compat_torch_load._minimal_adl_weights_only_compat = True  # type: ignore[attr-defined]
    torch.load = _compat_torch_load


def import_mlatom():
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _ensure_torch_load_compat()
    try:
        import mlatom as ml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import `mlatom`. Activate the correct ADL environment and verify that import works in that shell."
        ) from exc
    return ml


def detect_geometry_format(geometry_path: str | Path) -> str:
    suffix = Path(geometry_path).suffix.lower()
    if suffix == ".xyz":
        return "xyz"
    if suffix == ".json":
        return "json"
    raise ValueError(f"Unsupported geometry file format: {geometry_path}")


def create_mlatom_method(method_config: dict[str, Any]):
    return _create_mlatom_method(method_config)


def _normalize_gaussian_method_name(method_name: str) -> str:
    route = method_name.strip()
    if not route:
        return route
    if "/" not in route:
        functional = route
        basis = ""
    else:
        functional, basis = route.split("/", 1)
    normalized_functional = functional.strip()
    functional_key = normalized_functional.lower().replace("\u03c9", "w")
    gaussian_aliases = {"wb97x-d": "wB97XD", "wb97xd": "wB97XD"}
    normalized_functional = gaussian_aliases.get(functional_key, normalized_functional)
    if not basis:
        return normalized_functional
    return f"{normalized_functional}/{basis.strip()}"


def _create_mlatom_method(
    method_config: dict[str, Any],
    *,
    working_directory: str | Path | None = None,
):
    ml = import_mlatom()
    method_name = str(method_config["method"]).strip()
    program_name = method_config.get("program")
    kwargs = {
        "method": method_name,
        "nthreads": int(method_config.get("nthreads", 1)),
        "save_files_in_current_directory": bool(method_config.get("save_files_in_current_directory", False)),
    }
    if program_name:
        kwargs["program"] = program_name

    if str(program_name).lower() == "gaussian":
        normalized_method_name = _normalize_gaussian_method_name(method_name)
        try:
            gaussian_module = importlib.import_module("mlatom.interfaces.gaussian_interface")
            gaussian_methods = getattr(gaussian_module, "gaussian_methods")
            gaussian_kwargs = {
                "method": normalized_method_name,
                "nthreads": int(method_config.get("nthreads", 1)),
                "save_files_in_current_directory": bool(method_config.get("save_files_in_current_directory", False)),
            }
            if working_directory is not None and "working_directory" in inspect.signature(gaussian_methods).parameters:
                gaussian_kwargs["working_directory"] = str(Path(working_directory).resolve())
            return gaussian_methods(**gaussian_kwargs)
        except Exception:
            pass
        kwargs["method"] = normalized_method_name

    return ml.models.methods(**kwargs)


def _read_text_tail(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"<Could not read {path.name}: {type(exc).__name__}: {exc}>"
    return "\n".join(lines[-max_lines:])


def _extract_energy_and_gradients(
    molecule: Any,
    *,
    method_config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> tuple[float, np.ndarray]:
    energy_value = None
    for attr_name in ("energy", "scf_energy"):
        if hasattr(molecule, attr_name):
            try:
                energy_value = float(getattr(molecule, attr_name))
                break
            except Exception:
                pass

    gradients = None
    if hasattr(molecule, "get_energy_gradients"):
        try:
            gradients = np.asarray(molecule.get_energy_gradients(), dtype=float)
        except Exception:
            gradients = None
    if gradients is None and hasattr(molecule, "energy_gradients"):
        try:
            gradients = np.asarray(getattr(molecule, "energy_gradients"), dtype=float)
        except Exception:
            gradients = None
    if gradients is None and hasattr(molecule, "get_xyz_vectorial_properties"):
        try:
            gradients = np.asarray(molecule.get_xyz_vectorial_properties("energy_gradients"), dtype=float)
        except Exception:
            gradients = None

    if energy_value is not None and gradients is not None:
        return energy_value, gradients

    debug_lines = [
        "MLatom prediction finished but the molecule object did not expose both energy and energy_gradients.",
        f"method = {method_config.get('method')}",
        f"program = {method_config.get('program')}",
        f"molecule_attrs = {sorted(molecule.__dict__.keys())}",
    ]
    if hasattr(molecule, "error_message"):
        debug_lines.append(f"molecule.error_message = {getattr(molecule, 'error_message')}")
    if output_dir is not None:
        output_path = Path(output_dir)
        for suffix in ("*.log", "*.out", "*.com"):
            for file_path in sorted(output_path.glob(suffix)):
                debug_lines.append(f"===== {file_path.name} =====")
                debug_lines.append(_read_text_tail(file_path))
    raise RuntimeError("\n".join(debug_lines))


def label_geometry_with_mlatom(
    *,
    geometry_path: str | Path,
    method_config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ml = import_mlatom()
    molecule = ml.data.molecule()
    molecule.load(str(geometry_path), format=detect_geometry_format(geometry_path))

    method = _create_mlatom_method(method_config, working_directory=output_dir)
    method.predict(
        molecule=molecule,
        calculate_energy=True,
        calculate_energy_gradients=True,
        calculate_hessian=False,
    )

    energy, gradients = _extract_energy_and_gradients(
        molecule,
        method_config=method_config,
        output_dir=output_dir,
    )
    forces = -gradients
    return {
        "success": True,
        "method": method_config["method"],
        "program": method_config.get("program"),
        "geometry_file": str(Path(geometry_path).resolve()),
        "energy": energy,
        "energy_gradients": gradients.tolist(),
        "forces": forces.tolist(),
    }


def run_and_save_label_job(
    *,
    geometry_path: str | Path,
    method_config: dict[str, Any],
    output_dir: str | Path,
    method_key: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    label_path = output_dir / "label.json"

    try:
        payload = label_geometry_with_mlatom(
            geometry_path=geometry_path,
            method_config=method_config,
            output_dir=output_dir,
        )
        payload["method_key"] = method_key
        write_json(label_path, payload)
        write_json(
            status_path,
            {
                "success": True,
                "method_key": method_key,
                "geometry_file": str(Path(geometry_path).resolve()),
                "label_file": str(label_path.resolve()),
            },
        )
        return payload
    except Exception as exc:
        error_payload = {
            "success": False,
            "method_key": method_key,
            "geometry_file": str(Path(geometry_path).resolve()),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(status_path, error_payload)
        write_json(label_path, error_payload)
        raise


def build_molecular_database_from_geometry_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
):
    ml = import_mlatom()
    entries = load_manifest(manifest_path)
    molecular_database = ml.data.molecular_database()
    project_root_path = Path(project_root).resolve() if project_root is not None else None

    for entry in entries:
        geometry_file = Path(entry["geometry_file"])
        if not geometry_file.is_absolute():
            if project_root_path is None:
                raise ValueError("Relative geometry paths require project_root for reliable resolution.")
            geometry_file = project_root_path / geometry_file
        molecule = ml.data.molecule()
        molecule.load(str(geometry_file), format=detect_geometry_format(geometry_file))
        molecule.id = entry["sample_id"]
        molecular_database += molecule

    return molecular_database


def build_molecular_database_from_direct_dataset(
    *,
    npz_path: str | Path,
    metadata_path: str | Path,
):
    ml = import_mlatom()
    dataset = np.load(npz_path, allow_pickle=True)
    metadata = read_json(metadata_path)["samples"]

    molecular_database = ml.data.molecular_database()
    geometry_files = [entry["geometry_file"] for entry in metadata]

    for index, geometry_file in enumerate(geometry_files):
        molecule = ml.data.molecule()
        molecule.load(str(geometry_file), format=detect_geometry_format(geometry_file))
        sample_id = str(dataset["sample_ids"][index])
        molecule.id = sample_id
        molecule.reference_energy = float(dataset["E_target"][index])
        molecule.energy = float(dataset["E_target"][index])
        reference_force = np.asarray(dataset["F_target"][index], dtype=float)
        molecule.add_xyz_vectorial_property(-reference_force, "reference_energy_gradients")
        molecule.add_xyz_vectorial_property(-reference_force, "energy_gradients")
        molecular_database += molecule

    return molecular_database
