from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, read_json, write_json

SYMBOL_TO_ATOMIC_NUMBER = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}
ATOMIC_NUMBER_TO_SYMBOL = {value: key for key, value in SYMBOL_TO_ATOMIC_NUMBER.items()}


@dataclass
class GeometryRecord:
    sample_id: str
    symbols: list[str]
    coordinates: np.ndarray
    charge: int = 0
    multiplicity: int = 1
    source: str = ""
    source_kind: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.asarray([SYMBOL_TO_ATOMIC_NUMBER[symbol] for symbol in self.symbols], dtype=int)

    def to_manifest_entry(self, geometry_path: Path, project_root: Path) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "geometry_file": str(geometry_path.resolve().relative_to(project_root.resolve())),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "num_atoms": len(self.symbols),
            "source": self.source,
            "source_kind": self.source_kind,
            "metadata": self.metadata,
        }


def _load_xyz(path: Path) -> GeometryRecord:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"XYZ 文件内容过短：{path}")

    num_atoms = int(lines[0])
    comment = lines[1]
    body = lines[2 : 2 + num_atoms]

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in body:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"XYZ 行格式不正确：{line}")
        symbols.append(fields[0])
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])

    return GeometryRecord(
        sample_id=path.stem,
        symbols=symbols,
        coordinates=np.asarray(coords, dtype=float),
        source=comment,
    )


def _load_json(path: Path) -> GeometryRecord:
    payload = read_json(path)
    if "atoms" not in payload:
        raise ValueError(f"JSON 几何文件缺少 atoms 字段：{path}")

    symbols: list[str] = []
    for atom in payload["atoms"]:
        if atom.get("element_symbol"):
            symbols.append(str(atom["element_symbol"]))
            continue
        atomic_number = atom.get("atomic_number", atom.get("nuclear_charge"))
        if atomic_number is None:
            raise KeyError("JSON atom 缺少 element_symbol 和 atomic_number/nuclear_charge。")
        atomic_number = int(atomic_number)
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number)
        if symbol is None:
            raise KeyError(f"不支持的原子序数：{atomic_number}")
        symbols.append(symbol)
    coords = [atom["xyz_coordinates"] for atom in payload["atoms"]]

    return GeometryRecord(
        sample_id=payload.get("id", path.stem),
        symbols=symbols,
        coordinates=np.asarray(coords, dtype=float),
        charge=int(payload.get("charge", 0)),
        multiplicity=int(payload.get("multiplicity", 1)),
        source=path.name,
        source_kind=str(payload.get("source_kind", payload.get("source_format", "mlatom_json"))),
        metadata={"source_format": "mlatom_json"},
    )


def load_geometry(path: str | Path) -> GeometryRecord:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        return _load_xyz(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"暂不支持的几何格式：{path}")


def save_xyz(record: GeometryRecord, path: str | Path, comment: str | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [str(len(record.symbols)), comment or record.source or record.sample_id]
    for symbol, coord in zip(record.symbols, record.coordinates):
        lines.append(f"{symbol} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(entries: list[dict[str, Any]], path: str | Path) -> None:
    write_json(path, {"samples": entries})


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return payload.get("samples", [])
