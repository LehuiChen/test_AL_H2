from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import load_manifest
from .io_utils import read_json, timestamp_string


def normalize_selected_ids(payload: dict[str, Any]) -> list[str]:
    selected_ids = payload.get("selected_sample_ids")
    if isinstance(selected_ids, list) and selected_ids:
        return [str(item) for item in selected_ids]

    selected_samples = payload.get("selected_samples", [])
    if isinstance(selected_samples, list):
        if selected_samples and isinstance(selected_samples[0], dict):
            return [str(item.get("sample_id")) for item in selected_samples if item.get("sample_id") is not None]
        return [str(item) for item in selected_samples]
    return []


def rebuild_round_history(results_dir: str | Path) -> dict[str, Any]:
    results_dir = Path(results_dir).resolve()
    round_rows: list[dict[str, Any]] = []

    for summary_path in sorted(results_dir.glob("round_*_selection_summary.json")):
        payload = read_json(summary_path)
        round_index = int(payload.get("round_index", 0))
        selected_sample_ids = normalize_selected_ids(payload)
        selected_count = int(payload.get("selected_count") or payload.get("num_selected") or len(selected_sample_ids))

        manifest_path = results_dir / f"round_{round_index:03d}_selected_manifest.json"
        manifest_selected_count = None
        if manifest_path.exists():
            try:
                manifest_selected_count = len(load_manifest(manifest_path))
            except Exception:
                manifest_selected_count = None

        round_rows.append(
            {
                "round_index": round_index,
                "selected_count": selected_count,
                "selected_sample_ids": selected_sample_ids,
                "num_pool_samples": payload.get("num_pool_samples", payload.get("num_frame_samples")),
                "num_uncertain_samples": payload.get("num_uncertain_samples", payload.get("num_uncertain_frames")),
                "uncertain_ratio": payload.get("uncertain_ratio"),
                "converged": payload.get("converged"),
                "selection_basis": payload.get("selection_basis", "unknown"),
                "selection_summary_file": str(summary_path.resolve()),
                "selection_manifest_file": str(manifest_path.resolve()),
                "selected_manifest_exists": manifest_path.exists(),
                "manifest_selected_count": manifest_selected_count,
                "frame_manifest_file": payload.get("frame_manifest_file"),
                "updated_at": payload.get("generated_at") or payload.get("updated_at") or timestamp_string(),
            }
        )

    latest_round_index = max((row["round_index"] for row in round_rows), default=None)
    return {
        "total_rounds": len(round_rows),
        "latest_round_index": latest_round_index,
        "rounds": round_rows,
        "updated_at": timestamp_string(),
    }
