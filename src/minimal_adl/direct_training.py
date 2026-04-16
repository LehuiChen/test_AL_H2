from __future__ import annotations

from pathlib import Path

from .direct_model import DirectMLModel
from .io_utils import read_json
from .mlatom_bridge import build_molecular_database_from_direct_dataset


def create_direct_model_bundle(config: dict, workdir: str | Path) -> DirectMLModel:
    training_cfg = config.get("training", {})
    model = DirectMLModel(
        model_file=training_cfg.get("model_name", "direct_bundle"),
        ml_model_type=training_cfg.get("ml_model_type", "ANI"),
        validation_set_fraction=float(training_cfg.get("validation_set_fraction", 0.1)),
        device=training_cfg.get("device"),
        verbose=True,
        main_model_stem=training_cfg.get("main_model_stem", "direct_main_model"),
        aux_model_stem=training_cfg.get("aux_model_stem", "direct_aux_model"),
    )
    model.prepare_model_paths(workdir)
    return model


def train_direct_bundle(
    *,
    config: dict,
    train_main: bool,
    train_aux: bool,
) -> dict:
    paths_cfg = config["paths"]
    workdir = Path(paths_cfg["models_dir"])
    workdir.mkdir(parents=True, exist_ok=True)

    molecular_database = build_molecular_database_from_direct_dataset(
        npz_path=paths_cfg["direct_dataset_npz"],
        metadata_path=paths_cfg["direct_dataset_metadata"],
    )

    model = create_direct_model_bundle(config, workdir)
    state_file = workdir / config["training"].get("state_filename", "training_state.json")
    al_cfg = config.get("active_learning", {})
    uq_cfg = config.get("uncertainty", {})
    al_info = {
        "working_directory": str(workdir),
        "threshold_metric": uq_cfg.get("threshold_metric", al_cfg.get("threshold_metric", "m+3mad")),
    }
    if uq_cfg.get("uncertainty_threshold") is not None:
        al_info["uq_threshold"] = float(uq_cfg["uncertainty_threshold"])
    elif state_file.exists():
        try:
            previous_state = read_json(state_file)
        except Exception:
            previous_state = {}
        previous_threshold = previous_state.get("uq_threshold")
        if previous_threshold is not None:
            al_info["uq_threshold"] = float(previous_threshold)

    return model.train(
        molecular_database=molecular_database,
        al_info=al_info,
        train_main=train_main,
        train_aux=train_aux,
        summary_filename=config["training"].get("summary_filename", "training_summary.json"),
        state_filename=config["training"].get("state_filename", "training_state.json"),
    )


def load_training_state(config: dict) -> dict:
    state_path = Path(config["paths"]["models_dir"]) / config["training"].get("state_filename", "training_state.json")
    return read_json(state_path)
