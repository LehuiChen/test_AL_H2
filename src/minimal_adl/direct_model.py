from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import write_csv_rows, write_json
from .mlatom_bridge import import_mlatom

ml = import_mlatom()


class DirectMLModel(ml.al_utils.ml_model):
    """H2 直接学习场景下的主模型/副模型封装。"""

    def __init__(
        self,
        *,
        al_info: dict[str, Any] | None = None,
        model_file: str | None = None,
        device: str | None = None,
        verbose: bool = False,
        ml_model_type: str = "ANI",
        validation_set_fraction: float = 0.1,
        main_model_stem: str = "direct_main_model",
        aux_model_stem: str = "direct_aux_model",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            al_info=al_info or {},
            model_file=model_file,
            device=device,
            verbose=verbose,
            ml_model_type=ml_model_type,
            **kwargs,
        )
        self.validation_set_fraction = validation_set_fraction
        self.main_model_stem = main_model_stem
        self.aux_model_stem = aux_model_stem
        self.uq_threshold: float | None = None
        self.main_model = getattr(self, "main_model", None)
        self.aux_model = getattr(self, "aux_model", None)

    def _model_extension(self) -> str:
        return ".pt"

    def prepare_model_paths(self, workdir: str | Path) -> tuple[str, str]:
        workdir = Path(workdir)
        extension = self._model_extension()
        self.main_model_file = str((workdir / f"{self.main_model_stem}{extension}").resolve())
        self.aux_model_file = str((workdir / f"{self.aux_model_stem}{extension}").resolve())
        return self.main_model_file, self.aux_model_file

    def load_trained_models(self, workdir: str | Path, *, load_main: bool = True, load_aux: bool = True) -> None:
        self.prepare_model_paths(workdir)

        if load_main:
            if not os.path.exists(self.main_model_file):
                raise FileNotFoundError(f"找不到主模型文件：{self.main_model_file}")
            self.main_model = self.initialize_model(
                ml_model_type=self.ml_model_type,
                model_file=self.main_model_file,
                device=self.device,
                verbose=self.verbose,
            )

        if load_aux:
            if not os.path.exists(self.aux_model_file):
                raise FileNotFoundError(f"找不到副模型文件：{self.aux_model_file}")
            self.aux_model = self.initialize_model(
                ml_model_type=self.ml_model_type,
                model_file=self.aux_model_file,
                device=self.device,
                verbose=self.verbose,
            )

    def train(
        self,
        *,
        molecular_database=None,
        al_info: dict[str, Any] | None = None,
        train_main: bool = True,
        train_aux: bool = True,
        summary_filename: str = "training_summary.json",
        state_filename: str = "training_state.json",
    ) -> dict[str, Any]:
        if molecular_database is None:
            raise ValueError("训练前必须提供 molecular_database。")

        al_info = dict(al_info or {})
        workdir = Path(al_info.get("working_directory", ".")).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        self.prepare_model_paths(workdir)

        [subtraindb, valdb] = molecular_database.split(
            number_of_splits=2,
            fraction_of_points_in_splits=[1 - self.validation_set_fraction, self.validation_set_fraction],
            sampling="random",
        )

        if train_main:
            self.main_model = self.initialize_model(
                ml_model_type=self.ml_model_type,
                model_file=self.main_model_file,
                device=self.device,
                verbose=self.verbose,
            )
            self.model_trainer(model=self.main_model, subtraindb=subtraindb, valdb=valdb, learning_grad=True)
        elif os.path.exists(self.main_model_file):
            self.main_model = self.initialize_model(
                ml_model_type=self.ml_model_type,
                model_file=self.main_model_file,
                device=self.device,
                verbose=self.verbose,
            )

        if train_aux:
            self.aux_model = self.initialize_model(
                ml_model_type=self.ml_model_type,
                model_file=self.aux_model_file,
                device=self.device,
                verbose=self.verbose,
            )
            self.model_trainer(model=self.aux_model, subtraindb=subtraindb, valdb=valdb, learning_grad=False)
        elif not train_main and os.path.exists(self.aux_model_file):
            self.aux_model = self.initialize_model(
                ml_model_type=self.ml_model_type,
                model_file=self.aux_model_file,
                device=self.device,
                verbose=self.verbose,
            )
        else:
            self.aux_model = None

        if "uq_threshold" in al_info:
            self.uq_threshold = float(al_info["uq_threshold"])
        elif self.main_model is not None and self.aux_model is not None:
            valdb_copy = valdb.copy()
            self.predict(molecular_database=valdb_copy)
            uq_values = [float(molecule.uq) for molecule in valdb_copy]
            metric_name = al_info.get("threshold_metric", "m+3mad")
            self.uq_threshold = float(self.threshold_metric(uq_values, metric=metric_name))
        else:
            self.uq_threshold = None

        summary, summary_details = self.summary(subtraindb=subtraindb, valdb=valdb, include_details=True)
        artifact_paths = self.write_training_artifacts(workdir=workdir, summary=summary, summary_details=summary_details)
        state = {
            "main_model_file": self.main_model_file if self.main_model is not None else None,
            "aux_model_file": self.aux_model_file if self.aux_model is not None else None,
            "uq_threshold": self.uq_threshold,
            "train_main": train_main,
            "train_aux": train_aux,
            **artifact_paths,
        }
        write_json(workdir / summary_filename, summary)
        write_json(workdir / state_filename, state)
        return state

    def predict(self, molecule=None, molecular_database=None, **kwargs: Any) -> None:  # noqa: ARG002
        if molecule is not None:
            molecular_database = ml.data.molecular_database(molecule)
        elif molecular_database is None:
            raise ValueError("predict 需要 molecule 或 molecular_database。")

        if self.main_model is None:
            raise RuntimeError("主模型尚未初始化，无法预测。")

        self.main_model.predict(
            molecular_database=molecular_database,
            property_to_predict="energy",
            xyz_derivative_property_to_predict="energy_gradients",
        )

        if self.aux_model is not None:
            self.aux_model.predict(
                molecular_database=molecular_database,
                property_to_predict="aux_energy",
            )
            for molecule_item in molecular_database:
                molecule_item.uq = abs(float(molecule_item.energy) - float(molecule_item.aux_energy))
                molecule_item.uncertain = self.uq_threshold is not None and molecule_item.uq > self.uq_threshold

    def model_trainer(self, *, model, subtraindb, valdb, learning_grad: bool) -> None:
        if self.ml_model_type.casefold() != "ani":
            raise ValueError(f"当前 H2 工作流只支持 ANI，收到：{self.ml_model_type}")

        subtraindb_copy = subtraindb.copy()
        valdb_copy = valdb.copy()
        if learning_grad:
            model.train(
                molecular_database=subtraindb_copy,
                validation_molecular_database=valdb_copy,
                property_to_learn="energy",
                xyz_derivative_property_to_learn="energy_gradients",
            )
        else:
            model.train(
                molecular_database=subtraindb_copy,
                validation_molecular_database=valdb_copy,
                property_to_learn="energy",
            )

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_vector_property(molecule: Any, property_name: str) -> np.ndarray | None:
        try:
            return np.asarray(molecule.get_xyz_vectorial_properties(property_name), dtype=float)
        except Exception:
            pass

        raw_value = getattr(molecule, property_name, None)
        if raw_value is None:
            return None
        try:
            return np.asarray(raw_value, dtype=float)
        except Exception:
            return None

    def _normalize_history_payload(self, payload: Any) -> dict[str, list[float]]:
        if isinstance(payload, dict):
            normalized: dict[str, list[float]] = {}
            for key, value in payload.items():
                if isinstance(value, dict) and "history" in value:
                    nested = self._normalize_history_payload(value["history"])
                    if nested:
                        return nested
                    continue
                try:
                    series = [float(item) for item in np.ravel(value).tolist()]
                except Exception:
                    continue
                if series:
                    normalized[str(key)] = series
            return normalized

        if hasattr(payload, "history") and isinstance(payload.history, dict):
            return self._normalize_history_payload(payload.history)

        if isinstance(payload, (list, tuple)) and payload and all(isinstance(item, dict) for item in payload):
            normalized = {}
            keys = {key for item in payload for key in item.keys()}
            for key in keys:
                try:
                    series = [float(item[key]) for item in payload if key in item]
                except Exception:
                    continue
                if series:
                    normalized[str(key)] = series
            return normalized

        return {}

    def _extract_model_history(self, model: Any, model_key: str) -> dict[str, Any]:
        if model is None:
            return {"available": False, "model": model_key, "reason": "当前没有可用模型对象。"}

        for attr_name in ("history", "training_history", "history_", "_history", "learning_curve", "metrics_history"):
            if not hasattr(model, attr_name):
                continue
            payload = getattr(model, attr_name)
            normalized = self._normalize_history_payload(payload)
            if normalized:
                return {
                    "available": True,
                    "model": model_key,
                    "source_attribute": attr_name,
                    "history": normalized,
                }

        return {"available": False, "model": model_key, "reason": "当前 ANI/MLatom 接口没有暴露结构化 epoch history。"}

    def _artifact_paths(self, workdir: Path) -> dict[str, Path]:
        return {
            "training_split_file": workdir / "training_split.json",
            "train_main_predictions_file": workdir / "train_main_predictions.csv",
            "train_aux_predictions_file": workdir / "train_aux_predictions.csv",
            "train_main_history_file": workdir / "train_main_history.json",
            "train_aux_history_file": workdir / "train_aux_history.json",
        }

    def _build_prediction_rows(
        self,
        *,
        trainingdb_ref,
        trainingdb,
        split_labels: list[str],
        model_key: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        prediction_attr = "energy" if model_key == "main_model" else "aux_energy"

        for index, (ref_molecule, pred_molecule, split_label) in enumerate(zip(trainingdb_ref, trainingdb, split_labels)):
            sample_id = str(getattr(ref_molecule, "id", f"sample_{index:04d}"))
            y_true = self._safe_float(getattr(ref_molecule, "energy", None))
            y_pred = self._safe_float(getattr(pred_molecule, prediction_attr, None))
            residual = None if y_true is None or y_pred is None else y_pred - y_true
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "split": split_label,
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": residual,
                "abs_error": None if residual is None else abs(residual),
                "reference_energy": self._safe_float(getattr(ref_molecule, "reference_energy", None)),
                "predicted_energy_main": self._safe_float(getattr(pred_molecule, "energy", None)),
                "uncertainty": self._safe_float(getattr(pred_molecule, "uq", None)),
            }
            if model_key == "main_model":
                true_grad = self._safe_vector_property(ref_molecule, "energy_gradients")
                pred_grad = self._safe_vector_property(pred_molecule, "energy_gradients")
                if true_grad is not None:
                    true_force = -true_grad
                    row["true_gradient_norm"] = float(np.linalg.norm(true_grad))
                    row["true_force_norm"] = float(np.linalg.norm(true_force))
                if pred_grad is not None:
                    pred_force = -pred_grad
                    row["pred_gradient_norm"] = float(np.linalg.norm(pred_grad))
                    row["pred_force_norm"] = float(np.linalg.norm(pred_force))
                if true_grad is not None and pred_grad is not None:
                    grad_diff = pred_grad - true_grad
                    force_diff = (-pred_grad) - (-true_grad)
                    row["gradient_rmse"] = float(np.sqrt(np.mean(np.square(grad_diff))))
                    row["force_error_norm"] = float(np.linalg.norm(force_diff))
            else:
                row["predicted_energy_aux"] = self._safe_float(getattr(pred_molecule, "aux_energy", None))
            rows.append(row)
        return rows

    def write_training_artifacts(
        self,
        *,
        workdir: Path,
        summary: dict[str, Any],
        summary_details: dict[str, Any],
    ) -> dict[str, str]:
        artifact_paths = self._artifact_paths(workdir)
        split_rows = summary_details.get("split_rows", [])

        write_json(
            artifact_paths["training_split_file"],
            {
                "num_subtrain": summary.get("num_subtrain", 0),
                "num_validation": summary.get("num_validation", 0),
                "subtrain_sample_ids": [item["sample_id"] for item in split_rows if item["split"] == "subtrain"],
                "validation_sample_ids": [item["sample_id"] for item in split_rows if item["split"] == "validation"],
                "rows": split_rows,
            },
        )
        write_csv_rows(
            artifact_paths["train_main_predictions_file"],
            summary_details.get("main_prediction_rows", []),
            fieldnames=[
                "sample_id",
                "split",
                "y_true",
                "y_pred",
                "residual",
                "abs_error",
                "reference_energy",
                "predicted_energy_main",
                "uncertainty",
                "true_gradient_norm",
                "pred_gradient_norm",
                "gradient_rmse",
                "true_force_norm",
                "pred_force_norm",
                "force_error_norm",
            ],
        )
        write_csv_rows(
            artifact_paths["train_aux_predictions_file"],
            summary_details.get("aux_prediction_rows", []),
            fieldnames=[
                "sample_id",
                "split",
                "y_true",
                "y_pred",
                "residual",
                "abs_error",
                "reference_energy",
                "predicted_energy_main",
                "predicted_energy_aux",
                "uncertainty",
            ],
        )

        main_history = self._extract_model_history(self.main_model, "main_model")
        aux_history = self._extract_model_history(self.aux_model, "aux_model")
        if "main_model" in summary:
            main_history["final_metrics"] = summary["main_model"]
        if "aux_model" in summary:
            aux_history["final_metrics"] = summary["aux_model"]
        write_json(artifact_paths["train_main_history_file"], main_history)
        write_json(artifact_paths["train_aux_history_file"], aux_history)
        return {key: str(path.resolve()) for key, path in artifact_paths.items()}

    def summary(self, *, subtraindb, valdb, include_details: bool = False):
        summary = {
            "num_subtrain": len(subtraindb),
            "num_validation": len(valdb),
            "uq_threshold": self.uq_threshold,
        }
        if self.main_model is None:
            empty_details = {"split_rows": [], "main_prediction_rows": [], "aux_prediction_rows": []}
            return (summary, empty_details) if include_details else summary

        trainingdb_ref = subtraindb + valdb
        trainingdb = trainingdb_ref.copy()
        self.predict(molecular_database=trainingdb)

        n_subtrain = len(subtraindb)
        split_labels = ["subtrain"] * n_subtrain + ["validation"] * len(valdb)
        split_rows = [
            {"sample_id": str(getattr(molecule, "id", f"sample_{index:04d}")), "split": split_labels[index]}
            for index, molecule in enumerate(trainingdb_ref)
        ]
        values = trainingdb_ref.get_properties("energy")
        predicted_values = trainingdb.get_properties("energy")
        gradients = trainingdb_ref.get_xyz_vectorial_properties("energy_gradients")
        predicted_gradients = trainingdb.get_xyz_vectorial_properties("energy_gradients")

        summary["main_model"] = {
            "subtrain_energy_rmse": float(ml.stats.rmse(predicted_values[:n_subtrain], values[:n_subtrain])),
            "validation_energy_rmse": float(ml.stats.rmse(predicted_values[n_subtrain:], values[n_subtrain:])),
            "subtrain_energy_pcc": float(ml.stats.correlation_coefficient(predicted_values[:n_subtrain], values[:n_subtrain])),
            "validation_energy_pcc": float(ml.stats.correlation_coefficient(predicted_values[n_subtrain:], values[n_subtrain:])),
            "subtrain_gradient_rmse": float(ml.stats.rmse(predicted_gradients[:n_subtrain].flatten(), gradients[:n_subtrain].flatten())),
            "validation_gradient_rmse": float(ml.stats.rmse(predicted_gradients[n_subtrain:].flatten(), gradients[n_subtrain:].flatten())),
        }
        if self.aux_model is not None:
            aux_values = trainingdb.get_properties("aux_energy")
            summary["aux_model"] = {
                "subtrain_energy_rmse": float(ml.stats.rmse(aux_values[:n_subtrain], values[:n_subtrain])),
                "validation_energy_rmse": float(ml.stats.rmse(aux_values[n_subtrain:], values[n_subtrain:])),
                "subtrain_energy_pcc": float(ml.stats.correlation_coefficient(aux_values[:n_subtrain], values[:n_subtrain])),
                "validation_energy_pcc": float(ml.stats.correlation_coefficient(aux_values[n_subtrain:], values[n_subtrain:])),
            }

        if not include_details:
            return summary

        return summary, {
            "split_rows": split_rows,
            "main_prediction_rows": self._build_prediction_rows(
                trainingdb_ref=trainingdb_ref,
                trainingdb=trainingdb,
                split_labels=split_labels,
                model_key="main_model",
            ),
            "aux_prediction_rows": self._build_prediction_rows(
                trainingdb_ref=trainingdb_ref,
                trainingdb=trainingdb,
                split_labels=split_labels,
                model_key="aux_model",
            )
            if self.aux_model is not None
            else [],
        }
