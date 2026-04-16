from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.mlatom_bridge import create_mlatom_method


def check_python_module(module_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"module": module_name, "ok": False}
    try:
        module = importlib.import_module(module_name)
        payload["ok"] = True
        payload["file"] = getattr(module, "__file__", None)
        payload["version"] = getattr(module, "__version__", None)
    except Exception as exc:
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    return payload


def check_any_python_module(candidates: list[str], *, check_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "check": check_name,
        "candidates": candidates,
        "ok": False,
        "resolved_module": None,
        "details": {},
    }
    for module_name in candidates:
        result = check_python_module(module_name)
        payload["details"][module_name] = result
        if result.get("ok", False):
            payload["ok"] = True
            payload["resolved_module"] = module_name
            break
    if not payload["ok"]:
        payload["error_message"] = "Could not import any of: " + ", ".join(candidates)
    return payload


def check_command(command_name: str, version_args: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"command": command_name, "ok": False}
    command_path = shutil.which(command_name)
    payload["path"] = command_path
    if command_path is None:
        payload["error_message"] = f"Command `{command_name}` was not found in PATH."
        return payload

    payload["ok"] = True
    if version_args:
        try:
            result = subprocess.run(
                [command_name, *version_args],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            payload["returncode"] = result.returncode
            payload["stdout"] = result.stdout.strip()
            payload["stderr"] = result.stderr.strip()
        except Exception as exc:
            payload["version_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def check_torch_status(expect_gpu: bool) -> dict[str, Any]:
    payload = check_python_module("torch")
    if not payload.get("ok"):
        return payload

    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_version"] = torch.version.cuda
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["expected_gpu"] = expect_gpu
        if payload["cuda_available"]:
            payload["device_name"] = torch.cuda.get_device_name(0)
        elif expect_gpu:
            payload["warning"] = "GPU was requested for checking, but torch.cuda.is_available() is False."
    except Exception as exc:
        payload["ok"] = False
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    return payload


def run_optional_mlatom_g16_test(config_path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"test": "mlatom_g16_single_point", "ok": False}
    try:
        config = load_config(config_path)
        import mlatom as ml

        geometry_path = Path(config["paths"]["h2_xyz_source"]).resolve()
        molecule = ml.data.molecule()
        molecule.load(str(geometry_path), format="xyz")
        method = create_mlatom_method(config["methods"]["target"])
        method.predict(
            molecule=molecule,
            calculate_energy=True,
            calculate_energy_gradients=True,
            calculate_hessian=False,
        )
        payload["ok"] = True
        payload["geometry_file"] = str(geometry_path)
        payload["energy"] = float(molecule.energy)
    except Exception as exc:
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 H2 直接学习 ANI 主动学习环境。")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "base.yaml"), help="YAML 配置文件路径。")
    parser.add_argument("--expect-gpu", action="store_true", help="额外检查当前会话 CUDA 是否可用。")
    parser.add_argument("--test-mlatom-g16", action="store_true", help="执行一次最小 mlatom + g16 单点测试。")
    parser.add_argument("--test-mlatom-xtb", action="store_true", help="兼容旧接口；当前流程不会执行 xtb 测试。")
    parser.add_argument("--json-output", default=None, help="可选 JSON 报告路径。")
    parser.add_argument("--strict", action="store_true", help="关键检查失败时返回非零退出码。")
    args = parser.parse_args()

    config = load_config(args.config)
    checks: dict[str, Any] = {
        "yaml": check_python_module("yaml"),
        "pandas": check_python_module("pandas"),
        "matplotlib": check_python_module("matplotlib"),
        "seaborn": check_python_module("seaborn"),
        "mlatom": check_python_module("mlatom"),
        "pyh5md": check_python_module("pyh5md"),
        "joblib": check_python_module("joblib"),
        "sklearn": check_python_module("sklearn"),
        "torch": check_torch_status(expect_gpu=args.expect_gpu),
        "torchani": check_any_python_module(["torchani"], check_name="torchani"),
        "g16": check_command("g16", None),
        "Gau_Mlatom.py": check_command("Gau_Mlatom.py", None),
    }
    if args.test_mlatom_g16:
        checks["mlatom_g16_single_point"] = run_optional_mlatom_g16_test(config["config_path"])
    if args.test_mlatom_xtb:
        checks["mlatom_xtb_single_point"] = {
            "ok": True,
            "skipped": True,
            "note": "当前 H2 直接学习工作流不依赖 xtb，保留该参数仅为兼容旧命令。",
        }

    report: dict[str, Any] = {
        "config_file": str(Path(config["config_path"]).resolve()),
        "python": sys.executable,
        "ml_model_type": str(config.get("training", {}).get("ml_model_type", "ANI")),
        "checks": checks,
    }

    if args.json_output:
        out_path = Path(args.json_output)
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.strict:
        required_keys = ["yaml", "pandas", "matplotlib", "seaborn", "mlatom", "pyh5md", "joblib", "sklearn", "torch", "torchani", "g16"]
        failed_checks = [key for key in required_keys if not report["checks"].get(key, {}).get("ok", False)]
        if args.test_mlatom_g16 and not report["checks"].get("mlatom_g16_single_point", {}).get("ok", False):
            failed_checks.append("mlatom_g16_single_point")
        if failed_checks:
            raise SystemExit("Required environment checks failed: " + ", ".join(failed_checks))


if __name__ == "__main__":
    main()
