#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("check_environment.py 依赖 PyYAML，请先安装。") from exc

    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Config must be a YAML mapping: %s" % path)
    payload["_config_file"] = str(path)
    return payload


def check_python_module(module_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"module": module_name, "ok": False}
    try:
        module = importlib.import_module(module_name)
        result["ok"] = True
        result["file"] = getattr(module, "__file__", None)
        result["version"] = getattr(module, "__version__", None)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
    return result


def check_any_python_module(candidates: List[str], *, check_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "check": check_name,
        "candidates": list(candidates),
        "ok": False,
        "resolved_module": None,
        "details": {},
    }
    for name in candidates:
        probe = check_python_module(name)
        result["details"][name] = probe
        if probe.get("ok", False):
            result["ok"] = True
            result["resolved_module"] = name
            break
    if not result["ok"]:
        result["error_message"] = "无法导入以下任一模块: %s" % ", ".join(candidates)
    return result


def check_command(command_name: str, version_args: Optional[List[str]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"command": command_name, "ok": False}
    command_path = shutil.which(command_name)
    result["path"] = command_path
    if command_path is None:
        result["error_message"] = "PATH 中未找到命令 `%s`。" % command_name
        return result

    result["ok"] = True
    if version_args:
        try:
            proc = subprocess.run(
                [command_name, *version_args],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            result["returncode"] = proc.returncode
            result["stdout"] = proc.stdout.strip()
            result["stderr"] = proc.stderr.strip()
        except Exception as exc:
            result["version_error"] = "%s: %s" % (type(exc).__name__, exc)
    return result


def check_torch_status(expect_gpu: bool) -> Dict[str, Any]:
    result = check_python_module("torch")
    if not result.get("ok", False):
        return result
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["expected_gpu"] = bool(expect_gpu)
        if result["cuda_available"]:
            result["device_name"] = torch.cuda.get_device_name(0)
        elif expect_gpu:
            result["warning"] = "请求了 GPU 检查，但 torch.cuda.is_available() 为 False。"
    except Exception as exc:
        result["ok"] = False
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
    return result


def _best_effort_call(func: Any, kwargs: Dict[str, Any]) -> Any:
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


def _new_molecule(ml: Any) -> Any:
    if hasattr(ml, "data") and hasattr(ml.data, "molecule"):
        return ml.data.molecule()
    if hasattr(ml, "molecule"):
        return ml.molecule()
    raise RuntimeError("Cannot locate molecule class in current mlatom version.")


def run_optional_mlatom_g16_test(config: Dict[str, Any]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"test": "mlatom_g16_single_point", "ok": False}
    try:
        import mlatom as ml  # type: ignore

        system_cfg = config.get("system", {})
        reference_cfg = config.get("reference", {})
        xyz_path = Path(system_cfg.get("xyz", "inputs/h2.xyz"))
        if not xyz_path.is_absolute():
            xyz_path = (PROJECT_ROOT / xyz_path).resolve()
        if not xyz_path.exists():
            raise FileNotFoundError("g16 测试缺少 xyz 文件: %s" % xyz_path)

        molecule = _new_molecule(ml)
        try:
            molecule.load(str(xyz_path), format="xyz")
        except TypeError:
            molecule.load(str(xyz_path), "xyz")

        method = str(reference_cfg.get("refmethod", "B3LYP/6-31G*"))
        qmprog = str(reference_cfg.get("qmprog", "gaussian"))
        nthreads = int(reference_cfg.get("nthreads", 2))
        model = None
        candidates = [
            {"method": method, "program": qmprog, "nthreads": nthreads, "save_files_in_current_directory": False},
            {"method": method, "qmprog": qmprog, "nthreads": nthreads},
            {"method": method, "program": qmprog},
            {"method": method},
        ]
        for kwargs in candidates:
            try:
                model = _best_effort_call(ml.models.methods, kwargs)
                break
            except Exception:
                continue
        if model is None:
            raise RuntimeError("无法构建 g16 单点测试所需参考方法。")

        _best_effort_call(model.predict, {"molecule": molecule, "calculate_energy": True})
        report["ok"] = True
        report["xyz_file"] = str(xyz_path)
        report["energy"] = float(getattr(molecule, "energy"))
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 ANI-H2 AL 运行环境（直接学习，不依赖 XTB）。"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--expect-gpu", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--test-mlatom-g16", action="store_true")
    parser.add_argument(
        "--test-mlatom-xtb",
        action="store_true",
        help="ADL 兼容参数；本直接学习流程中保留为 no-op。",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    checks: Dict[str, Any] = {
        "yaml": check_python_module("yaml"),
        "mlatom": check_python_module("mlatom"),
        "torch": check_torch_status(expect_gpu=args.expect_gpu),
        "torchani": check_any_python_module(["torchani"], check_name="torchani"),
        "g16": check_command("g16", None),
        "xtb": check_command("xtb", ["--version"]),
    }
    if args.test_mlatom_g16:
        checks["mlatom_g16_single_point"] = run_optional_mlatom_g16_test(config)
    if args.test_mlatom_xtb:
        checks["mlatom_xtb_single_point"] = {
            "ok": True,
            "skipped": True,
            "note": "本直接学习 AL 流程不包含 XTB 测试。",
        }

    report: Dict[str, Any] = {
        "config_file": str(Path(config["_config_file"]).resolve()),
        "python": sys.executable,
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
        required = ["yaml", "mlatom", "torch", "torchani", "g16"]
        failed = [name for name in required if not report["checks"].get(name, {}).get("ok", False)]
        if args.test_mlatom_g16 and not report["checks"].get("mlatom_g16_single_point", {}).get("ok", False):
            failed.append("mlatom_g16_single_point")
        if failed:
            raise SystemExit("环境必需项检测失败: " + ", ".join(failed))


if __name__ == "__main__":
    main()
