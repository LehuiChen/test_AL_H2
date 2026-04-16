from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "当前环境缺少 PyYAML。请先在 ADL_env 中安装 `PyYAML` 后再运行脚本。"
        ) from exc
    return yaml


def _resolve_path_values(node: Any, project_root: Path) -> Any:
    """把 paths 节中的相对路径统一解析成绝对路径。"""

    if isinstance(node, dict):
        return {key: _resolve_path_values(value, project_root) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_path_values(item, project_root) for item in node]
    if isinstance(node, str):
        candidate = Path(node)
        if candidate.is_absolute():
            return str(candidate)
        return str((project_root / candidate).resolve())
    return node


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并补齐 project_root / config_path / paths。"""

    yaml = _import_yaml()
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {config_path}")

    config["config_path"] = str(config_path)
    config["project_root"] = str(project_root)
    config["paths"] = _resolve_path_values(config.get("paths", {}), project_root)
    return config


def get_method_config(config: dict[str, Any], method_key: str) -> dict[str, Any]:
    methods = config.get("methods", {})
    if method_key not in methods:
        raise KeyError(f"配置文件中找不到方法 `{method_key}`。")
    method_config = methods[method_key]
    if not isinstance(method_config, dict):
        raise TypeError(f"方法 `{method_key}` 的配置必须是字典。")
    return method_config
