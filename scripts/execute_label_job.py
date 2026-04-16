from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import get_method_config, load_config
from minimal_adl.mlatom_bridge import run_and_save_label_job


def main() -> None:
    parser = argparse.ArgumentParser(description="执行单个 Gaussian target 标注任务。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--geometry", required=True, help="单个几何文件路径。")
    parser.add_argument("--method-key", required=True, choices=["target"], help="方法键名。")
    parser.add_argument("--output-dir", required=True, help="作业输出目录。")
    args = parser.parse_args()

    config = load_config(args.config)
    method_config = get_method_config(config, args.method_key)
    run_and_save_label_job(
        geometry_path=args.geometry,
        method_config=method_config,
        output_dir=args.output_dir,
        method_key=args.method_key,
    )
    print(f"{args.method_key} 标注完成：{args.geometry}")


if __name__ == "__main__":
    main()
