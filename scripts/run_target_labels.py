from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.label_jobs import launch_label_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="批量提交 Gaussian target 标注任务。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--manifest", required=True, help="几何 manifest 路径。")
    parser.add_argument("--submit-mode", choices=["local", "pbs"], default="pbs", help="提交方式。")
    parser.add_argument("--wait", dest="wait", action="store_true", help="等待所有 PBS 作业结束。")
    parser.add_argument("--no-wait", dest="wait", action="store_false", help="提交后直接返回。")
    parser.add_argument("--force", action="store_true", help="即使结果已存在也重新提交。")
    parser.set_defaults(wait=True)
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = launch_label_jobs(
        config=config,
        manifest_path=args.manifest,
        method_key="target",
        submit_mode=args.submit_mode,
        wait=args.wait,
        force=args.force,
    )
    print(f"target 任务处理完成，共 {len(jobs)} 个样本。")


if __name__ == "__main__":
    main()
