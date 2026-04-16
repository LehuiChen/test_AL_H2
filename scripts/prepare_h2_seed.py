from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.h2_seed import prepare_h2_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="把 H2 平衡结构和频率文件整理成统一 seed 产物。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--xyz", default=None, help="覆盖 H2 xyz 输入路径。")
    parser.add_argument("--freq-json", default=None, help="覆盖 H2 频率 JSON 路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = prepare_h2_seed(
        xyz_source_path=args.xyz or config["paths"]["h2_xyz_source"],
        frequency_json_source_path=args.freq_json or config["paths"]["h2_frequency_source"],
        xyz_output_path=config["paths"]["h2_seed_xyz"],
        json_output_path=config["paths"]["h2_seed_json"],
        summary_output_path=config["paths"]["h2_seed_summary_file"],
    )
    print(
        "Prepared H2 seed: "
        f"{summary['h2_seed_json']} / {summary['h2_seed_xyz']} "
        f"(num_atoms = {summary['num_atoms']})"
    )


if __name__ == "__main__":
    main()
