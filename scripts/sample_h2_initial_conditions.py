from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.h2_sampling import generate_h2_initial_conditions
from minimal_adl.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="从 H2 seed 采样 round-0 或后续轮次的初始条件。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--round-index", type=int, default=0, help="写入 manifest 的轮次编号。")
    parser.add_argument("--num-initial-conditions", type=int, default=None, help="覆盖本轮采样数。")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录。")
    parser.add_argument("--manifest", default=None, help="覆盖 manifest 输出路径。")
    parser.add_argument("--summary-output", default=None, help="可选 JSON 汇总输出路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    sampling_cfg = config.get("sampling", {})
    default_count = (
        sampling_cfg.get("initial_conditions_initial", 200)
        if args.round_index == 0
        else sampling_cfg.get("initial_conditions_per_round", 100)
    )
    number_of_initial_conditions = int(args.num_initial_conditions or default_count)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "raw" / f"round_{args.round_index:03d}_initial_conditions"
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(config["paths"]["results_dir"]) / f"round_{args.round_index:03d}_initial_conditions_manifest.json"
    )
    payload = generate_h2_initial_conditions(
        config=config,
        round_index=args.round_index,
        number_of_initial_conditions=number_of_initial_conditions,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )

    if args.summary_output:
        write_json(args.summary_output, payload)
    print(
        "Sampled H2 initial conditions: "
        f"{payload['num_initial_conditions']} -> {payload['manifest_file']}"
    )


if __name__ == "__main__":
    main()
