from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.h2_sampling import select_md_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="从 MD 返回帧中选择下一轮需要标注的不确定点。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--round-index", type=int, required=True, help="当前轮次编号。")
    parser.add_argument("--frame-manifest", default=None, help="覆盖 MD 帧 manifest 路径。")
    parser.add_argument("--max-new-points", type=int, default=None, help="显式限制本轮新增点数量。")
    parser.add_argument("--dedup-rmsd-threshold", type=float, default=None, help="覆盖去重 RMSD 阈值。")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = select_md_frames(
        config=config,
        round_index=args.round_index,
        frame_manifest_path=args.frame_manifest,
        max_new_points=args.max_new_points,
        dedup_rmsd_threshold=args.dedup_rmsd_threshold,
    )
    print(
        f"Round {args.round_index} selection finished: "
        f"{summary['selected_count']} kept, converged = {summary['converged']}"
    )


if __name__ == "__main__":
    main()
