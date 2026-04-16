from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minimal_adl.config import load_config
from minimal_adl.h2_sampling import run_md_sampling_round
from minimal_adl.io_utils import write_json
from minimal_adl.pbs import launch_python_job


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 H2 主副模型驱动的 MD，并在首个不确定点停止。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--round-index", type=int, required=True, help="当前采样轮次编号。")
    parser.add_argument("--num-initial-conditions", type=int, default=None, help="覆盖本轮初始条件个数。")
    parser.add_argument("--submit-mode", choices=["local", "pbs"], default="local", help="运行方式。")
    parser.add_argument("--wait", dest="wait", action="store_true", help="PBS 提交后等待完成。")
    parser.add_argument("--no-wait", dest="wait", action="store_false", help="PBS 提交后直接返回。")
    parser.add_argument("--device", default=None, help="覆盖设备，例如 cpu 或 cuda。")
    parser.add_argument("--maximum-propagation-time", type=float, default=None, help="覆盖最大传播时间 fs。")
    parser.add_argument("--time-step", type=float, default=None, help="覆盖 MD 时间步 fs。")
    parser.add_argument("--save-interval-steps", type=int, default=None, help="每隔多少步保存一帧。")
    parser.add_argument("--status-file", default=None, help="可选状态 JSON 路径。")
    parser.set_defaults(wait=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device

    status_path = (
        Path(args.status_file)
        if args.status_file
        else Path(config["paths"]["results_dir"]) / f"round_{args.round_index:03d}_md_sampling_status.json"
    )
    status_path = status_path.resolve()

    if args.submit_mode == "pbs":
        job_info = launch_python_job(
            config=config,
            job_key="md_sampling",
            submit_mode="pbs",
            wait=args.wait,
            script_path=Path(__file__),
            script_args=[
                "--config",
                str(Path(config["config_path"]).resolve()),
                "--round-index",
                str(args.round_index),
                "--submit-mode",
                "local",
                "--status-file",
                str(status_path),
                *([] if args.num_initial_conditions is None else ["--num-initial-conditions", str(args.num_initial_conditions)]),
                *([] if args.device is None else ["--device", args.device]),
                *([] if args.maximum_propagation_time is None else ["--maximum-propagation-time", str(args.maximum_propagation_time)]),
                *([] if args.time_step is None else ["--time-step", str(args.time_step)]),
                *([] if args.save_interval_steps is None else ["--save-interval-steps", str(args.save_interval_steps)]),
            ],
            output_dir=Path(config["paths"]["results_dir"]) / "jobs" / f"md_round_{args.round_index:03d}",
            status_file=status_path,
            job_name=f"adl_md_r{args.round_index:03d}",
        )
        print(f"已提交 MD 任务：{job_info}")
        return

    try:
        payload = run_md_sampling_round(
            config=config,
            round_index=args.round_index,
            number_of_initial_conditions=int(args.num_initial_conditions or config.get("sampling", {}).get("initial_conditions_per_round", 100)),
            maximum_propagation_time=args.maximum_propagation_time,
            time_step=args.time_step,
            save_interval_steps=args.save_interval_steps,
            device_override=args.device,
        )
        write_json(
            status_path,
            {
                "success": True,
                "round_index": args.round_index,
                "frame_manifest_file": payload["frame_manifest_file"],
                "trajectory_summary_file": payload["trajectory_summary_file"],
                "num_trajectories": payload["num_trajectories"],
                "num_frames": payload["num_frames"],
                "num_uncertain_points": payload.get("num_candidate_samples"),
            },
        )
        print(
            f"MD round {args.round_index} completed: "
            f"{payload['num_trajectories']} trajectories, {payload.get('num_candidate_samples', payload['num_frames'])} uncertain points"
        )
    except Exception as exc:
        write_json(
            status_path,
            {
                "success": False,
                "round_index": args.round_index,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
