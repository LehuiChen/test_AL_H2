# test_AL_H2

本仓库用于复现 H2 的主动学习（AL）流程，接口风格与 `minimal_adl_ethene_butadiene_ANI` 保持一致：

- 直接学习（不走 delta 管线）
- 训练后端固定 ANI
- 参考标注使用 Gaussian `g16`
- 支持 PBS 一键提交

## 1. 服务器拉取仓库

```bash
mkdir -p /share/home/Chenlehui/work
cd /share/home/Chenlehui/work
git clone https://github.com/LehuiChen/test_AL_H2.git
cd test_AL_H2
```

## 2. 环境要求（ADL_env）

建议运行环境：`ADL_env`

必须可用：

- `python`
- `mlatom`
- `torch`
- `torchani`
- `g16`

可选：

- `xtb` 可以存在，但本流程不依赖 `xtb` 参与学习或标注。

环境激活：

```bash
source ~/.bashrc
conda activate ADL_env
```

## 3. 配置文件说明

主要配置：

- `configs/base_smoke.yaml`：冒烟配置
- `configs/base.yaml`：完整配置

常用可改项：

- `cluster.queue/nodes/ppn/walltime`
- `cluster.submit_command/cluster.python_command`
- `cluster.env_blocks.default`（按你集群环境加载 g16）
- `reference.refmethod/reference.qmprog`
- `active_learning.*`
- `uncertainty.*`

主副模型不确定性配置示例：

```yaml
uncertainty:
  committee_size: 2
  metric: energy_forces
  uq_threshold: null
```

## 4. 环境检测

严格检测：

```bash
python scripts/check_environment.py --config configs/base.yaml --strict
```

可选 g16 + MLatom 单点测试：

```bash
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-g16
```

ADL 兼容参数（保留接口）：

```bash
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-xtb
```

说明：`--test-mlatom-xtb` 在本项目中会被接受，但不会执行 xtb 测试。

## 5. 一键运行 AL

主命令（ADL 风格）：

```bash
python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  --no-wait
```

快捷脚本：

```bash
bash scripts/run_smoke.sh
bash scripts/run_full.sh
```

## 6. 作业监控与结果验收

查看队列：

```bash
qstat -u "$USER"
```

输出目录（`runs/h2_ani_smoke` 或 `runs/h2_ani_full`）重点文件：

- `submission.json`（含 `job_id`）
- `status.json`
- `run_config.json`
- `al_info.json`
- `active_learning_experiment_summary.json`

结果检查：

```bash
python scripts/inspect_al_results.py --run-dir runs/h2_ani_smoke --min-new-points 5
python scripts/inspect_al_results.py --run-dir runs/h2_ani_full --min-new-points 5
```

## 7. 论文与方法对齐

- 仓库论文 PDF：
  `physics-informed-active-learning-for-accelerating-quantum-chemical-simulations.pdf`
- DOI：
  `10.1021/acs.jctc.4c00821`
- 方法对齐说明：
  `docs/h2_al_paper_alignment.md`
- PBS 快速清单：
  `docs/pbs_quickstart.md`
