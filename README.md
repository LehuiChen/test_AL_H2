# test_AL_H2

这个仓库按 `minimal_adl_ethene_butadiene_ANI` 的骨架组织，但算法语义改成了 H2 的直接学习主动学习。

当前唯一入口：

- `python scripts/active_learning_loop.py --config ... --submit-mode-labels pbs --submit-mode-train pbs --submit-mode-md pbs`

当前工作流特征：

- 标注只走 Gaussian `g16` + `B3LYP/6-31G*`
- 训练后端固定为 ANI
- 不构造 delta 数据集
- 不依赖 xtb 参与学习或标注
- 不确定性采用主模型 / 副模型的 energy disagreement

## 目录结构

核心产物目录与参考 ADL 仓库保持同构：

- `configs/`：配置文件
- `inputs/`：H2 输入，仅保留 `h2.xyz` 和 `h2_freq.json`
- `data/raw/`：采样与筛选后的原始几何
- `data/processed/`：累计 manifest 与训练数据集
- `labels/gaussian/`：Gaussian 标注结果
- `models/`：主模型、副模型、训练诊断
- `results/`：轮次选择汇总、实验总汇总、环境检测报告
- `scripts/`：多阶段调度器与阶段脚本
- `src/minimal_adl/`：对齐参考 ADL 写法的最小实现模块

## 环境要求

推荐环境：`ADL_env`

关键依赖：

- `python`
- `PyYAML`
- `mlatom`
- `torch`
- `torchani`
- `g16`
- `pyh5md`
- `joblib`
- `scikit-learn`

激活环境：

```bash
source ~/.bashrc
conda activate ADL_env
```

## 配置说明

全量配置：`configs/base.yaml`

冒烟配置：`configs/base_smoke.yaml`

当前资源分层与参考 ADL 对齐：

- `target`：`queue=default`，`ppn=16`，`worker_count=4`，`walltime=360:00:00`
- `training`：`queue=GPU`，`ppn=24`，`walltime=360:00:00`
- `uncertainty`：`queue=GPU`，`ppn=24`，`walltime=360:00:00`
- `md_sampling`：`queue=GPU`，`ppn=24`，`walltime=360:00:00`

Gaussian 环境块也按参考 ADL 的写法保留了：

- `PERLLIB` 保护
- `set +u` / `set +o pipefail`
- `GAUSS_SCRDIR` 创建与退出清理

## 环境检测

严格检测：

```bash
python scripts/check_environment.py --config configs/base.yaml --strict
```

加一轮最小 `mlatom + g16` 单点测试：

```bash
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-g16
```

兼容旧命令的 `xtb` 开关还保留，但这里只是 no-op：

```bash
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-xtb
```

## 一键运行

冒烟：

```bash
python scripts/active_learning_loop.py \
  --config configs/base_smoke.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs
```

全量：

```bash
python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs
```

也可以直接用包装脚本：

```bash
bash scripts/run_smoke.sh
bash scripts/run_full.sh
```

## 后台挂载运行

这套主控器是阻塞式协调器。要后台挂着跑，就用外层 shell 管理，而不是让 Python 主程序立即返回。

推荐 `nohup`：

```bash
mkdir -p logs
nohup python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  > logs/active_learning.out 2>&1 &
```

然后查看后台 PID：

```bash
jobs -l
ps -fu "$USER" | grep active_learning_loop.py
```

如果你更习惯会话保活，也可以用 `screen` 或 `tmux`。

## 结果查看

重点文件：

- `results/check_environment_latest.json`
- `results/pipeline_run_summary.json`
- `results/active_learning_experiment_summary.json`
- `results/active_learning_round_history.json`
- `results/round_*_selection_summary.json`
- `results/round_*_selected_manifest.json`
- `models/training_diagnostics.json`

快速验收：

```bash
python scripts/inspect_al_results.py --results-dir results --min-new-points 5
```

查看 PBS：

```bash
qstat -u "$USER"
```

## 论文与方法对齐

仓库内论文：

- `physics-informed-active-learning-for-accelerating-quantum-chemical-simulations.pdf`

补充说明：

- `docs/h2_al_paper_alignment.md`
- `docs/pbs_quickstart.md`
