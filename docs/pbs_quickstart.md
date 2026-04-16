# PBS 快速运行清单

## 1. 拉取代码

```bash
mkdir -p /share/home/Chenlehui/work
cd /share/home/Chenlehui/work
git clone https://github.com/LehuiChen/test_AL_H2.git
cd test_AL_H2
```

## 2. 初始化交互式运行环境

```bash
source /share/apps/gaussian/g16-env.sh
source /share/env/ips2018u1.env
source ~/.bashrc
conda activate ADL_env
export PATH=/share/apps/gaussian/g16:/share/pubbin:/share/home/Chenlehui/bin:/share/apps/xtb-6.7.1/xtb-dist/bin:$PATH
export dftd4bin=/share/apps/dftd4-3.5.0/bin/dftd4
```

说明：

- 这段是给你在登录节点手动检查环境、手动启动主控器时使用的。
- PBS 里的 `target` 标注作业也已经在配置文件里内置了同一套 Gaussian 环境块。

## 3. 检查环境

```bash
python scripts/check_environment.py --config configs/base.yaml --strict
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-g16
```

## 4. 冒烟运行

```bash
python scripts/active_learning_loop.py \
  --config configs/base_smoke.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs
```

## 5. 全量运行

```bash
python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs
```

如果前面已经跑过 `configs/base_smoke.yaml`，建议先重新拉一个干净仓库，或者清理以下目录后再跑 full：

```bash
rm -rf results models labels/gaussian data/raw data/processed
mkdir -p results models labels/gaussian data/raw data/processed logs
```

## 6. 后台挂载

```bash
mkdir -p logs
nohup python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  > logs/active_learning.out 2>&1 &
```

## 7. 查看状态与验收

```bash
qstat -u "$USER"
python scripts/inspect_al_results.py --results-dir results --min-new-points 5
```
