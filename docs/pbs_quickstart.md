# PBS 快速运行清单

## 1. 拉取代码

```bash
mkdir -p /share/home/Chenlehui/work
cd /share/home/Chenlehui/work
git clone https://github.com/LehuiChen/test_AL_H2.git
cd test_AL_H2
```

## 2. 激活环境

```bash
source ~/.bashrc
conda activate ADL_env
```

## 3. 加载 Gaussian 与相关环境

```bash
source /share/apps/gaussian/g16-env.sh
source /share/env/ips2018u1.env
source ~/.bashrc
export PATH=/share/apps/gaussian/g16:/share/pubbin:/share/home/Chenlehui/bin:/share/apps/xtb-6.7.1/xtb-dist/bin:$PATH
export dftd4bin=/share/apps/dftd4-3.5.0/bin/dftd4
```

## 4. 检查环境

```bash
python scripts/check_environment.py --config configs/base.yaml --strict
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-g16
```

## 5. 冒烟运行

```bash
python scripts/active_learning_loop.py \
  --config configs/base_smoke.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs
```

## 6. 全量运行

```bash
python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs
```

## 7. 后台挂载

```bash
mkdir -p logs
nohup python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  > logs/active_learning.out 2>&1 &
```

## 8. 查看状态与验收

```bash
qstat -u "$USER"
python scripts/inspect_al_results.py --results-dir results --min-new-points 5
```
