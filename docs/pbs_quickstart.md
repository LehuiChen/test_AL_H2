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

## 3. 环境检测

```bash
python scripts/check_environment.py --config configs/base.yaml --strict
python scripts/check_environment.py --config configs/base.yaml --strict --test-mlatom-g16
```

## 4. 提交冒烟任务

```bash
python scripts/active_learning_loop.py \
  --config configs/base_smoke.yaml \
  --mode smoke \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  --no-wait
```

## 5. 提交完整任务

```bash
python scripts/active_learning_loop.py \
  --config configs/base.yaml \
  --mode full \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  --no-wait
```

## 6. 监控与验收

```bash
qstat -u "$USER"
python scripts/inspect_al_results.py --run-dir runs/h2_ani_smoke --min-new-points 5
python scripts/inspect_al_results.py --run-dir runs/h2_ani_full --min-new-points 5
```
