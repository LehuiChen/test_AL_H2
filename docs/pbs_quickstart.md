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
