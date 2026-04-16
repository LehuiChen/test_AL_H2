# H2 教程与论文对齐

## 核心映射

当前仓库把 H2 最小案例放进了参考 ADL 仓库的同构骨架里。阶段映射如下：

1. `prepare_h2_seed`
   作用：把 `inputs/h2.xyz + inputs/h2_freq.json` 整理成统一 seed。
2. `sample_round_000_initial_conditions`
   作用：从 H2 seed 采样首轮初始条件。
3. `labels_round_000`
   作用：用 Gaussian `B3LYP/6-31G*` 做目标标注。
4. `build_training_dataset`
   作用：只用 Gaussian 标注构造直接学习数据集。
5. `train_main_model` + `train_aux_model`
   作用：训练主模型和副模型，副模型只学能量，用于分歧不确定性。
6. `run_md_sampling_round_*`
   作用：用主/副模型进行 MD 采样，在首个不确定点停止轨迹。
7. `select_round_*_frames`
   作用：按主副模型分歧筛出下一轮需要补标的点。
8. `active_learning_loop`
   作用：串起所有轮次，直到 `converged` 或 `selected_count < min_new_points`。

## 和原论文/教程的关系

教程里的 H2 案例强调的是一个最小 AL 闭环：

- 采样
- 不确定性评估
- 触发标注
- 增量重训
- 收敛停止

论文把同样的思想推广到更复杂体系，并量化了加速收益。当前仓库保留了这条方法链，只把参考体系缩到 H2，并把参考 ADL 仓库中的直接学习语义落实到这套 PBS 多阶段框架里。

## 当前不确定性定义

这里不再依赖差值学习，而是：

- 主模型：学习 energy + force
- 副模型：学习 energy
- 不确定性：`|E_main - E_aux|`

这和教程示例里的“主/副模型分歧触发标注”是一致的，只是具体实现落在参考 ADL 的多阶段 PBS 框架里。
