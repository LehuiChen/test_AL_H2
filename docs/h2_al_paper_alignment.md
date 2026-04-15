# 教程流程与论文方法对齐说明

本说明将 H2 教程复现流程映射到论文中的主动学习闭环：

- Hou et al., *J. Chem. Theory Comput.* 2024, 20, 7744-7754
- DOI: `10.1021/acs.jctc.4c00821`

## 闭环步骤映射

1. 采样（Sampling）
- 教程复现：以平衡结构和频率为起点，做 Wigner 初始条件采样（`inputs/h2_freq.json`）。
- 论文思想：物理启发采样，优先探索化学相关区域。

2. 不确定性评估（UQ）
- 当前实现：保留主/副模型（committee）配置，按模型分歧触发标注；阈值行为由 `uncertainty.*` 与 `max_excess/min_excess` 控制。
- 论文思想：基于不确定性选择高价值样本，减少昂贵量化标注次数。

3. 触发标注（Query + Label）
- 当前实现：不确定性高的点交给参考方法标注，默认 `B3LYP/6-31G* + Gaussian(g16)`。
- 论文思想：只在低置信区域调用高精度电子结构计算。

4. 增量重训（Incremental Retraining）
- 当前实现：新增标注并入训练后重训 ANI 模型，迭代历史写入 `al_info.json` 和派生汇总文件。
- 论文思想：通过信息增量数据逐轮提升势能面模型质量。

5. 收敛停止（Convergence）
- 当前实现：新增点数量降到阈值以下（`new_points < min_new_points`）停止。
- 论文思想：当大部分采样点置信度足够高时停止迭代。

## 验收信号

- `al_info.json` 存在且可解析。
- `status.json.success=true`（或本地模式流程正常结束）。
- 新增点数量随轮次整体下降并触发停止条件。
