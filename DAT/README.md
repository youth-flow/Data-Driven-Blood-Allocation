# DAT

主架构与训练在 `src/datransformer/model.py`、`trainer.py`。仅使用日级样本直接端到端训练所有参数。
不导入其他模型包。现有真实数据配置为 `configs/main.yaml`，合成独立运行配置为 `configs/synthetic.yaml`。

## 结构与损失消融

由 `src/datransformer/experiments.py` 实现，列表在 `configs/experiments.yaml`：

| variant | 改动 |
| --- | --- |
| full | 原 DAT |
| no_self_attention | 取消申请间自注意力 |
| mean_pool | 交叉注意力换为掩码均值池化 + 线性映射 |
| sum_pool | 交叉注意力换为掩码求和池化 + 线性映射 |
| no_orders | 关闭逐条集合输入，保留日级汇总 |
| no_inventory | 取消预测器库存输入；决策公式仍使用真实 I |
| pinball | 精确 pinball 损失 |
| softplus_beta6 | softplus beta 从 2 改为 6 |
| no_planned_use | 屏蔽计划日期及声明依赖它的日级汇总 |

池化替换保留残差、LayerNorm 和前馈部分。每个实验重新训练；不把测试时损坏输入当作重训消融。

## 固定架构的信息阶梯

| information | 可用信息 |
| --- | --- |
| L0 | 日级外生环境 + I |
| L1 | L0 + 集合数量、申请量、备血量和龄期汇总 |
| L2 | L1 + 临床汇总 |
| L3 | L2 + 逐条申请基础数值、类型与科室 |
| L4 / full | L3 + 逐条临床指标、计划日期 |

所有等级具有相同参数槽位；标准化后屏蔽未开放字段。L0–L2 将整个集合分支替换成一个全掩码占位，不能泄漏集合大小。
特征来源在合成 schema 的 `information_groups` 声明，缺失声明的真实旧版数据不允许直接运行该阶梯。
不改变 I 的决策含义。no_orders 不等于没有申请信息，因为 X 仍含申请汇总。
结构消融和信息消融不叠加。不使用订单级辅助训练、预训练、微调或冻结。

```powershell
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --variant sum_pool --name dat_sum
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --information L0 --name dat_L0
```

`diagnostics.inventory_sensitivity()` 提供固定其他输入的库存有限差分聚合统计，默认扰动 10 单位；当前总入口不自动运行此额外诊断。
