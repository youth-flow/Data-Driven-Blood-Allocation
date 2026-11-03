# 简化 DAT

唯一生产实现是 `src/datransformer/model.py`，对应本机理论 PDF 的式 (6)–(7)；直接以日级原始 pinball 训练。旧多分位数网络、平滑损失、预训练、三步走、冻结与 LoRA 均不提供训练入口。旧权重格式不兼容，不自动转换。

## 架构

设订单编码为 `E0 = patient_MLP(A)`，日级编码为 `u = day_affine(X)`：

```text
E = E0 + self_attention(LN_sa(E0), LN_sa(E0), LN_sa(E0))
v = u  + cross_attention(LN_q(u), LN_m(E), LN_m(E))
u_plus = v + day_FFN(v)
Q_tau = clip(linear(u_plus), 0, B)
```

订单侧没有 FFN；日级 FFN 前没有额外 LN；输出头前没有 LN。全网恰好三个 LN，epsilon=1e-5，两个注意力各一层，无位置编码、dropout=0。订单 MLP 与日级 FFN 都是两层仿射加中间 ReLU；日级编码为仿射。两种注意力的头数独立配置，默认均为4。

每个 τ 训练独立的标量网络。五个 τ 为0.5、0.6、0.7、0.8、0.9；不共享权重、不排序纠正，报告交叉比例。target 只进入决策和评价。

| 人工数据配置 | 潜维 | 订单 MLP 隐藏宽 | 日级 FFN 隐藏宽 | 每个 τ 的参数数目 |
| --- | ---: | ---: | ---: | ---: |
| 理论基准 M=2 | 20 | 32 | 32 | 5,801 |
| 性能主实验 M=8 | 64 | 128 | 256 | 76,161 |

输入为3维订单、7维日级坐标及独立库存坐标，库存进入网络一次。理论充分条件为 `d_model >= max(d_m+2, d_m+d_x+1)`，这里所需下限分别17和57。
`synthetic/scripts/check_protocol.py` 用实际模块复核参数量、置换不变性、输出范围和全局参数球投影。

## 只用于本次实验的结构干预

下列选项使用同一份简化 DAT 代码，每次从头训练，不保留另一套 DAT 主模型：

| variant | 改动 |
| --- | --- |
| full | 简化 DAT |
| no_self_attention | 跳过集合自注意力及其 LN |
| mean_pool | 以掩码均值代替交叉注意力权重，保留 V/O 映射 |
| sum_pool | 以掩码求和代替交叉注意力权重，保留 V/O 映射 |
| no_day_query | 采用可训练常数 query；日级残差 u 保留 |

均值/求和中 Q/K 及 query LN 不参与计算；同时报告分配参数和结构有效参数。总实验另有 `mean_pool_matched`、`sum_pool_matched`，只扩大日级 FFN，使有效参数量与 full 相差不超过5%。这两个名称是同一干预的容量设置。

## 固定架构的信息阶梯

| 等级 | 开放信息 |
| --- | --- |
| L0 | 两个日级情境及 I |
| L1 | L0 + 人数、申请量总和、备血量总和 |
| L2 | L1 + 风险均值、二阶矩 |
| L3 | L2 + 逐订单申请量与备血量 |
| L4 | 完整 A、X、I，包括逐订单风险 |

所有等级保留同一64维结构与分配参数数目；L0–L2使用一个全屏蔽占位 token，避免暴露人数。未开放字段在固定变换后置零。结构与信息干预不能叠加。库存始终参与运营决策。

## 使用

```powershell
python scripts/run_synthetic_experiments.py --family performance --group ablation --name dat_ablation
python scripts/run_synthetic_experiments.py --family performance --group information --name dat_information
```

独立模型入口需要先生成纯人工数据：

```powershell
python synthetic/scripts/generate.py --scenario S2 --output synthetic/generated/reference
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --variant mean_pool --name dat_mean
```

性能轨道按验证 pinball 选择轮数；冻结后以训练+验证从头重训，不读测试标签选参。理论轨道采用固定预算、原训练目标最小 checkpoint，详见 [六组协议](../docs/synthetic_protocol.md)。
