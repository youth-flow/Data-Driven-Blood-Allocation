# 合成 AXID 实验

本目录负责完全人工的数据生成，不读取真实原表、真实派生数据或训练好的预测模型。
`configs/reference.yaml` 是可执行的参考机制；参数是实验设定，不是对医院真实分布的估计。
当前用于搭建和验证实验，不能把小规模检查当成模型性能结论。

## 数据与决策

- `A`：目标日早上已知的申请集合，每条申请包含龄期、用血量、备血量、类型、科室、临床指标和已填写的计划日期。
- `X`：日历、已知计划业务强度、已知急诊压力，以及从 A 计算的基础和临床汇总。
- `I`：目标日期初库存；给定已知环境后，其噪声与需求冲击独立。
- `D`：可见申请当天实现的需求，加当天新增申请需求。
- 单一商品为 O 型红细胞。决策始终为 `q = max(Q_tau + target - I, 0)`，不展开患者分配或跨日库存递推。

验证集起点固定为人工日期 2020-01-01；训练集向前取指定天数，测试集紧接验证集。日期只作为索引和日历变量，不表示真实覆盖年份。
各日独立抽取外生随机量；不同场景使用相同种子时，A、X、I 保持相同，仅需求机制改变。

## 生成顺序与参数

1. 生成已知环境 `scheduled_load`、`emergency_pressure`，均独立服从 Uniform(-1,1)。
2. 可见申请数 `N ~ Binomial(32, sigmoid(-0.2+0.4*scheduled_load))`；另有 5% 概率为空集合。
3. 每条申请类型概率为急救 0.15、手术备血 0.30、病房用血 0.30、病房备血 0.15、其他 0.10；科室为 6 个纯人工类别，龄期为 1/2/3 天。
4. 先生成 `r ~ Beta(2,3)` 以构造相关的申请特征：申请用血量 `0.5 Binomial(12,0.25+0.25r)`，备血量 `0.5 Binomial(16,0.15+0.20r)`；不规定二者大小关系。血红蛋白由 `160-100r+Normal(0,8)` 截于 [40,180]；压积为 `0.003*Hb+Normal(0,0.02)` 截于 [0.1,0.55]。
5. 计划日期偏移为 -1/0/1/2，概率 0.1/0.5/0.3/0.1；15% 缺失。它在需求生成之前已经可见，是人为定义的计划信息，绝不是事后输用日期。
6. `I = 315 + 20*scheduled_load + Uniform(-70,70)`。I 与 target 分开定义，不随测试 target 调整。
7. 最后独立生成未来冲击和订单结果，不将这些随机量放入输入。

设 `s_i = clip((160-Hb_i)/100,0,1)`，`peer_i` 为同科室其他申请的平均 s，没有同科室其他申请时为零。
订单发生概率为 `sigmoid(f_i + alpha*g_i + gamma*h_i + rho*U)`，其中：

```text
f_i = -2 + 0.35 log(1+申请用血量) + 0.15 log(1+备血量)
      + 1.2 s_i + 0.5 急救标志 - 0.25 delta + 0.4 已知计划当天标志
g_i = 0.8 scheduled_load * 手术备血标志 + 0.6 emergency_pressure * 急救标志
h_i = peer_i * s_i
U ~ Normal(0,1)，同一天所有订单共享
```

发生后的数量为 `V_i = 0.5*(1+Binomial(5,p_i))`，
`p_i = sigmoid(-0.5+0.3 log(1+申请用血量)+0.15 log(1+备血量)+0.7 s_i)`。
未发生时贡献为零，`Y_i = B_i*V_i`。

新增申请数为 `M ~ Binomial(12, sigmoid(-0.7+0.4 emergency_pressure+0.3 scheduled_load+rho U))`。
新增单笔数量也是半单位网格，其 Binomial 概率为 `sigmoid(-0.4+0.5 emergency_pressure+0.2 rho U)`。
最终 `D = sum(Y_i) + D_new`。新增申请明细不进入 A；训练期订单结果仅供需要订单监督的方法使用。
默认机制 D 的支持上界为 `3*(32+12)=132`，不对实现的 D 进行截尾；与当前 DAT/DeepQR 的 150 输出上界相容。
调整集合上限后须同时检查预测器输出范围，不能按某个模型的胜负删改标签。

## 四个预先定义的场景

| 场景 | alpha | gamma | rho | 作用 |
| --- | --- | --- | --- | --- |
| S1 | 0 | 0 | 0 | 条件独立订单 |
| S2 | 1 | 0 | 0 | 日级环境与申请类型交互 |
| S3 | 1 | 1.5 | 0 | 加入集合构成作用 |
| S4 | 1 | 1.5 | 0.8 | 加入共同冲击与条件相关性 |

这些是人工统计机制，不声称已经识别真实临床因果关系，也不预设哪种模型必胜。
当前数据没有跨日漂移；Rolling-SAA 可作为参照，但这里不检验其应对漂移的优势。
没有真实 q_hist_receipt，不伪造 Historical Receipt。

## 划分与输出

主实验开发训练 2000 日、验证 500 日、测试 3000 日。选定超参数/轮数后，所有需拟合的方法从头在训练+验证共 2500 日上重训，再测试。正式报告同时给出开发训练量和最终拟合量；不能将最终模型称为只见过 2000 日。数据 seed 与模型训练 seed 独立。
生成器第 2 版通过 SeedSequence 为三个样本集合分配独立随机流，再按样本日期分配输入和需求流。相同数据 seed 下只增加训练天数时，验证/测试样本完全不变，原训练样本是新训练样本的子集。

- `validation/`：训练数据 + 验证数据；所有标准化和类别编码只用训练部分拟合。
- `test/`：原训练 + 原验证组成重训数据，另外评估独立测试数据；转换在重训部分重新拟合。
- 测试轮数和超参数必须来自开发阶段冻结的 recipe，测试标签不参与选择。
- 输出包括订单级结果，仅供 Flat-ETO 的订单监督使用；D、D_new、订单结果不进入预测特征。
- 三类实验统一使用普通 DAT，全参数日级端到端训练；DAT 不使用订单级监督、预训练或微调。
- 输出文件使用 `synthetic_axid_v1` 数据合同；与真实预处理的 `morning_t_v1` 分开标识。
- 生成内容、权重和结果均不进入 Git。`generated/` 已忽略，源码白名单也拒绝该目录。

```powershell
python synthetic/scripts/generate.py --scenario S2 --output synthetic/generated/reference
python scripts/run_synthetic_experiments.py --group all --name synthetic_dev --seeds 0 1 2 3 4
python scripts/run_synthetic_experiments.py --group all --fold test --name synthetic_test --seeds 0 1 2 3 4 --recipe-root outputs/synthetic_dev
```

快速接口检查分别使用新输出目录，避免覆盖：

```powershell
python synthetic/scripts/generate.py --smoke --output synthetic/generated/check
python scripts/run_synthetic_experiments.py --data-root synthetic/generated/check --group all --name synthetic_check --smoke
```

`--smoke` 配方不能用于正式测试。`--resume` 只跳过内容校验通过的已完成运行；不支持训练轮内恢复，未完成运行需换新 suite 名称。
默认只有 seed 0；正式多 seed 运行必须显式传 `--seeds`。每个模型/变体独立保存配置、配方、预测器和指标；合并指标为 `metrics_long.csv`。

## Oracle 与可解释性边界

`oracle_quantiles()` 固定完整的 A、X，重新抽取未来冲击及订单结果，给出条件分位数的 Monte Carlo 近似。
它不读取当天实际 D、订单实际结果或实现后的共同冲击，不能称为知道未来实现值的完美信息解。
这是完整信息 oracle；信息消融各等级的专属 oracle 需要额外积分掉被遮蔽的信息，当前没有实现。
这些函数便于后续分析，当前总入口不会自动将 oracle 作为训练模型加入比较。

## 样本量与重复实验的最终安排

完整登记见 `configs/study.yaml`，该文件记录实验安排，不是自动启动大规模训练的入口。

- 主实验 S1–S4：2000 开发训练日、500 验证日、2500 最终重训日、3000 独立测试日。
- 学习曲线 S2/S3：开发训练量 500/1000/2000/5000，对应最终拟合量 1000/1500/2500/5500；验证固定 500、测试固定 3000。绘图明确使用哪一种样本量，建议横轴用最终拟合量。
- 每个配置 10 个独立数据 seed；每组神经模型和 wSAA 使用 3 个训练 seed。先在同一数据重复内平均训练 seed，再评价跨数据重复的不确定性。确定性方法每组运行一次。不能把同一数据集上的不同初始化当作独立数据重复。
- 所有方法在同一重复内共享完全相同的测试日；不逐模型重新抽取测试集，也不按测试结果更改生成器。
- 3000 是独立单周期场景数，不是模拟连续运营 8 年。大测试集降低评价的 Monte Carlo 误差，不能弥补训练样本不足。
- 在近似校准且样本独立的连续情形，0.95 覆盖率用 500 样本估计的标准误约 0.975 个百分点，3000 样本约 0.398 个百分点；期望越界数分别约 25 和 150。当前离散需求可能有保守覆盖，实际误差应按实测覆盖率估计。
- 成本差和 P 的精度须从配对损失与共同分母估计，不能把上述覆盖率误差直接当成 P 的误差。报告测试抽样误差与训练/数据重复波动。
- 先固定 3000 测试场景；若需要更高精度，另生成预先规定的独立评估批次，不用测试表现反复调参。
- 各场景输出均值、标准差、0.50/0.90/0.95 分位数、新增需求占比。S1–S4改变需求机制时也可能改变边际均值/方差，不能将场景间所有差异解释为单一结构的因果贡献。
- I 的噪声条件独立于 D；因此本机制下 I 主要改变配送决策，不预设库存输入一定提高需求预测。

学习曲线的生成示例（同一数据 seed，验证/测试保持一致）：

```powershell
python synthetic/scripts/generate.py --scenario S2 --data-seed 20261102 --n-train 500 --output synthetic/generated/S2_n500
python synthetic/scripts/generate.py --scenario S2 --data-seed 20261102 --n-train 2000 --output synthetic/generated/S2_n2000
```

统计依据：Morris, White & Crowther (2019), *Using simulation studies to evaluate statistical methods*, DOI 10.1002/sim.8086。样本数应服务于评价精度，而不必遵循固定训练/测试百分比。上述具体规模是本项目的预设协议，并非文献给出的充分样本量保证。
