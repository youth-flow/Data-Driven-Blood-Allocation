# 04｜AXID-v4 性能实验正式候选协议：P1 八模型、P2 八结构、P3 四信息

> **DGP 唯一来源**：`02_AXID_v4_DGP_Design.md` + 与之同版的 `AXID_v4_reference.py`。**本文件只调整 v3 的三处实验编排**：P1 移除 Rolling-SAA；P2 增加 `no_self_attention_param_matched`；P3 移除 L3。其余训练预算、数据规模、核心架构、主要基线算法及评价逻辑原则上与 v3 一致。  
> **状态**：待工程移植与冻结的 AXID-v4 **新实验协议**；不得覆盖、改写或选择性删除旧 AXID-v3 已完成的 9/7/5 正式结果。  
> **主要指标优先级**：**① PrescriptiveP（主指标）**；**② MeanCost、SL_G（辅助）**。`τ=0.9`；`G∈{260,280,300,320,340}`；`G=300` 为主要展示，五 G 综合 P 也作为主要报告口径。

## 1. 研究目标与变化矩阵

| 实验 | v3 | **v4** | 调整理由 |
|---|---:|---:|---|
| P1 方法比较 | 9 | **8** | 删除 Rolling-SAA：i.i.d. 数据中“最近窗口”没有特殊统计优势；旧结果仍保留附录或历史版本 |
| P2 结构消融 | 7 | **8** | 新增 **No-SA 参数匹配组**，区分移除 SA 与参数量变化 |
| P3 信息共享 | 5 | **4** | 删除 L3 人为遮蔽状态，仅保留实际供应链可解释的 L0/L1/L2/L4 |

研究预期：v4 同时包含患者自身周围的局部病例组合效应 `T_i(A)` 与日级状态条件化患者权重 `ω_i(A,X)`。这是对 **SA→CA 两段归纳偏置**的统计学机制检验；**不保证 Full DAT 一定胜出**，也不能利用正式测试排名反选 DGP。

## 2. 数据、模型及训练预算（固定）

| 项目 | 正式设定 |
|---|---|
| 单一 DGP | AXID-v4；每日 `A[N,4]`、`X[9]`、`3≤N≤8`、`D∈[0,60.5]` |
| 独立重复 | **10** 次；每重复重新生成 **2,500** 训练日和 **3,000** 独立测试日 |
| 随机模型 | 每数据重复 **3 Seed**，先对 Seed 求平均，再按 10 个独立重复统计 |
| 目标分位数 | `τ=0.9`；均使用 Pinball 导出的同一补货策略 |
| 库存目标 | `G={260,280,300,320,340}`，改变 G 不重新训练分位数模型 |
| 训练配置 | 每种算法仅一套**事前冻结**的超参数；无测试驱动搜索、无最佳 Seed 选择、无正式验证集候选筛选 |
| 数据隔离 | v4 与 v3/S4 不共享正式测试日；所有 v4 方法在同重复上共享测试日和实现需求以形成配对比较 |

### 2.1 固定 DAT 拓扑与输入维数

`Patient MLP(4→d)` → `one Pre-LN Self-Attention + residual` → `Day Linear(9→d)` → `one single-query Cross-Attention (day query + patient memory, Pre-LN) + residual` → `Day FFN + residual` → `linear scalar readout + Clip[0,60.5]`。

- 恰好三处 LayerNorm（SA 输入、CA Query、CA Memory）；无位置编码、患者 FFN、Day FFN 前额外 LN 或 Final LN；Mask 严格遮盖 Padding。
- 候选性能配置 `d_model=80`，四个 Self/四个 Cross 头，`patient_hidden=96`、`day_ffn_hidden=128`（**只有在正式运行前再次核对实现并冻结后才算唯一配置**）；改动隐藏宽度不改变 DAT 模块拓扑。
- 性能 `M=8,d_a=4,X=9`，原逼近充分潜维核算为 `d_m=min(C(12,8)-1,2·8·4+1)=65`、`d_eff=73`，故 `d≥74`；取 80 满足该维度充分条件。若转入 T1–T3，需独立更新参数计数和具体理论假设。

### 2.2 传统摘要特征（不由机制泄漏）

为保留 v3 可比性，传统条件方法统一使用事前构造的 **23 维**：

\[
\boxed{\psi(A,X)=[X,\overline{u^2},\overline{v^2},\overline{uv},\overline{ur},\overline{vr},\overline{us},\overline{vs},\overline{rs},\min u,\max u,\min v,\max v,\min r,\max r].}
\]

所有输入只由事前 `A,X` 计算；**不得把 v4 内部的 `k_ij,T_i,H_ctx,H_local,ω_i,ℓ_i,π_i` 填入任何模型或人为摘要**。Flat-ETO 的单患者输入为 `[a_i,ψ]`，即 **27 维**。DeepQR、Set-ETO 和 DAT 都直接获得完整 `A,X`。

## 3. 统一运营策略与评价公式

对于 `f_m(Z)`（`0≤f≤60.5`），实施：

\[
 q_{m,G}=[f_m(Z)+G-I]_+,\qquad I_{\rm end}=I+q_{m,G}-D.
\]

实现需求 `D` 下的单日成本：

\[
 c_{m,G}=0.9[G-I_{\rm end}]_++0.1[I_{\rm end}-G]_+.
\]

**主指标：Oracle-normalized Prescriptiveness（越大越好）**

\[
\boxed{\mathcal P_{m,G}=
\frac{C_G(\mathrm{SAA})-C_G(m)}{C_G(\mathrm{SAA})-C_G(\mathrm{Full\ Oracle})}.}
\]

五 G 综合必须**先汇总成本再计算一次比值**：

\[
\boxed{\mathcal P_{m,5G}=
\frac{\sum_{G\in\mathcal G}(\overline C_{\mathrm{SAA},G}-\overline C_{m,G})}
{\sum_{G\in\mathcal G}(\overline C_{\mathrm{SAA},G}-\overline C_{\mathrm{Oracle},G})}.}
\]

**辅助指标**：

\[
\boxed{\operatorname{MeanCost}_{m,G}=\frac1{3000}\sum_{t=1}^{3000}c_{m,G,t},\qquad
SL_{m,G}=\frac1{3000}\sum_t\mathbf1\{I_{{\rm end},t}\ge G\}.}
\]

- P 在同一个 G 下与 MeanCost 严格反向排序，但它额外说明该方法实现了多少 **SAA→完整信息 Oracle** 的潜在运营改善。
- `SL_G` 是目标库存达标率，**不设“越高越好”的不计成本排名原则**：过高服务率可能来自持有更多超目标库存，增加成本；主文围绕 P 解读 MeanCost/SL 的权衡。
- Oracle 是条件于当前决策前 `A,X` 的 Bayes 分位数，**不能读取当日实际 U、D 或新增订单数 L**。计算得到的 Oracle 补货量在与所有模型相同的 3,000 个测试日 `D_t` 上实现评价。
- 30,000 个测试日的 Oracle PMF 只计算一次；同一天得到的 Oracle `Q*` 可对五个 G 复用，不能重算成不同的 G-specific Oracle 预测器。

## 4. P1：**八种方法**，主要比较信息价值实现率

| 方法 | 事前输入 | 监督标签 | 预测构造与训练目标 |
|---|---|---|---|
| **DAT** | `A[N,4], X[9], Mask` | 日级 D | 单 SA + 动态 CA + Day FFN；直接 Pinball(0.9) |
| **SAA** | 仅训练日的 D；测试仍用 I,G 决策 | 日级 D | 无条件经验 0.9 分位数，定义 P=0 |
| **wSAA** | `ψ[23]` | 日级 D | 随机森林叶邻近权重 + 加权经验 0.9 分位数，作为 predictive-to-prescriptive 文献框架的代表实例 |
| **Flat-ETO** | 单患者 `[a_i,ψ][27]`，新增需求模型用 `ψ[23]` | D，且训练时额外 `Y_i` | 七类患者 PMF + 剩余需求条件经验 CDF；按固定独立边际卷积构造总需求分位数 |
| **Set-ETO** | `A,X` | 日级 D | DeepSets mean pooling + 固定 HurdleGamma 条件 NLL，导出 0.9 分位数 |
| **LQR** | `ψ[23]` | 日级 D | 固定正则化线性 Pinball(0.9) |
| **GBQR** | `ψ[23]` | 日级 D | 固定量化梯度提升树 Pinball(0.9) |
| **DeepQR** | `A,X` | 日级 D | DeepSets mean pooling + 直接 Pinball(0.9) |

**Rolling-SAA：从新 P1 主表删除。** 原因是生成器每天 i.i.d.、没有趋势和时序移位，滚动窗口不利用额外事前条件信号；但 v3 已完成的 Rolling 结果不得删除，并可放在附录或真实医院非平稳数据评价中讨论。

公平性限定：Flat-ETO 获得额外患者级监督；wSAA/LQR/GBQR 的 `ψ` 不是完整患者集合的无损编码；Set-ETO 的 Gamma 分布在 v4 下仍可能误设，且 NLL 与运营目标可能失配。**这些是不同估计范式的比较，不能把所有 P1 差异都单独归因于某个注意力模块。**

## 5. P2：**八个**结构条件及识别解释

| 变体 ID | 唯一结构变化 | 主要比较意义 |
|---|---|---|
| `full` | 完整 DAT | 同时存在患者上下文化 SA 与日级动态 CA |
| `no_self_attention` | 删除 SA 分支，保留动态 CA、X、FFN | 观察移除 SA 的综合效应（参数也减少） |
| **`no_self_attention_param_matched`** | 删除 SA，但通过 Day FFN 宽度补足有效参数量 | 控制一阶参数预算差异，排查“Full 较差只因参数更多”的解释 |
| `no_day_query` | CA 的日级 Query 换成**可学习常数 Query** | X 仍经 Day Linear/残差/Day FFN 保留，仅取消聚合阶段的日级动态选择 |
| `mean_pool` | 将 CA 换成有效患者表示 Mean Pool + 可学习投影；保留 SA | 对照静态均值聚合 |
| `mean_pool_param_matched` | Mean Pool，同时补足模型参数量 | 排除其弱势仅来自参数较少 |
| `sum_pool` | CA 换成 Sum Pool + 可学习投影；保留 SA | 对照静态求和聚合 |
| `sum_pool_param_matched` | Sum Pool，同时补足参数量 | 排除其弱势仅来自参数较少 |

**参数匹配合同（与预测表现无关）**：

- 统一报告每个模型**有效可训练参数量**，目标匹配到 Full 的差距不超过 **5%**，优先小于 1%；只根据参数计数设定宽度，禁止用任何数据的成本或 P 来挑宽度。
- v3 工程的已报告参数数目：`P_full=82,129`，`P_noSA=56,049`。若 v4 的 Full 与 No-SA 确实沿用这两个实际数目，且 `d_model=80`、原 Day FFN 隐层宽度为 **128**，则 Day FFN 每增宽 1 增加 `2d+1=161` 个参数。令 No-SA matched 的 Day FFN 宽度从 128 改为 **290**，可得到 `P=82,131`，只比 Full 多 **2** 个参数。这是**条件性工程候选公式**，必须在实际模型构建后重新数参数；若工程实现不一致，则按参数数目重新计算，不能直接宣称匹配。
- `mean_pool_param_matched` 与 `sum_pool_param_matched` 继续沿用 v3 仅基于参数数目确定的匹配策略，重新验算模型参数量。
- 参数匹配**不能**保证训练优化难度、函数类、计算图完全匹配；它只是减少一个重要混杂因素。

主要机制假说：患者关系 `T_i(A)` 先进入个体需求，日级 `ω_i(A,X)` 再影响被重视的病例组合，故 Full 的 SA→CA 有**合理的归纳偏置优势可能性**。但 No-SA 和 DeepSets 仍可学习这些关系；尤其不允许将 Full 在本版必须胜出作为 DGP 验收标准。

## 6. P3：**四个**现实信息共享等级（删除 L3）

| 信息权限 | 血液中心获得什么 | DAT 患者输入 | 现实解释 |
|---|---|---|---|
| **L0** | `c1,c2,I` | 固定 Dummy Token + 固定 Mask | 只有可公开获得的区域运营压力代理和库存，无患者数据 |
| **L1** | L0 + `N/8,Σu/8,Σv/8` | 同 L0 | 医院仅提供患者人数、总申请量、总备血量 |
| **L2** | L1 + `mean(r),mean(r²),mean(s)` | 同 L0 | 医院提供病例组合风险和术式统计摘要 |
| **L4** | 完整 `X[9]` + 完整逐患者 `A[N,4]` | 真实有效患者 Token + Mask | 医院—血液中心通过共享信息系统提供患者级细节 |

- L0/L1/L2 不能让模型从真实 `N` 个 Token、Padding 长度或 Mask 还原患者人数；统一使用固定 Dummy Token 与固定有效 Mask。未授权的 X 坐标用固定无信息常数屏蔽，**Day Encoder 不能自行重新拼接 N**。
- L4 可以读取完整患者记录，但 `k,T,ω,H_ctx,H_local` 仍是不可见的真实生成机制，**不是**共享系统中的字段。
- 四组 DAT **网络拓扑和已分配参数数目一致**，分别重新初始化/训练；没有 L3 组，不用把 L4 重编号成 L3，便于与既有 v3 报告明确对应。
- 理论信息集合满足 `𝓘_L0⊂𝓘_L1⊂𝓘_L2⊂𝓘_L4`，所以**各信息集自己的 Bayes 最优成本**必然非增；有限样本重新训练的 DAT 成本与 P **不要求严格单调**。
- 主文着重对比 L1→L2（临床汇总价值）与 L2→L4（逐患者风险／订单／术式对应与局部病例结构的**实际学习增量**）。要称为“精确 Bayes 信息价值”，须另计算各信息等级自己的条件 Oracle；本主实验只有统一的 Full Oracle。

## 7. Oracle 后处理与 P 的统计合同

1. 对每个测试日，Oracle 在 25 节点正态 GH 下，**先固定 U 卷积 N+12 个七类 PMF，再积分 U**，取连续 `Q^*_{0.9}(A,X)`；禁止分别对患者边际积分后再卷积。
2. 对每个测试日仅计算一次 Oracle 分位数，复用到五个 G；同一天的 SAA、各方法、Oracle 使用**相同实际 D**计算实现成本和 `SL_G`。
3. 每个数据重复 `r`、方法 `m`、G，先平均 3 个随机 Seed 的 `MeanCost_{r,s,m,G}`；确定性方法只用一个真实 Seed，不复制成伪随机种子。Oracle 无训练 Seed。
4. 在各方法 **10 个独立数据重复平均成本**上计算主 `P_{m,G}` 和综合 `P_{m,5G}`。**不先对每重复 P 求均值**；当某 G 分母接近 0 时如实标注，不裁剪 P 至 `[0,1]`。
5. CI 使用**10 个独立重复为 Cluster** 的配对 Bootstrap：每次抽中的重复整体带走其五个 G、SAA、Oracle 与所有模型成本；每次重新计算“先汇总成本差再取比值”。所有相同的 Full DAT 在 P1/P2/P3 必须使用**相同的 Bootstrap 索引/主结果**，避免三表出现不同的 CI。
6. 补算 Oracle 需独立记录 16/25/32 节点的小样本数值差、PMF 概率和、分位数反演误差、患者置换不变性、条件相关性处理、测试日预测与需求实现的严格权限隔离。

## 8. 输出文件及主图建议

| 文件／图 | 应含信息 |
|---|---|
| `axid_v4_oracle_costs.csv` | 每重复、每 G 的 Oracle MeanCost 与 SL_G；Oracle 分位数另存供审计 |
| `axid_v4_prescriptive_p.csv` | 每方法/消融/等级、每 G 与五 G 的 P 点估计、95% CI、分母 |
| `axid_v4_p1.csv` / P1 主图 | **8 方法以 P 为中心排序**，列出 MeanCost/SL_G；wSAA 和 DeepQR 为重点基准 |
| `axid_v4_p2.csv` / P2 主图 | **8 结构**与 Full 的**配对 P/成本差**、实际参数数目；重点 Full–NoSA–NoSA-matched 与 Full–NoDayQuery |
| `axid_v4_p3.csv` / P3 主图 | **L0、L1、L2、L4 四级**的信息阶梯，以 P 为主，MeanCost/SL_G 辅助 |
| `axid_v4_manifest.json` | DGP 版本与 SHA、训练与测试种子、模型配置、参数量、生成器/Oracle 校验摘要 |

## 9. 冻结前必须核查（Checklist）

- [ ] v4 生成器数学表达、参考代码、工程生成器逐项一致；不遗留 v3 全局 `0.25 H_peer` 对所有患者的统一修正。
- [ ] 数据维度 `4/9/23/27`、库存、五个 G、需求 `B=60.5` 与所有模型 Clip 对齐。
- [ ] 新的局部核包含 `(u,v,s)`，排除自环；`T_i` 直接进入每名患者 `ℓ_i,π_i`，`H_local` 进入 `p_new`；保持真实患者顺序置换不变。
- [ ] Oracle 预检查的 SAA→Oracle 分母非微小；患者局部机制对条件分位数及期望成本确有可测响应；没有大量发生概率饱和。
- [ ] No-SA matched 的**实际参数量**已由工程确认，宽度不是按测试指标调出来的。
- [ ] P1 仅 8 组、P2 仅 8 组、P3 仅 L0/L1/L2/L4；所有旧 v3 报告完整保留。
- [ ] 分位数训练与测试都不访问 `U,T,H_ctx,H_local,ω,k,p_true,Q^*` 等生成器内部真值；Flat-ETO 的 `Y_i` 仅作训练监督。
- [ ] DGP、模型配置与正式测试 Seed 在运行前冻结；不因 v4 的 DAT/No-SA/P1/P3 排名再调整本版生成参数。

> **历史/新实验界线**：AXID-v4 明确是 *post-v3 hypothesis refinement*。任何新的机制和超参数变动都应该作为下一版，而不是覆盖已有 v4 正式结果。对于真实医院数据，不可以假设这些合成机制必然存在，需要独立验证。
