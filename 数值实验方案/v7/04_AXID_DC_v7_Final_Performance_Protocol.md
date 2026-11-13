# 04｜AXID-DC v7-Final：八模型、八结构与四级信息共享性能实验协议

> **唯一本版 DGP**：[`02_AXID_DC_v7_Final_DGP.md`](02_AXID_DC_v7_Final_DGP.md)，版本 `axid_dc_v7_continuous_final`。**区别于已经完成正式运行的旧离散 v7**；本版的患者连续特征、今日患者边际化和日级匹配机制均已改变。原 v7 的已发布结果/Oracle **不得引用为本版的正式结果**。
>
> **核心原则**：保留固定 DAT 架构与三组性能实验框架；修正连续隐藏患者的 Full Oracle 算法；保留传统基线的 16 维 `ψ` 和朴素 Flat 的**两部分独立 MSE 监督**。主指标为 Oracle-normalized Prescriptiveness，MeanCost 和 `SL_G` 为辅助指标。

## 1. 规模、训练与统计口径

| 项目 | 唯一主协议 |
|---|---|
| 训练/测试 | 10 个独立 DGP 重复；每重复 2,500 个独立训练日 + 3,000 个独立测试日 |
| 随机种子 | 有随机训练过程的方法每重复 3 个固定 Seed；**先在重复内平均**，再跨 10 重复统计 |
| 分位数/目标库存 | `τ=0.9`，`G={260,280,300,320,340}`；**五 G 共用同一预测，不重新训练** |
| 方法比较 | P1：8 模型；P2：8 结构条件；P3：L0–L3 四级信息 |
| 主指标 | `PrescriptiveP`：基于血液中心事前 L3 Full Oracle 的成本差归一化 |
| 辅助指标 | `MeanCost`、目标库存达标率 `SL_G`；不以最高 `SL_G` 代替成本最优 |
| 训练规则 | 各方法只有一套**运行前冻结**的模型宽度、学习率、Epoch、Batch、正则等配置；不按测试集调参、挑种子或改 DGP |

**实验集合**：P1 为 DAT / SAA / wSAA / Flat / Set-ETO / LQR / GBQR / DeepQR；**没有 Rolling-SAA**。P2 为 Full / No-SA / No-SA-matched / No-Day-Query / Mean Pool / Mean Pool matched / Sum Pool / Sum Pool matched。P3 为 L0 / L1 / L2 / L3，**不设置 L4**。

## 2. 事前输入与模型边界

血液中心的完整事前信息是 `Z=(A⁻,X)`。输入患者仅为过去登记者：

\[
A^- = \{(u_i,r_i,s_i):i=1,\ldots,N^-\},\quad
X=\left(\frac{c_p+1}{2},\frac{c_e+1}{2},\frac{I-205}{130},
\frac{N^-}{8},\frac{\sum_{A^-}u_i}{8},\frac{\sum_{A^-}r_i}{8},\frac{\sum_{A^-}s_i}{8}\right)\in[0,1]^7.
\]

**不要将 `b_i` 作为 DAT 的第四个患者坐标**：所有可见患者 `b_i=0`，这个常数是医院端用于区分过去/今日患者的规则输入；今日 `b=1` 患者及其特征均不可提供给血液中心。

传统表格模型**保留用户选定的 16 维人工特征**：

\[
\boxed{\psi(A^-,X)=\left[X,\overline{ur},\overline{us},\overline{rs},
\min u,\max u,\min r,\max r,\min s,\max s\right]\in\mathbb R^{16}.}
\]

均值、极值、交叉矩只能从 `A⁻` 计算；不使用今日患者、真实需求或医生内部 `ζ,T,ω,h,U`。由于 `N⁻≥3`，均值/极值有定义。

### DAT 的固定架构（不因 DGP 改变）

```text
A- [batch, N_pad, 3] → Shared Patient 2-layer ReLU MLP → E0
E = E0 + SelfAttention(LN_sa(E0))                # patient mask
X [batch,7] → u = DayLinear(X)                   # single day token
v = u + CrossAttention(LN_q(u), LN_m(E))        # dynamic day query
u+ = v + DayFFN(v)                               # 2-layer ReLU, no pre-FFN LN
f(A-,X) = Clip(Linear(u+), 0, 61)                # no Final LN
```

固定 1 层 SA、1 层 CA、各自残差、**恰好 3 个独立 LayerNorm**、无患者位置编码/患者 FFN/Final LN。沿用现有性能合同的建议参数：`d_model=80`、SA/CA 各 4 头、`patient_hidden=96`、`day_ffn_hidden=128`，并由真实工程重新审计参数量。DAT 只用**日级需求标签** `D` 训练 Pinball(0.9)。在 P1 DAT、P2 Full、P3 L3 中**复用同一个训练实例和预测**。

## 3. P1：八种模型与公平性边界

| 模型 | 部署输入 | 唯一方法与目标 | 解释 |
|---|---|---|---|
| **DAT** | 完整 `A⁻,X` | SA → 动态 CA → Day FFN；日级 `D` 的 Pinball(0.9) | 主方法 |
| **SAA** | 不用当日条件特征；订货仍知道 `I,G` | 全训练日 `D` 的无条件经验 0.9 分位数 | P 的零点 |
| **wSAA** | `ψ∈R¹⁶` | 随机森林叶节点邻近权重；加权历史 `D` 的经验 0.9 分位数 | 经典条件加权 SAA 思路 |
| **Flat** | 单人 `a_i=(u,r,s)`；日级 `X` | **患者 MSE + 剩余需求 MSE**，两个预测量直接相加 | 朴素均值点预测，有额外患者训练监督 |
| **Set-ETO** | 完整 `A⁻,X` | DeepSets Mean Pool → HurdleGamma 参数分布，用 NLL → 0.9 分位数 | 参数分布可能误设 |
| **LQR** | `ψ` | 带固定正则的线性 Pinball(0.9) 回归 | 全局线性分位数 |
| **GBQR** | `ψ` | 固定参数的 Quantile Gradient Boosting，目标 0.9 | 非线性表格分位数 |
| **DeepQR** | 完整 `A⁻,X` | DeepSets Mean Pool + 融合 MLP；日级 `D` 的 Pinball(0.9) | 无 Attention 的集合分位数模型 |

**公平性说明**：DAT、DeepQR、Set-ETO 读取完整患者 Token；wSAA、LQR、GBQR 读取手工摘要 `ψ`，所以 P1 的方法差异同时包含表示差异。Flat 可以使用更丰富的**训练监督**，但其预测目标为均值而非 0.9 分位数。**只有 P2 的同输入、同 Pinball 结构消融可用于识别 Attention 模块作用。** 所有方法最终进入补货政策的标量仅在部署时限制到 `[0,61]`，并使用相同 `G` 和 `I`。

## 4. Flat：两部分独立监督、朴素相加（不校准分位数）

定义训练期可分解标签

\[
\boxed{R_t=D_t-\sum_{i\in A_t^-}Y^-_{ti}=\sum_{j\in A_t^+}Y^+_{tj}+J_t.}
\]

**患者部分**：共享的小型回归器 `g_θ:R³→R`，单独使用已登记患者的实际发血标签 `Y⁻_{ti}`，最小化

\[
\boxed{\min_\theta\sum_{t\in\mathrm{train}}\sum_{i\in A_t^-}\left(Y^-_{ti}-g_\theta(a_{ti})\right)^2.}
\]

**当天剩余需求部分**：另一小型回归器 `h_η:R⁷→R`，仅利用当天可见日级 `X_t`，以 `R_t` 作独立 MSE 监督：

\[
\boxed{\min_\eta\sum_{t\in\mathrm{train}}\left(R_t-h_\eta(X_t)\right)^2.}
\]

**最终预测**：

\[
\boxed{\widehat D_{\mathrm{Flat}}=
\operatorname{clip}_{[0,61]}\left(
\sum_{i\in A^-}\operatorname{clip}_{[0,6]}g_\theta(a_i)
+\operatorname{clip}_{[0,13]}h_\eta(X)
\right).}
\]

- 预测不拟合七档 PMF、没有分布卷积、**不预测条件分位数**、不使用 `ψ`、不添加 SAA 锚定或安全库存缓冲。
- 预测量是**两个朴素条件均值型回归分量之和**。由于 `g` 忽略当日背景和竞争，总和不一定严格等于 `E[D|A⁻,X]`；这正是 Flat 的简约性。
- `R` 标签只有训练期从 `D` 与患者发血记录计算。测试期不得查看真实 `Y⁻,R,A⁺,U`。Flat 的监督权限多于其他 P1 算法，必须披露。
- MSE 均值直接用于 `τ=0.9` 不对称库存成本，可能显著落后于 SAA；**不得为了抬高 P 而事后加分位数校准**。

## 5. P2：八种架构条件

| ID | 与 Full 的差异 | 用于判断 |
|---|---|---|
| `full` | 完整 DAT | 共同参照 |
| `no_self_attention` | 去掉 Self-Attention 分支，保留动态 CA | SA 观察增量 |
| `no_self_attention_param_matched` | 去 SA 后只调整非注意力层宽度，使参数量接近 Full | 排除总参数量差异 |
| `no_day_query` | 将 CA 的 Query 改为**可学习常数向量**，`X` 仍通过 Day Linear 残差/FFN 输入 | 动态日级 Query |
| `mean_pool` | 有效患者 Memory 均值 + 输出投影替代 CA，保留 SA | 静态均值聚合 |
| `mean_pool_param_matched` | Mean Pool 结构不变，调整其他隐藏宽度匹配参数量 | 容量控制 |
| `sum_pool` | 有效患者 Memory 求和 + 输出投影替代 CA，保留 SA | 静态求和聚合 |
| `sum_pool_param_matched` | Sum Pool 结构不变，调整其他隐藏宽度匹配参数量 | 容量控制 |

三组参数匹配只根据**实际有效参数数目**在正式训练前确定，目标与 Full 差异 `≤5%`（最好 `≤1%`）；若改 `d_model=80` 的 Day FFN 隐宽 `H`，该 FFN 参数为 `161H+80`。所有结构用相同的完整 `A⁻,X`、测试日、训练目标、预算与种子。消融结果不能仅凭各模型分别的 CI 推断配对差显著性。

## 6. P3：四种现实信息共享等级

| 信息级 | 允许的日级输入 | 允许的患者 Token | 对应业务情景 |
|---|---|---|---|
| **L0** | `c_p,c_e,I`（`X` 的第 1–3 维） | 1 个固定 Dummy Token + 固定 Mask | 没有医院患者数据 |
| **L1** | L0 + `N⁻,Σu_i`（`X` 前 5 维） | 同一固定 Dummy | 医院共享基础人数/预计需求汇总 |
| **L2** | L1 + `Σr_i,Σs_i`（完整 `X`） | 同一固定 Dummy | 医院共享风险、计划/急诊病例汇总 |
| **L3** | 完整 `X` | 真实过去患者 `A⁻` + 真正的 Mask | 医院共享完整逐患者画像及对应关系 |

L0–L2 **不准把真实患者个数通过 Token 数、Mask、排序或未授权 `X` 坐标泄漏**；未授权坐标采用固定无信息占位值，所有患者端只有一个有效固定 Dummy。L2 **不能**接收 `ψ` 的交叉矩与极值；`ψ` 只服务于 P1 的传统基准。所有级别均不知道今日患者 `A⁺`。各级 DAT 结构/分配预算一致、分别从头训练；L3 复用 Full DAT 结果。Bayes 成本随信息嵌套非增，但分别训练的网络测试成本不要求严格单调。

## 7. 连续隐藏患者的 Oracle 与 P 后处理

**Full Oracle**：

\[
\boxed{f^*(Z)=Q_{0.9}(D\mid A^-,X),\quad Z=(A^-,X).}
\]

**正确流程**：按 `Binomial(2,p_+(c_e))` 精确拆分今日患者人数 `n=0,1,2` → 对 `s_j∈{0,1}` 精确求和 → 对今日患者连续 `u_j,r_j` 的 Beta 分布使用随机化 Sobol/QMC 积分 → 在每个可能今日患者集合、同一个正态冲击 `U` 下重新计算**全部患者**的 `ζ,T,ω,h` 并联合卷积 `Y_i` 的七类概率 → 用 25 节点 GH 积分共同 `U` → 按今日患者组合加权 → `J∼Unif(0,1)` 后反演条件分位数。最大连续积分维数 **4**。**不可沿用离散 v7 的 45 种隐藏画像枚举，也不可提前对各患者的共享 `U` 独立积分。**

- 每独立重复，在**相同测试日**先生成/冻结全部真实 `D`；Oracle 的预测输入只有 `A⁻,X`，绝不能偷看该日真实今日患者或 `U,D`。
- 每测试日 Oracle **仅求一次** `f*(Z)`，总共 30,000 个测试状态，五个 `G` 和 P1/P2/P3 共用；Full DAT 模型在三组实验中也仅训练一组。
- 正式采用的 QMC 点数必须依据**独立数值精度检验后冻结**，以 `2^m` 为基本点数；小样本对照例如 128/256/512 或更高点数及不同 scramble，并检查 16/25/32 GH 结果。不要声称固定 256 点天然足够。优化只限**计算精度和向量化**，不得按 DAT 排名修改生成参数。
- `p_k(Z)=Pr(D_0=k|Z)` 的 `k=0,…,60`，由 `J` 的区间均匀性反演 0.9 分位数；检查 PMF 归一化、CDF 反演、患者置换不变以及禁止信息泄漏。

### Prescriptiveness 汇总（主指标）

每个 `replicate r` 内先平均随机训练的 3 Seed 成本，再对 10 个独立数据重复求均值 `\bar C_{m,G}`。对于 SAA 与 Oracle 采用相同测试数据/评价口径。

\[
\boxed{\widehat{\mathcal P}_{m,G}=\frac{\bar C_{\rm SAA,G}-\bar C_{m,G}}{\bar C_{\rm SAA,G}-\bar C_{\rm Oracle,G}}},
\]

\[
\boxed{\widehat{\mathcal P}_{m,5G}=\frac{\sum_{G\in\mathcal G}(\bar C_{\rm SAA,G}-\bar C_{m,G})}{\sum_{G\in\mathcal G}(\bar C_{\rm SAA,G}-\bar C_{\rm Oracle,G})}.}
\]

不得先对重复或五 `G` 各自算 P 再平均；P 可以为负或大于 1，不裁剪。分母非正或接近零需标识。对**10 个独立重复做配对簇 Bootstrap**（事前固定如 10,000 次），重抽样时整组保留同重复下全部方法、五个 `G`、SAA 和 Oracle，然后重新计算 P；报告 CI。辅助指标 `MeanCost`、`SL_G=Pr(I_end≥G)` 一并给出，不能宣称服务率越高就越好。

## 8. 运行前验收与最低产出

**必须先通过的机制检查**：患者行动 7 类 PMF 和条件需求 PMF 归一；选择概率无普遍饱和；优先级挤出方向；随 `c_p` 移动的患者匹配权重；需求与患者顺序无关；今日患者在 Oracle 中正确边际化；Oracle 数值精度满足误差预算。

**研究信号（先检查、再冻结、不能保证排名）**：总体 SAA–L3 Oracle 成本差非微小；完整患者级共享具有可检验的额外价值；关闭患者竞争或动态匹配的反事实分位数在真实需求下有非微小运营损失。L2 条件 Oracle 在连续特征下需要专门的条件推断方案，**不能**直接用 Full Oracle 或带 45 状态的旧程序替代；若未计算，就只报告 L2/L3 已训练模型之间的实现价值，不把它说成 Bayes 信息价值。

**最低交付**：

```text
v7_final_manifest.json                   # 新版 DGP/训练/Oracle 精度/种子与 Git SHA
v7_final_performance_metrics.csv         # repeat × experiment × model × seed × G: cost, SL
v7_final_oracle_quantiles.csv           # repeat × test_day: Q*(A-,X)
v7_final_oracle_costs.csv               # repeat × G: oracle cost, oracle SL
v7_final_prescriptive_p.csv             # P1/P2/P3 × model × G/5G: P, paired CI, denominator
figures/P1_Prescriptiveness.png
figures/P2_Prescriptiveness.png
figures/P3_Prescriptiveness.png
```

**历史声明**：已经完成的离散 `AXID-DC v7` 结果（DeepQR、Sum Pool、L2 优于 Full DAT 等）保持独立存档；此连续修订版在所有指标上都属于新实验。**新 DGP 并不数学保证 DAT、Full 或 L3 的有限样本排名；正式实验必须如实保留结果。**
