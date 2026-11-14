# 04｜AXID-DC v8：简洁性能实验协议（P1 / P2 / P3）

> **唯一 DGP**：[`02_AXID_DC_v8_DGP.md`](02_AXID_DC_v8_DGP.md)，版本 `axid_dc_v8_hb_context`。不混用 v3、v7 或 v7-Final 的任何正式模型结果和 Oracle。
>
> **执行顺序**：先完成全部模型训练和 **MeanCost / SL_G**；保留同一批测试日及逐重复成本；**Full Oracle 与 Prescriptiveness 可以后算**，不需要为计算 P 重训任何模型。除这套后处理外，不开展额外的中间 Oracle 大规模计算。

## 1. 规模、指标与运行原则

| 项目 | 固定设置 |
|---|---|
| 训练/测试 | 10 个独立 DGP 重复；每重复 2,500 训练日 + 3,000 独立测试日 |
| 训练种子 | 随机方法每重复 3 个固定 Seed；**先在每重复内平均，再跨重复汇总**。确定性方法保留一个真实输出 |
| 运营任务 | `τ=0.9`；`G={260,280,300,320,340}`；各方法对同一测试日**只预测一次**，五 G 不重训 |
| P1 | 八模型：DAT、SAA、wSAA、Flat、Set-ETO、LQR、GBQR、DeepQR |
| P2 | 八种结构：Full、No-SA、No-SA Matched、No-Day-Query、Mean/Sum Pool 及两个 Pool Matched |
| P3 | 四级信息：L0、L1、L2、L3 |
| **主指标** | **Oracle-normalized Prescriptiveness `P`**（Oracle 后算） |
| 辅助指标 | `MeanCost`、目标服务率 `SL_G=Pr(I+q−D≥G)` |

全部模型使用相同测试日、相同 `I,G,D` 和同一补货政策；固定模型宽度、训练预算、优化器与其他超参数，**不根据正式测试结果选 Seed、模型或重新调 DGP**。一次实验中的 10 个独立数据重复才是统计推断单位，不能把 `10×3` 当成 30 个独立样本。

## 2. 输入、DAT 架构与统一表格摘要

可见患者仅为过去患者：

\[
A^-=\{a_i=(u_i,r_i,\mathrm{Hb}_i)\}_{i=1}^{N^-},\qquad
X=\left(\frac{c_p+1}{2},\frac{c_e+1}{2},\frac{I-205}{130},\frac{N^-}{8},\frac{\sum u_i}{8},\frac{\sum r_i}{8},\frac{\sum \overline{\mathrm{Hb}}_i}{8}\right)\in\mathbb R^7.
\]

`Hb不足` 定义为 `\overline{\mathrm{Hb}}_i=1-\mathrm{Hb}_i`；患者内部到达标记 `b=0` 对所有 `A⁻` 恒定，**不用作预测输入**。今日患者 `A⁺`、医生 `ζ,T,\widetilde r,ω,R,h,U`、测试实现的 `Y,D` 均不得进入测试期模型输入。

**传统表格模型保留 16 维人工摘要 `ψ`**，只用过去患者计算：

\[
\boxed{\begin{aligned}
\psi(A^-,X)=\bigl[&X,\;
\operatorname{avg}(ur),\operatorname{avg}(u\overline{\mathrm{Hb}}),\operatorname{avg}(r\overline{\mathrm{Hb}}),\\
&\min u,\max u,\min r,\max r,\min\overline{\mathrm{Hb}},\max\overline{\mathrm{Hb}}\bigr]\in\mathbb R^{16}.
\end{aligned}}
\]

`ψ` 是 P1 中表格模型的手工特征，**不能泄漏到 P3 的 L2**；其三个交叉矩与六个极值均有严格的授权来源。

**DAT 不改变模块拓扑**：患者共享两仿射层 ReLU MLP → 1 层带残差 Self-Attention → `X` 经 Day Linear 形成单日级 Query → 1 层带残差 Cross-Attention → 带残差的 Day FFN → 线性标量并裁剪到 `[0,61]`。保留 **3 个独立 LayerNorm**；无位置编码、患者 FFN、Final LN、额外注意力层。直接用日级 `D` 训练 Pinball(0.9)。

**v8 主容量预先固定**：`d_model=64`、Self/ Cross 各 4 头、`patient_hidden=64`、`day_ffn_hidden=96`；按标准 Linear / MHA / LN 实现预计约 **51,105** 个有效参数，必须以实际模型计数核验。选 64 是因为新增 Hb 后 `d_a=3,d_x=7,M=8` 对应原构造性逼近充分维度下限约 57，并考虑 2,500 日样本；**不能据此保证 DAT 优于基准**。其他算法优化超参数继承事先冻结的 v7-Final 训练配置，若实现不兼容须在训练前记录唯一修订值。`P1 DAT = P2 Full = P3 L3` 的同一模型及预测**只训练一次、在三处复用**。

## 3. P1：八模型（最小定义）

| 方法 | 预测输入 | 固定算法 / 目标 |
|---|---|---|
| **DAT** | 完整 `A⁻,X` | 固定 SA→动态 CA→Day FFN；对日级 `D` 优化 Pinball(0.9) |
| **SAA** | 训练日总需求 `D`，不使用测试日条件特征 | 全样本无条件经验 0.9 分位数；是 P 的零点 |
| **wSAA** | `ψ∈R¹⁶` | 随机森林叶节点邻近权重 → 历史 `D` 的加权 0.9 分位数 |
| **Flat** | 患者 `a_i`；日级 `X` | **逐患者 MSE + 剩余需求 MSE**，预测值直接相加；**不做分位数校准或概率卷积** |
| **Set-ETO** | 完整 `A⁻,X` | DeepSets Mean Pool → HurdleGamma 条件分布，NLL 训练 → 0.9 分位数 |
| **LQR** | `ψ` | 固定正则的线性 Pinball(0.9) 回归 |
| **GBQR** | `ψ` | 固定参数的分位数梯度提升树，目标 0.9 |
| **DeepQR** | 完整 `A⁻,X` | DeepSets Mean Pool + 融合 MLP；对日级 `D` 优化 Pinball(0.9) |

**Flat 唯一训练定义**。训练期允许使用 `Y^-_{ti}` 和剩余需求标签

\[\boxed{D_t^{\rm rem}=D_t-\sum_{i\in A_t^-}Y^-_{ti}=\sum_{j\in A_t^+}Y^+_{tj}+J_t.}\]

两个小型回归器**分别**按 MSE 训练：

\[
\boxed{\begin{aligned}
g_\theta&:\mathbb R^3\to\mathbb R,& \min_\theta\sum_{t,i\in A_t^-}(Y_{ti}^--g_\theta(a_{ti}))^2,\\
h_\eta&:\mathbb R^7\to\mathbb R,& \min_\eta\sum_t(D_t^{\rm rem}-h_\eta(X_t))^2.
\end{aligned}}
\]

最终：

\[
\boxed{\widehat D_{\rm Flat}=\operatorname{clip}_{[0,61]}\left(
\sum_{i\in A^-}\operatorname{clip}_{[0,6]}g_\theta(a_i)
+\operatorname{clip}_{[0,13]}h_\eta(X)
\right).}
\]

Flat 是朴素**均值型**点预测，不预测 0.9 分位数；在非对称库存成本下可能大幅落后于 SAA，**不能事后补加安全库存缓冲**。它使用额外患者级训练监督，须披露；其他模型通常只使用日级 `D`。P1 的 DAT/DeepQR/Set-ETO 读取完整患者集合，wSAA/LQR/GBQR 用 `ψ`，因此 P1 差异混合了表示与估计方法差异；**Attention 的独立贡献依赖 P2 判断**。

## 4. P2：八个结构条件

| ID | 相对 Full 的唯一变动 |
|---|---|
| `full` | 完整 DAT |
| `no_self_attention` | 去掉 SA 分支，保留动态 CA 和完整输入 |
| `no_self_attention_param_matched` | 去 SA，适度加宽非注意力层，使有效参数与 Full 匹配 |
| `no_day_query` | CA 的 Query 改为**与 X 无关的可学习常数**；`X` 仍经过 Day Linear 残差及 FFN |
| `mean_pool` | 用有效患者 Memory 均值 + 输出映射替代 CA；保留 SA、完整 `X` |
| `mean_pool_param_matched` | Mean Pool 不变，仅调整隐藏宽度匹配参数 |
| `sum_pool` | 用有效患者 Memory 求和 + 输出映射替代 CA；保留 SA、完整 `X` |
| `sum_pool_param_matched` | Sum Pool 不变，仅调整隐藏宽度匹配参数 |

三种参数匹配配置依据**训练前实际有效参数计数**固定，优先控制在与 Full 差 `≤1%`，最宽 `≤5%`。匹配不得使用测试成绩。Full DAT 的 64 维主配置下，如果 No-SA 同时移除其专属 LN，扩大 Day FFN 隐宽由 96 至 **226**，理论上可补回约 16,768 个参数；实际计数与实现一致性必须核验。所有 P2 组使用同一 `A⁻,X`、Pinball(0.9)、数据划分、训练预算和预设随机种子。

## 5. P3：四级医院信息共享

| 信息级 | 日级授权 | 患者 Token | 管理含义 |
|---|---|---|---|
| **L0** | 仅 `X` 前 3 坐标：`c_p,c_e,I` | **一个固定 Dummy Token、固定 Mask** | 血液中心尚未获得患者信息 |
| **L1** | L0 + `N⁻,Σu_i`（`X` 前 5 维） | 同一 Dummy | 医院共享基本订单汇总 |
| **L2** | L1 + `Σr_i,ΣHb不足_i`（完整 `X`） | 同一 Dummy | 医院共享临床风险/Hb 汇总 |
| **L3** | 完整 `X` | 真实过去患者 `(u,r,Hb)` 与 Mask | 医院共享完整患者级信息及对应关系 |

L0–L2 的未授权 `X` 坐标用固定常数占位，绝不能通过真实 Token 数、Mask、位置、填充模式等泄漏 `N⁻`；L2 **不允许**使用 `ψ` 的交叉矩或极值。四级 DAT 保持同样的结构与预算，各自从头训练；L3 复用 P1 Full。Bayes 成本随信息增强非增，独立训练的神经网络测试成本**不保证单调**。所有级别均不能看到今日患者。

## 6. 两阶段计算与 Oracle/P 后处理

### 阶段一：正式训练并保存 MeanCost / SL_G

每个重复先独立生成并**持久化** 2,500 训练日、3,000 测试日。全部方法/消融/信息等级共用该重复的测试日；训练完成后，对每个测试日只生成一个标量预测（必要时裁剪至 `[0,61]`），复用于五个 `G`：

\[
\boxed{q_{m,G}=[\widehat D_m+G-I]_+,\quad
c_{m,t,G}=0.9(G-I-q_{m,G}+D)_++0.1(I+q_{m,G}-D-G)_+,\quad
SL_{m,t,G}=\mathbf1\{I+q_{m,G}-D\ge G\}.}
\]

每个随机模型的 3 Seed 在各重复内取平均，再对 10 重复求跨重复均值与 CI。阶段一完成后可以诚实地报告 **MeanCost/SL_G 已完成、P pending**；不应以缺少 Oracle 为由重训。

### 阶段二：仅补算 Full Oracle，再计算 P

血液中心 Oracle：`f*(Z)=Q₀.₉(D|A⁻,X)`，不能使用当天真实 `A⁺,U,D` 计算预测分位数。今日人数 `N⁺=0,1,2` 的概率精确加权；今日患者 **3n 维**连续 `(u,r,Hb)` Beta 画像用随机化 QMC（最多 **6 维**）；在每种候选今日患者集合、每个共享 `U` 的 GH 节点下先联合卷积七类患者 PMF，再对 `U` 和隐藏今日患者边际化。`J~Unif(0,1)` 后反演分位数；**不能沿用旧离散版 45 种状态 Oracle，也不能先各自积分 U 再卷积**。

为了省时：**每个测试日求一次 Oracle 分位数**（总共 30,000 日），五 G、P1/P2/P3 共用；GH 以 25 节点为主，QMC 点数根据**小规模独立精度测试**冻结（如从 128/256/512 点检查），不对全部日期扫多个节点数、不要求 L0/L1/L2 Oracle 或额外中间机制 Oracle。检查 PMF 质量、CDF 反演、置换不变性、16/25/32 GH 的小样本一致性以及 QMC 加密收敛。使用相同测试日真实 `D` 计算 Oracle 已实现的成本和 `SL_G`。

按 `repeat` 内先平均三 Seed，再跨 10 独立重复汇总成本 `\overline C_{m,G}`。主指标：

\[
\boxed{\widehat{\mathcal P}_{m,G}=
\frac{\overline C_{\rm SAA,G}-\overline C_{m,G}}
{\overline C_{\rm SAA,G}-\overline C_{\rm Oracle,G}}.}
\]

五 G 综合：

\[
\boxed{\widehat{\mathcal P}_{m,5G}=
\frac{\sum_{G\in\mathcal G}(\overline C_{\rm SAA,G}-\overline C_{m,G})}
{\sum_{G\in\mathcal G}(\overline C_{\rm SAA,G}-\overline C_{\rm Oracle,G})}.}
\]

**不能先算每重复或每 G 的 P 再平均**；不裁剪负 P 或 P>1；分母非正或太小时单独标识并说明。对 10 个独立数据重复执行**配对簇 Bootstrap**，每次保留该重复的全部模型及五 G，再重新计算比值，报告 95% CI；不得将日内或 Seed 伪装成独立重复。相同 DAT 的 P1/P2/P3 主结果复用**同一配对重抽样索引**，确保 CI 一致。

## 7. 最小数据产物与验收

```text
axid_v8_manifest.json              # DGP 版本、配置、种子、代码 SHA、训练与 Oracle 数值精度
v8_test_days.*                     # 10×3000，含 A-/X/I/D；真实 A+ 与 U 仅保存在私有核验区
v8_performance_metrics.csv         # repeat×experiment×model×seed×G：MeanCost、SL_G
v8_performance_summary.csv         # 10 独立重复汇总；P 暂缺时标为 pending
v8_oracle_quantiles.csv            # repeat×test_day：Q*(A-,X)；第二阶段生成
v8_oracle_costs.csv                # repeat×G：Oracle MeanCost/SL；第二阶段生成
v8_prescriptive_p.csv              # P1/P2/P3×model×G/5G：P、配对 CI、分母
```

**运行前最低验收**：DGP 患者置换不变；同日一个共享 `U`；医生个体 PMF 归一；不同样本 `N` 的 Mask 正确；`Y^-`/`D_rem` 仅对 Flat 训练开放；L0–L2 无患者信息泄漏；日级聚合 `R` 不与 Flat 剩余需求重名；Oracle 的 Beta/QMC 和 GH 积分精度抽检。不得用已完成的其他版本的模型测试排名为 v8 提供“确认性”结论。
