# 04｜AXID-DC v7：精简、可复现的性能实验协议

> **唯一生成机制**：`02_AXID_DC_v7_DGP.md`（与 v6 的 DGP 公式、系数相同）。**v7 相对 v6 的唯一方法学实质修改**：`Flat-ETO` 改为 **Flat**，采用**患者预测 + 剩余需求预测**两个分别监督的 MSE 回归器；不再拟合七类 PMF、残差 CDF 或做卷积。**16 维 \(\psi\) 完整保留**，其他七种 P1 方法保持原有算法。优先做到可复现、无信息泄漏、无需重复训练/重复 Oracle。

## 1. 正式实验和固定预算

| 项目 | 固定设置 |
|---|---|
| 主任务 | \(\tau=0.9\)，\(G\in\{260,280,300,320,340\}\)；每模型只训练一次分位数/点预测器，五 \(G\) 共用输出 |
| 数据规模 | **10 个独立重复**，每重复 **2,500 个训练日 + 3,000 个测试日**（同一重复所有方法共用测试日） |
| 随机性 | 随机模型 **3 个事先固定训练 Seed**；同重复先平均三 Seed，再跨 10 重复推断 |
| 输入边界 | 血液中心只见已登记患者 \(A^-\) 和事前 \(X\)；测试期不可见 \(A^+\)、\(U\)、\(Y_i\)、\(D\) |
| 结果指标 | **主：Oracle-normalized Prescriptiveness \(\mathcal P\)**；辅：**MeanCost、\(SL_G\)** |
| 不允许 | 测试驱动调参、选择最佳 Seed、根据 DAT 排名换 DGP、将 30 个 Seed 当作 30 次独立重复 |

- **P1：8 模型**：DAT、SAA、wSAA、**Flat**、Set-ETO、LQR、GBQR、DeepQR；删除 Rolling-SAA。
- **P2：8 结构**：Full、No-SA、No-SA-matched、No-Day-Query、Mean、Mean-matched、Sum、Sum-matched。
- **P3：4 信息等级**：L0、L1、L2、L3（L3 是完整事前患者共享，不存在 L4）。

所有树模型、MLP、优化器、宽度、轮数及正则化的**唯一配置在正式运行前冻结**；不搜索超参数。原 v3/v6 结果不混合为 v7 的结果。

## 2. 输入合同与 DAT 不变的拓扑

### 2.1 血液中心实际观察

\[
A^-=\{a_i=(u_i,r_i,s_i)\}_{i=1}^{N^-},\quad
X=\left(\frac{c_p+1}{2},\frac{c_e+1}{2},\frac{I-205}{130},\frac{N^-}{8},\frac{\sum u_i}{8},\frac{\sum r_i}{8},\frac{\sum s_i}{8}\right)\in\mathbb R^7.
\]

**传统表格基线沿用原 16 维人工摘要，不删改**：

\[
\boxed{\psi(A^-,X)=\left[X,\overline{ur},\overline{us},\overline{rs},\min u,\max u,\min r,\max r,\min s,\max s\right]\in\mathbb R^{16}.}
\]

\(\psi\) 的全部坐标只能使用 \(A^-\)。P3 的 L2 **只允许 7 维 \(X\)**，不能借用包含交叉矩与极值的 \(\psi\)。

### 2.2 DAT

```text
A⁻ [B,N_pad,3] → 共享两仿射层 ReLU Patient MLP → E⁰
E = E⁰ + SelfAttention(LN_sa(E⁰))                   # 患者 Mask
X [B,7] → u = DayLinear(X)
v = u + CrossAttention(LN_q(u), LN_m(E))            # 单个动态 Day Query
u⁺ = v + DayFFN(v)                                  # 两层 ReLU、无附加 LN
f(A⁻,X) = Clip(Linear(u⁺), 0, 61)
```

固定 **1 层 SA、1 层 CA、三处独立 LayerNorm、无位置编码/患者 FFN/最终 LN**。建议维持冻结的 v6 主配置：`d_model=80`、SA/CA 各 4 头、`patient_hidden=96`、`day_ffn_hidden=128`；工程必须审计真实参数总数。DAT/DeepQR 以日级 \(D\) 的 Pinball(0.9) 训练。DAT 在 P1、P2 Full、P3 L3 使用同一个模型实例/预测结果。

## 3. P1 八种模型：输入、学习目标、输出

| 方法 | 测试输入 | 训练与预测 | 备注 |
|---|---|---|---|
| **DAT** | 完整 \(A^-,X\) | SA → 动态 CA → Day FFN，Pinball(0.9) 输出分位数 | 主模型 |
| **SAA** | 不用条件输入；执行补货仍知 \(I,G\) | 训练日 \(D\) 的无条件经验 0.9 分位数 | \(\mathcal P=0\) 的参照 |
| **wSAA** | \(\psi\) | RF 叶节点历史日相似权重 → 加权经验 0.9 分位数 | Bertsimas–Kallus 风格的加权 SAA |
| **Flat** | 每位旧患者单独 \(a_i\)，第二预测器仅 \(X\) | **两个独立 MSE 回归器，点预测直接相加** | 允许额外患者级训练监督；见 §4 |
| **Set-ETO** | 完整 \(A^-,X\) | DeepSets Mean Pool → HurdleGamma NLL → 拟合分布 0.9 分位数 | 参数分布近似的 ETO |
| **LQR** | \(\psi\) | 正则化线性 Pinball(0.9) 分位数回归 | 预先固定正则参数 |
| **GBQR** | \(\psi\) | 梯度提升树 Quantile(0.9) | 预先固定树与提升设置 |
| **DeepQR** | 完整 \(A^-,X\) | DeepSets Mean Pool + Pinball(0.9) 直接输出分位数 | 无显式 Attention 的集合对照 |

除 Flat 外，所有基准均只需训练日期的总需求标签 \(D\)；Flat 的额外监督权限**仅限训练日**。P1 不是对单一 Attention 模块的纯因果检验，因为各模型的输入表示/训练目标不同；模块证据来自 P2。

**统一预测范围**：各算法交给补货政策的标量在测试/部署时均裁剪至 \([0,61]\)（DAT 自身已有输出 Clip）；不可只对 DAT 裁剪而允许其他方法输出无界预测。训练目标仍按各自合同执行。

## 4. Flat：按最终决定执行，不再叫 Flat-ETO

真实需求分解（**训练期可使用患者结果标签**）：

\[
\boxed{D_t=\sum_{i\in A_t^-}Y_{ti}^-+R_t,\qquad
R_t=D_t-\sum_{i\in A_t^-}Y_{ti}^-=\sum_{j\in A_t^+}Y_{tj}^++J_t.}
\]

**第一部分：患者 MSE 回归。** 共享小 MLP \(g_\theta:\mathbb R^3\to\mathbb R\)，每名已知患者仅输入 \(a_{ti}\)，用**该患者实际发血标签**分别监督：

\[
\boxed{\widehat Y_{ti}=g_\theta(a_{ti}),\quad
\min_\theta\sum_{t\in\mathrm{train}}\sum_{i\in A_t^-}(Y_{ti}^--g_\theta(a_{ti}))^2.}
\]

**第二部分：剩余需求 MSE 回归。** 另一小 MLP \(h_\eta:\mathbb R^7\to\mathbb R\)，仅使用 \(X_t\)，用实际残差标签 \(R_t\) 独立监督：

\[
\boxed{\widehat R_t=h_\eta(X_t),\quad
\min_\eta\sum_{t\in\mathrm{train}}(R_t-h_\eta(X_t))^2.}
\]

**预测和补货**（只在部署时限制物理输出，训练期按正常 MSE 回归）：

\[
\boxed{\widehat D_{\mathrm{Flat}}=
\operatorname{clip}_{[0,61]}\left[
\sum_{i\in A^-}\operatorname{clip}_{[0,6]}g_\theta(a_i)
+\operatorname{clip}_{[0,13]}h_\eta(X)
\right],\quad
q_{\mathrm{Flat},G}=[\widehat D_{\mathrm{Flat}}+G-I]_+.}
\]

其中最多两个新增患者，各自最多 6 单位且 \(J<1\)，故 \(R\in[0,13]\)。**不预测分位数、不拟合七档 PMF、不拟合残差 CDF、不卷积、不使用 \(\psi\)**。此朴素 Flat 估计的是各分量的条件**均值**而不是 0.9 分位数，允许其直接参与非对称库存决策，但不得将其差距全部解释为缺少 Attention。

**必须披露监督不对称**：Flat 额外使用训练日 \(Y_{ti}^-\) 和由其计算的 \(R_t\)；在测试日预测时，Flat **不能读实际 \(Y_i^-,R,D\)**。两部分分别拟合时 \(g(a_i)\) 忽略患者/日级交互，且 \(\sum_i g(a_i)+h(X)\) 不保证等于 \(\mathbb E[D\mid A^-,X]\)，这正是朴素基线的限定。

## 5. P2 八组受控结构消融

| ID | 相对 Full 的唯一操作 | 主要对照 |
|---|---|---|
| `full` | 完整 DAT | 主结构 |
| `no_self_attention` | 去掉患者 SA 分支，其余不变 | SA 的观察增量 |
| `no_self_attention_param_matched` | 去 SA，用 Day FFN 宽度匹配 Full 参数量 | 控制参数量差异 |
| `no_day_query` | CA 改用**可学习常数 Query**，但 \(X\) 仍经 Day Linear 残差及 FFN | 动态日级 Query |
| `mean_pool` | 用有效患者 Memory 的 Mean Pool + 投影代替 CA；SA 保留 | 静态均值聚合 |
| `mean_pool_param_matched` | Mean Pool 不变，调非注意力隐藏宽度匹配参数量 | 容量匹配均值聚合 |
| `sum_pool` | 用有效患者 Memory 的 Sum Pool + 投影代替 CA；SA 保留 | 静态求和聚合 |
| `sum_pool_param_matched` | Sum Pool 不变，调非注意力隐藏宽度匹配参数量 | 容量匹配求和聚合 |

Mean/Sum 只聚合有效患者的 \(\mathrm{LN}_m(E_i)\)，均保留完整 \(A^-,X\)、日级残差及 FFN。参数匹配只根据**真实可训练参数数目**于训练前确定，误差 \(\le5\%\)（尽量 \(\le1\%\)）；如只改 Day FFN 宽度 \(H\)，\(d=80\) 时 \(P_{\rm FFN}(H)=161H+80\)。不可按测试 P 排名反复调宽度。对比 Full 与 No-SA、Matched No-SA、No-Day-Query、Matched Mean/Sum 的**配对差**。

## 6. P3 四级信息共享（L0–L3）

| 等级 | 获得的日级信息 | 患者 Token | 血液中心的现实情景 |
|---|---|---|---|
| **L0** | \(c_p,c_e,I\)（\(X\) 前 3 维） | 固定 Dummy | 只有运营环境与库存，无患者数据 |
| **L1** | L0 + \(N^-,\sum_{A^-}u_i\) | 固定 Dummy | 医院提供患者人数及预计用血基本汇总 |
| **L2** | L1 + \(\sum_{A^-}r_i,\sum_{A^-}s_i\)（完整 \(X\)） | 固定 Dummy | 进一步共享高风险及计划/急诊病例统计 |
| **L3** | 完整 \(X\) | 真实 \(A^-\) 和 Mask | 完整逐患者共享，保留风险—强度—类型的对应关系 |

L0–L2 必须**统一仅有一个有效 Dummy Token**，不允许真实患者数通过 Token/Mask 泄漏；未授权 \(X\) 坐标为固定占位值。L3 才使用真实患者集合。**任何等级都不知道 \(A^+\)**。各级同一 DAT 结构和参数预算、同样训练样本与 Seed、从头训练；实际训练结果不必严格单调，而对应条件 Bayes 成本因信息嵌套而单调不增。共同的主指标分母始终为**L3 信息权限下的 Full Oracle**；本版**不要求正式运行额外 L2 Oracle**。

## 7. Full Oracle、Prescriptiveness 与最快的正确后处理

**唯一合法 Full Oracle**：\(f^*(Z)=Q_{0.9}(D\mid A^-,X)\)。对两名潜在新增患者的 45 个无序状态精确求和；在每状态和同一 \(U\) 下卷积全部患者七档选择分布，再用 25 节点 Gauss–Hermite 积分；需求整数部分加 \(J\sim U(0,1)\) 后反演连续分位数。**Oracle 不能使用真实新增患者、当日实际 \(D\) 或 \(U\) 作为预测输入。**

### 7.1 一次计算，多处复用

1. **每独立重复**生成并冻结训练/测试数据，以及训练 Flat 所需的 `Y_old_train`、`R_train` 标签。模型测试端只接收 \(A^-,X\)。
2. **所有预测器只输出一次**测试日的标量 \(f_m(Z_t)\)，五个 \(G\) 使用同一预测；**完整 DAT 只训练一组**，P1、P2 Full、P3 L3 共用。
3. **Oracle 每测试日只求一次** \(Q^*(Z_t)\)：总共 \(10\times3000=30,000\) 个不同测试状态，而不是 30,000 × 5 × 模型数 × Seed。计算时可按八类型计数分组同分布患者、使用编译/批量小卷积；保持 45 状态与共享 \(U\) 的正确处理。
4. 对每个 \(G\) **广播计算**所有模型和 Oracle 的 \(q\)、MeanCost、\(SL_G\)，使用相同测试日实际 \(D_t\)。不对每个方法另算 Oracle、不为每个 \(G\) 重训。**不额外枚举 L2/L1 Oracle**，除非单独开展机制研究。
5. 小样本检查 16/25/32 节点积分、CDF(\(Q^*\))≈0.9、45 状态归一及排列不变性；正式统一使用 25 节点。数值错误不能以盲目裁剪概率掩盖。

### 7.2 正式指标与统计单位

定义 \(C_{m,r,G}\) 为**同一独立重复内**该模型随机 Seed 平均后的测试 MeanCost（确定性模型只留一个真实结果），\(\bar C_{m,G}=\frac1{10}\sum_{r=1}^{10} C_{m,r,G}\)。

\[
\boxed{\widehat{\mathcal P}_{m,G}=\frac{\bar C_{\rm SAA,G}-\bar C_{m,G}}{\bar C_{\rm SAA,G}-\bar C_{\rm Oracle,G}}},
\]
\[
\boxed{\widehat{\mathcal P}_{m,5G}=\frac{\sum_{G\in\mathcal G}(\bar C_{\rm SAA,G}-\bar C_{m,G})}{\sum_{G\in\mathcal G}(\bar C_{\rm SAA,G}-\bar C_{\rm Oracle,G})}.}
\]

先**Seed 内聚合 → 独立重复平均 → 汇总五 \(G\) 的成本差 → 取比值**；不得先平均 10 个逐重复 P 或五个逐 \(G\) 的 P。\(\mathcal P\) 不裁剪 \([0,1]\)；分母非正或近零必须显式报告。辅助指标：\(\mathrm{MeanCost}=\bar C_{m,G}\)；\(SL_G=\Pr(I+q_m-D\ge G)\)，它不要求越高越好。

对 **10 个独立数据重复**作事前固定 10,000 次**配对簇 Bootstrap**；每次抽样同时保留重复内全部方法、五个 \(G\)、SAA、Oracle，并从重抽样平均成本**重新计算比值**。报告 P/MeanCost/SL 的 95% CI；同一 Full DAT 在 P1/P2/P3 的点估计和 CI 必须一致。

## 8. 最低验收和交付

- 信息权限：已知患者与新增患者严格分离；Flat 的逐患者标签/剩余需求**仅训练期**可读；测试所有方法只读批准的上游字段。
- 真实性：每名患者七档 PMF、45 状态权重、完整 Oracle 的离散 PMF 非负且归一，输入患者重排结果一致；\(\tau\) 分位数反演精度合格。
- 可比性：同一测试日、同一 \(G\) 网格，Full DAT 共用；所有随机方法 3 Seed、10 独立重复；基准预设超参数且不选最佳 Seed。
- 运营机制：记录各 \(G\) 补货活跃率；报告 SAA–Oracle 分母及 L2→L3 Bayes 价值较小的既有机制诊断，不把后者包装成保证 DAT 增益。

**最低结果文件**（无需不必要的中间扫描）：

```text
axid_dc_v7_manifest.json                # DGP、算法配置、数据/训练种子、Oracle 节点与 Git SHA
axid_dc_v7_metrics.csv                 # replicate × P1/P2/P3 × method × seed × G: Cost, SL
axid_dc_v7_oracle_quantiles.csv        # replicate × test_day: Q*(A-,X)
axid_dc_v7_oracle_costs.csv            # replicate × G: Oracle MeanCost / SL
axid_dc_v7_prescriptive_p.csv          # P1/P2/P3 × method × G/5G: P、CI、分母
figures/P1_Prescriptiveness.png
figures/P2_Prescriptiveness.png
figures/P3_Prescriptiveness.png
```

> **版本声明**：AXID-DC v7 的 DGP 与 v6 一致；更换 Flat 基准后，v7 必须在独立冻结数据上按本协议统一评价。不能将历史 v6 的 Flat-ETO 结果重命名为 v7 Flat，也不能将 v3 的数值当作 v7 结果。
