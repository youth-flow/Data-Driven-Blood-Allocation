# 04｜AXID-DC v6：性能实验正式协议（P1 八模型、P2 八结构、P3 四级共享）

> **唯一生成机制**：`02_AXID_DC_v6_DGP.md`，版本 `AXID-DC v6 / Priority-Based Doctor Choice`。本版使用高优先级患者挤出、动态日级 Softmax、库存稀缺惩罚及七档医生离散选择；它与历史 AXID-v3/v4/v5 的数据、Oracle、报告不合并。
>
> **核心信息合同**：上游血液中心只在决策前观察 **\(A^-\)（提前登记患者）和 \(X\)**；医院医生基于 \(A^-\cup A^+\)、事前库存 \(I\) 生成随机需求。实际 \(A^+\)、医生选择、共同冲击与 \(D\) **不能泄漏**到任何模型或 Full Oracle。
>
> **评价顺序**：① **Oracle-normalized Prescriptiveness \(\mathcal P\)**（主指标）；② **MeanCost**；③ **目标库存达标率 \(SL_G\)**（两项辅助指标）。较高服务率可能源于超目标持有成本增加，不按“越高越好”将模型排名。

---

## 1. 研究问题及正式分组

| 实验 | 组数 | 研究问题 | v6 正式组别 |
|---|---:|---|---|
| **P1：方法对比** | **8** | 哪种方法在已共享的同一事前信息下，能实现更高比例的潜在运营价值？ | DAT、SAA、wSAA、Flat-ETO、Set-ETO、LQR、GBQR、DeepQR |
| **P2：架构消融** | **8** | 患者间优先级竞争与日级条件化聚合是否需要 SA 和动态 CA？ | Full、No-SA、**No-SA-matched**、No-Day-Query、Mean、Mean-matched、Sum、Sum-matched |
| **P3：信息共享** | **4** | 医院从无患者信息到基础汇总、临床汇总及逐患者共享的增量价值有多大？ | **L0、L1、L2、L3**，没有 L4 |

**删除 Rolling-SAA**：合成环境每天 i.i.d.，不存在滚动估计针对的趋势/分布漂移；它不作为本版 P1 主基线。原 v3 已执行的 Rolling-SAA 数据和结论保留，不能从历史文稿中选择性抹去。

**理论机制与模型排名必须区分**：医院端真实优先级竞争不保证 Full DAT 一定胜过 No-SA；动态 \(\omega_i\) 不保证 Full 一定胜过 DeepQR。P1/P2/P3 的胜负只由冻结方案下的独立测试数据决定。

## 2. 固定生成、训练和信息权限

| 设置 | 正式值 |
|---|---|
| 数据生成 | `AXID-DC v6`，以 DGP 参数清单及 SHA 固定，不得按方法排名改系数 |
| 独立数据重复 | **10** 个；每次重新生成 **2,500 个训练日 + 3,000 个测试日** |
| 每重复随机种子 | 随机训练方法 **3 Seed**；先在重复内平均，再跨 10 个独立数据重复统计 |
| 主任务 | 需求 \(\tau=0.9\) 分位数；无其他主 \(\tau\)；不同 \(G\) 不重新训练 |
| 库存目标 | \(\mathcal G=\{260,280,300,320,340\}\)，主文重点展示 \(G=300\) 和五 \(G\) 综合 |
| 上游患者输入 | \(A^-\in\mathbb R^{N^-\times3}\)，\(3\le N^-\le8\)；患者无顺序 |
| 日级输入 | \(X\in[0,1]^7\)；统计量仅基于 \(A^-\)，包括事前库存 \(I\) |
| 需求/输出范围 | \(0\le D\le61\)，预测输出 \(f(Z)\in[0,61]\) |
| 医院端新增患者 | \(N^+\in\{0,1,2\}\)，对上游模型及 Oracle **未知实现** |
| 优化器与超参 | 在正式训练前冻结每种方法唯一配置；**不开展测试驱动候选搜索、不选最佳 Seed** |
| 重复的配对性 | 每独立重复内，各模型、变体与信息级共用相同测试日 \(D\) |

**确定性基线**（如 SAA、固定求解的 LQR）每重复只保留一个真实结果，不能假造三个 Seed 以增加统计样本量。独立统计单位始终是 10 个数据重复，而不是 30 个训练 Seed。

### 2.1 DAT 拓扑固定，仅输入维度适配 v6

```text
A⁻: [B,N_pad,3] → 共享两仿射层 ReLU Patient MLP → E⁰
                          E = E⁰ + SA(LN_sa(E⁰))          # 有效患者 Mask
X:  [B,7]       → u = Linear(X)                       # 日级 Query
                          v = u + CA(LN_q(u),LN_m(E))  # 单个日级 Query
                          u⁺ = v + FFN_day(v)          # 两层 ReLU，无附加 LN
                          f(Z) = Clip(Linear(u⁺),0,61)
```

- 保留**一层患者 Self-Attention、一层日级到患者 Cross-Attention、三处独立 LayerNorm、Patient MLP、Day FFN 和残差**；无患者 FFN、位置编码、Day FFN 前额外 LN、Final LN。
- 性能主模型事前固定：`d_model=80`；SA/CA 各 `4` 头；`patient_hidden=96`；`day_ffn_hidden=128`。如果工程实现维度或层数不同，**先独立审计参数总数、再冻结配置，不参考测试结果修正**。
- 有效患者 Mask 不允许包含新增患者 \(A^+\)。Full Model 的宽度及输入权限与 P1/P2/P3 的 Full/L3 结果一致，避免人为重复训练得到不一致表。
- 旧逼近定理的维度充分条件：基于上游已知患者上界 \(M=8\)、\(d_a=3,d_x=7\)，\(d_m=\min\{\binom{11}{8}-1,2\cdot8\cdot3+1\}=49\)，\(d_{\rm eff}=56\)，故 \(d=80\ge57\) 满足该**维度条件**；它不代表有限样本网络必然优于消融。

### 2.2 传统方法共用的 16 维事前摘要

令 \(N=N^-\)，所有平均、最值、交叉矩**只由提前登记患者**计算：

\[
\boxed{\psi(A^-,X)=\left[
X,\overline{ur},\overline{us},\overline{rs},
\min u,\max u,\min r,\max r,\min s,\max s
\right]\in\mathbb R^{16}.}
\]

其中 \(X\) 为 7 维，交叉矩 3 维，极值 6 维。Flat-ETO 患者输入 \([a_i,\psi]\) 为 **19 维**。

> **输入公平性边界**：DAT、Set-ETO、DeepQR 直接读取完整 \(A^-,X\)，传统 LQR/GBQR/wSAA 使用固定的 \(\psi\)；因此 P1 的差异包括**输入表示 + 估计方法**两部分。判别 Attention 模块的直接证据优先使用 P2 的同输入受控比较。

## 3. 补货政策、正确 Oracle 与主要评价

对任何方法 \(f\)，在所有五个 \(G\) 下执行**相同政策**：

\[
q_{f,G}(Z)=[f(Z)+G-I]_+,\qquad I_{\rm end}=I+q_{f,G}-D,
\]
\[
c_{G}(f)=0.9[G-I_{\rm end}]_++0.1[I_{\rm end}-G]_+.
\]

### 3.1 基准与唯一合法 Full Oracle

- **SAA**：仅从该重复的 2,500 个训练日 \(D\) 学习无条件经验 0.9 分位数；测试时只用该常数分位数以及用于执行补货的 \(I,G\)。
- **Full-information conditional Oracle**：\(f^*(Z)=Q_{0.9}(D\mid A^-,X)\)。Oracle 知道真实 DGP 但不能知道测试日实际 \(A^+\)、\(U\) 或 \(D\)，需要对两次新患者机会的 **45 个无序组合**求和，并在每个组合与共享 \(U\) 下卷积全部患者选择 PMF 后积分。
- **数值规范**：25 节点 Gauss–Hermite 为正式计算，部分输入比较 16/25/32；检查每个患者 PMF、45 态权重、\(D_0\) PMF 的概率总和，及 \(F_{D\mid Z}(f^*(Z))\approx0.9\)。
- **评价口径**：在相同 3,000 个测试日的**实际 \(D_t\)** 上评价所有模型和 Oracle 的补货成本；可另保存解析条件期望成本进行 Oracle 数值诊断，但不得与实际实现成本混为同一主表的配对比较。

### 3.2 主指标：Oracle-normalized Prescriptiveness

逐 \(G\) 的正式估计：

\[
\boxed{\widehat{\mathcal P}_{m,G}=
\frac{\bar C_{\rm SAA,G}-\bar C_{m,G}}
{\bar C_{\rm SAA,G}-\bar C_{\rm Oracle,G}}.}
\]

五 \(G\) 的综合主指标：

\[
\boxed{\widehat{\mathcal P}_{m,5G}=
\frac{\sum_{G\in\mathcal G}(\bar C_{\rm SAA,G}-\bar C_{m,G})}
{\sum_{G\in\mathcal G}(\bar C_{\rm SAA,G}-\bar C_{\rm Oracle,G})}.}
\]

\(\bar C\) 的计算顺序：**每重复内先平均随机 Seed → 10 个重复平均成本 → 跨五个 \(G\) 加总成本差 → 最后取比值**。不能先求 10 个重复的 \(\mathcal P\) 再平均，也不能先平均逐 \(G\) 的比值。主文使用五 \(G\) 综合 \(\mathcal P\) 与 \(G=300\) 的 \(\mathcal P\)；逐 \(G\) 全表保留。

\(\mathcal P\) 不裁剪到 \([0,1]\)：有限测试样本、数值误差或算法性能较差可产生越界值。若分母接近零，报告成本分母和数值误差并标注 NA。

### 3.3 两个辅助指标

\[
\boxed{\operatorname{MeanCost}_{m,r,G}=\frac1{3000}\sum_{t=1}^{3000}c_{m,r,G,t},\qquad
SL_{m,r,G}=\frac1{3000}\sum_{t=1}^{3000}\mathbf1\{I_{{\rm end},t}\ge G\}.}
\]

相同 \(G\) 下 \(\mathcal P\) 与 MeanCost 的排名严格反向等价；MeanCost 给出绝对成本量纲，\(\mathcal P\) 给出相对事前信息价值的解释。目标服务率**不等于需求填充率**；高达标率可能与高超目标库存成本并存，不当作越高越好的一维胜负标准。

## 4. P1：八种方法的正式定义

| 方法 | 上游测试输入 | 训练方式 / 0.9 分位数输出 | 主要研究对照 |
|---|---|---|---|
| **DAT** | 完整 \(A^-,X\) | Patient SA + 动态 Day Query CA；Pinball(0.9) | 本研究预测–决策模型 |
| **SAA** | 无 \(A^-,X\) 需求特征；执行补货仍知 \(I,G\) | 无条件经验分位数 | \(\mathcal P=0\) 的基准 |
| **wSAA** | \(\psi\in\mathbb R^{16}\) | RF 叶邻近权重，历史 \(D\) 的加权经验分位数 | Bertsimas–Kallus 风格的条件加权 SAA |
| **Flat-ETO** | 每名旧患者 \([a_i,\psi]\in\mathbb R^{19}\)，残差模型 \(\psi\) | 用**训练期旧患者实际 \(Y_i^-\)** 训练七类 Softmax；对训练期残差 \(R=D-\sum_{i\in A^-}Y_i^-\) 使用 RF 加权经验分布，近似独立卷积并取分位数 | 额外患者级监督、但固定摘要和条件独立卷积可能误设 |
| **Set-ETO** | 完整 \(A^-,X\) | DeepSets Mean Pool + 事前固定 HurdleGamma NLL，提取 0.9 分位数 | 分布族误设与 ETO 路线 |
| **LQR** | \(\psi\in\mathbb R^{16}\) | 预先固定正则的线性分位数回归 | 简单全局线性条件方法 |
| **GBQR** | \(\psi\in\mathbb R^{16}\) | 预先固定的分位数梯度提升树 | 非线性表格方法 |
| **DeepQR** | 完整 \(A^-,X\) | DeepSets Mean Pool + 直接 Pinball(0.9) | 同级输入、直接分位数学习，但无显式 Attention |

**Flat-ETO 的特殊披露**：它在训练时获得逐旧患者发血标签 \(Y_i^-\)，其他主体基线只使用日级总需求标签；这一监督不对称必须在方法表注明。测试阶段 Flat-ETO 不能看到真实 \(Y_i^-\)。残差 \(R\) 同时包含新患者需求、当日冲击及新患者对旧患者发血决策的间接影响，**独立卷积只是近似 ETO 方法，不能充当真实 Oracle**。

**wSAA 的定位**：以随机森林历史相似日权重近似条件需求分布，不是“患者之间的相似度核 \(k_{ij}\)”（本版 DGP 已不使用该核）。保留它作为经典 Predictive-to-Prescriptive 路线的重要对照。

## 5. P2：八个结构变体（3 个参数匹配对照）

| 结构 ID | 单一结构操作 | 识别目的 |
|---|---|---|
| `full` | 完整 DAT：Self-Attention + 动态 Cross-Attention | 研究主结构 |
| `no_self_attention` | 删除患者 Self-Attention 支路，其余全部不变 | SA 的原始边际表现 |
| **`no_self_attention_param_matched`** | 去 SA，但仅通过调整 Day FFN 宽度达到 Full 参数量附近 | 排除纯粹模型参数量差异 |
| `no_day_query` | 将 CA Query 改为**可学习常数向量**，而非由 \(X\) 生成 | 去掉“日级状态决定患者聚合权重”；\(X\) 仍经 Day Linear 残差与 FFN 进入 |
| `mean_pool` | 用有效患者 Memory 的 Mean Pool + 输出投影代替 CA，保留 SA | 静态均值聚合 |
| `mean_pool_param_matched` | 采用相同 Mean Pool 但调隐藏宽度至 Full 参数量附近 | 参数量匹配的静态均值聚合 |
| `sum_pool` | 用有效患者 Memory 的 Sum Pool + 输出投影代替 CA，保留 SA | 静态求和聚合 |
| `sum_pool_param_matched` | 采用相同 Sum Pool 但调隐藏宽度至 Full 参数量附近 | 参数量匹配的静态求和聚合 |

设有效患者 Mask 为 \(m_i\in\{0,1\}\)，\(M_i=\operatorname{LN}_m(E_i)\)；固定池化为

\[
\operatorname{MeanPool}=W_o\left(\frac{\sum_i m_iM_i}{\sum_i m_i}\right),\qquad
\operatorname{SumPool}=W_o\left(\sum_i m_iM_i\right).
\]

- 所有变体使用**完全相同的 \(A^-,X,D\) 权限和训练数据**。No Day Query 不得偷偷删除 \(X\) 的残差路径。
- 参数匹配只根据**实际有效可训练参数数目**事前设定宽度：\(|P_{\rm variant}-P_{\rm Full}|/P_{\rm Full}\le5\%\)，优先 \(\le1\%\)；保留真实参数数目、宽度、初始化 Seed 至 manifest。
- 若仅调整 Day FFN 隐藏宽度 \(H\)，在 \(d=80\) 下，\(P_{\rm FFN}(H)=(2d+1)H+d=161H+80\)，每增 1 宽度加 161 参数。**不可直接复制 v3 旧模型的匹配宽度**，须按 v6 输入维重新核算。
- 参数量匹配并非使模型函数类完全相同。Full 可能因优化/泛化而落后于 No-SA；不允许用测试结果继续调整宽度。

**重点的成对比较**：`full vs no_self_attention`、`full vs no_self_attention_param_matched`；`full vs no_day_query`；`full vs mean_pool_param_matched`、`full vs sum_pool_param_matched`。报告 \(\mathcal P\) 差和 MeanCost 配对差的 CI；不得只用各方法独立 CI 是否重叠来判断显著性。

## 6. P3：四个真实信息共享层级（最终命名 L0–L3）

此实验不改变真实 DGP，也不按信息级重新生成未来患者。只改变**血液中心在补货前可看见的字段**，同一 DAT 从头按固定预算训练。

| 层级 | 血液中心的授权日级信息 | 授权患者 Token | 实际制度解释 |
|---|---|---|---|
| **L0** | \(c_p,c_e,I\)（\(X\) 前三维） | 一枚**固定 Dummy Token** | 无医院患者层数据共享；只有区域环境与库存 |
| **L1** | L0 + \(N^-/8,\sum_{A^-}u_i/8\)（\(X\) 前五维） | 同一固定 Dummy Token | 医院提供基本申请人数与预计用血总量 |
| **L2** | L1 + \(\sum_{A^-}r_i/8,\sum_{A^-}s_i/8\)（\(X\) 全七维） | 同一固定 Dummy Token | 医院进一步共享高风险人数及计划/急诊病例组合汇总 |
| **L3** | 全 \(X\) | **完整提前登记患者 \(A^-\) 的真实 Token + Mask** | 医院与血液中心共享逐患者画像；可学习风险/强度/类型的具体对应关系与优先级竞争 |

**关键权限控制**：L0–L2 绝不能使用真实患者 Token 个数、其 Padding Mask 或位置泄露 \(N^-\)；全部使用同一固定 Dummy Token 和固定有效 Mask（恰好一个有效 Token）。隐藏的 \(X\) 坐标固定为无信息占位值，不能通过归一化参数或 Mask 泄露未授权字段。L3 才使用真实患者集合，且任何层级都看不到 \(A^+\)。

嵌套信息保证**条件 Bayes 最优成本**满足

\[
\boxed{C_G^*(L0)\ge C_G^*(L1)\ge C_G^*(L2)\ge C_G^*(L3).}
\]

**但分别训练的 DAT 不保证成本严格单调**。同一个 Full Oracle \(f^*(A^-,X)\) 用作四个 P 的共同标尺；因此各级 P 体现的是该级网络**实际实现的潜在总体价值比例**，不等于该信息集合自身的 Bayes 价值。如果想精确报告 \(C_G^*(L2)-C_G^*(L3)\)，使用 DGP 独立计算的 L2 条件 Oracle，并作为额外机制证据，不偷换 P 的分母。

> **重要预期边界**：v6 生成机制的初步确证表明 \(L2\to L3\) Bayes 成本增量约 0.0186，为正但相对于 SAA–L3 Oracle 总改善仍较小；不能提前写“L3 必定比 L2 优出 20 个百分点的 P”。

## 7. 正式训练前的 DGP/Oracle 验收

1. **输出与概率**：\(N^-\in[3,8],N^+\in[0,2],Y_i\in\{0,\ldots,6\},D\in[0,61]\)；每名患者七类 PMF 非负且和 1；最终 PMF 非负且和 1。
2. **两级边界**：医院端 \(A^{\rm all}\) 生成 \(D\)，血液中心测试输入与 Oracle 仅使用 \(A^-,X\)；\(D\) 绝不依赖补货政策输出 \(q\)。
3. **45 态隐藏患者**：不使用测试日实际 \(A^+\)，按新增机会概率和八种画像概率赋权；同画像不同患者保留多份随机行动；共同 \(U\) 先卷积再积分。
4. **置换/节点校验**：旧患者排列不改变条件 PMF/分位数；16/25/32 GH 对照，CDF 反演误差；不要用截断/归一化掩盖严重计算错误。
5. **行为方向**：固定患者 \(i\) 和日级环境，加入更高优先级患者后 \(h_i\) 下降；库存下降时一般谨慎和竞争挤出变强。
6. **总体和增量价值**：独立校准池估计 SAA，无训练模型前检验 SAA–Full Oracle 成本差以及 \(L2\to L3\) Bayes 成本差；确认日级机制并非数值上失效，发血选择没有大面积概率饱和。
7. **运营截断**：逐 \(G\) 报告 \(\Pr(q>0)\) 或等价正补货活跃率。由于当前 \(I\in[205,335]\) 且预测需求非负，\(G=340\) 必然触发正补货，较高 \(G\) 的成本可能非常接近；不因此修改预设网格或删掉重复口径。
8. **冻结与可复现**：DGP 参数、模型宽度、树模型超参、学习率、种子、训练迭代预算等必须在正式训练前写入锁定的配置和 Git SHA。已有 v3 结果不得混作 v6 正式结果。

## 8. 统计推断、表格及文件合同

- 独立统计单元为 10 个训练/测试数据重复；**随机 Seed 先在同重复内部平均**。
- 主 \(\widehat{\mathcal P}\) 由十重复平均成本及同一个 SAA–Oracle 分母计算一次；不能把各重复 P 的算术平均冒充主结果。
- 置信区间：事前固定 10,000 次**以数据重复为簇的配对 Bootstrap**；一次抽中一个重复，必须连同该重复所有方法/变体/信息级、五个 \(G\)、SAA 和 Oracle 一起带走。每次在重抽样成本上重新计算 P。
- 在 P1/P2/P3 中完全相同的 Full DAT 结果使用**同一套 Bootstrap 索引**，使 P/CI 保持一致。
- 结果主表列顺序：**PrescriptiveP ↑（95% CI）｜MeanCost ↓（95% CI）｜\(SL_G\)（95% CI）**。主图纵轴 P，另附 MeanCost、SL_G 解释图。
- 报告每个 \(G\) 的 SAA–Oracle 绝对分母以及五 \(G\) 综合分母；分母近零不裁剪不硬算。

最低交付：

```text
axid_dc_v6_manifest.json                # DGP/模型/Oracle SHA、权限、配置和全部随机种子
axid_dc_v6_performance_metrics.csv      # replicate × experiment × method × seed × G
axid_dc_v6_oracle_quantiles.csv         # replicate × test_day: Qstar(A-,X)，不读取实际 A+
axid_dc_v6_oracle_costs.csv             # replicate × G: Oracle MeanCost, Oracle SL_G
axid_dc_v6_prescriptive_p.csv           # experiment × method × G/5G: P/CI/分母
axid_dc_v6_p1_summary.csv               # P1 八模型
axid_dc_v6_p2_summary.csv               # P2 八结构与有效参数总数
axid_dc_v6_p3_summary.csv               # P3 L0/L1/L2/L3
figures/P1_prescriptive_p.png
figures/P2_attention_ablation_p.png
figures/P3_information_sharing_p.png
```

**正式协议声明**：这是患者优先级挤出型 AXID-DC v6。已经完成的是 DGP 的独立机制和 Oracle 数值预检查；**尚未运行 v6 的八方法、八结构和四级信息正式性能实验**。今后的实验结果无论是否符合预想都应如实报告，不应通过重新选择 DGP 系数或最优 Seed 改变结论。
