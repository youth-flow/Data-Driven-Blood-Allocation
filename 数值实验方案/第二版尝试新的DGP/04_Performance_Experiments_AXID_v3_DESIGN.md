# 04｜AXID-v3 性能实验协议（候选版）：P1、P2、P3

> **状态**：2026-11-07 候选协议；须在新一轮正式运行前冻结生成器和全部模型配置。保留已完成旧 S4 正式结果，不覆盖、不合并。  
> **生成器**：`02_AXID_DGP_v3_DESIGN.md`；**单个完整机制**，不再按 S1–S4 做四套模型训练。  
> **DAT**：保留原单层 Self-Attention、单层 Cross-Attention、Day FFN、残差和输出 Clip 拓扑；为四维患者、九维日级输入更新形状和宽度。  
> **三项正式评价**：`MeanCost`、`SL_G` 和 `PrescriptiveP`；后者由同口径 MeanCost、SAA、完整信息 Bayes Oracle 归一化得到。  
> **计算预算规则**：每个独立重复 2500 日训练 + 3000 日独立测试；10 个独立数据重复；随机拟合方法每重复 3 Seed；不设置正式验证集、不进行自动超参数搜索、不按测试结果挑选初始化或训练轮数。  
> **主决策**：`tau=0.9`，`G={260,280,300,320,340}`，`G=300` 为主展示，其余 G 仍完整报告。改变 G 不重训模型。

## 1. 试验目的与预期模式（不是预定胜负）

| 实验 | 核心解释问题 | 允许提出的理想机制假说 |
|---|---|---|
| P1 九模型 | 在 SAA→Oracle 信息价值空间内，哪些算法挖掘的信息最多？ | 使用条件信息比无条件 SAA 更有机会降低成本；DAT 相对简单模型可能具有更好的数据效率 |
| P2 七结构 | Patient Self-Attention、动态 Day Query、静态池化各起什么作用？ | 局部患者关系及日级条件化选择可能使完整 DAT 在有限训练集更有利 |
| P3 五信息等级 | 汇总和患者级细节分别创造多少运营价值？ | 完整患者级风险对应具有超出日级统计汇总的增量价值；有限样本实际结果不保证严格单调 |

理想的顺序关系是 **SAA MeanCost > Full Oracle MeanCost**，而非先写死 `DAT < DeepQR < ...`。即使最终 DAT 不领先，也不能按测试结果修改生成器或模型后仍称为同一次确认性实验。

## 2. 共同输入、补充决策及三项评价

- 每名患者输入 `a_i=(u_i,v_i,r_i,s_i)`，共 **4** 维；完整日级输入 **9** 维，已包含 `N/M`；性能样本人数上界 **8**。
- 当前机制需求支持 `D∈[0,60.5]`，模型标量预测须 Clip 至 `[0,60.5]`。
- 库存环境候选仍为 `I=315+20c1+Uniform(-70,70)`，五个 G 保持原试验网格；若今后改变 I/G，需要另立版本，但不能以 DAT 排名作为调整依据。

全部模型使用：

\[
\boxed{q_{\tau,G}(Z)=[\hat Q_\tau(Z)+G-I]_+,\qquad I_{\rm end}=I+q_{\tau,G}-D.}
\]

**指标 1：MeanCost（越低越好）**

\[
\boxed{\widehat C_{m,G}=\frac1{3000}\sum_{t=1}^{3000}\left\{\tau[G-I_{{\rm end},t}]_++(1-\tau)[I_{{\rm end},t}-G]_+\right\}.}
\]

**指标 2：SL_G（期末达到目标 G 的服务率，报告百分比）**

\[
\boxed{\widehat{SL}_{m,G}=\frac1{3000}\sum_t\mathbf1\{I_{{\rm end},t}\ge G\}.}
\]

**指标 3：Oracle-normalized PrescriptiveP（越高越好）**

\[
\boxed{\mathcal P_{m,G}=\frac{C_G(\mathrm{SAA})-C_G(m)}{C_G(\mathrm{SAA})-C_G(\mathrm{Oracle})}.}
\]

- Oracle 使用给定完整事前 `(A,X)` 的真实**条件**需求分布（Bayes Oracle），并不是提前知道当天真实 D 的完美先知。
- 对每份独立数据重复，SAA 仍由该重复的 2500 天训练需求估计；Oracle 在该重复的同一测试输入上计算自己的补充决策。所有算法、SAA、Oracle 使用同一测试集及实际需求实现，以便配对。
- 正式展示时先对三个 Seed 在每个数据重复内平均 MeanCost/SL，再在十个独立重复上汇总。`PrescriptiveP` 应以**汇总后的成本差比值**计算，而不是先对每个重复的比值求平均（避免小分母的极端比值）。
- 若有一份数据或一个 G 的估计分母近似零，应记录并报告；不得强行裁剪系数到 `[0,1]`。总体上应独立用高精度 Oracle 校准确认分母非微小。
- 五 G 综合系数必须**先汇总五个 G 的平均成本再取一次比值**：
  \[
  \mathcal P_{m,5G}=\frac{\sum_{G\in\mathcal G}[C_G(\mathrm{SAA})-C_G(m)]}{\sum_{G\in\mathcal G}[C_G(\mathrm{SAA})-C_G(\mathrm{Oracle})]}.
  \]
- 结果的配对置信区间以 **10 个独立数据重复**为统计单位；三个训练 Seed 不是三份独立数据。

## 3. 新的传统方法固定特征摘要（23 维）

完整日级 `X` 为 9 维。为了允许传统方法获得来源相同的事前统计信息但不直接暴露真实机制 `H/peer/omega`，统一预定：

\[
\boxed{\begin{aligned}
\psi(A,X)=[&X,\overline{u^2},\overline{v^2},\overline{uv},\overline{ur},\overline{vr},\\
             &\overline{us},\overline{vs},\overline{rs},
             \min u,\max u,\min v,\max v,\min r,\max r].
\end{aligned}}
\]

维度为 `9 + 5 + 3 + 6 = 23`。所有特征仅由事前 A、X 构造；特征缩放和缺失值规则在运行前固定，不能从 DGP 真实隐含机制计算 `H` 作为专门的捷径。

**公平性说明**：\(\psi\) 与原始患者多重集 A 不一定是信息等价的；传统方法使用人造摘要本身就是信息表示的一部分。DeepQR 与 DAT 的集合直接输入对比、P2 消融提供更强的结构对照。

## 4. P1：九个模型及实现输入

| 方法 | 输入 | 训练监督 | 固定算法 |
|---|---|---|---|
| DAT | 完整 `A[N,4]`, `X[9]`, Mask | 日级总需求 D | 单 SA、单 CA、Day FFN，直接 Pinball 0.9 |
| SAA | 训练日 D | D | 无条件经验分布的 0.9 分位数 |
| wSAA | `psi[23]` | D | 固定随机森林叶邻近加权经验 CDF，再取 0.9 分位数 |
| Rolling-SAA | 预定训练需求窗口 | D | 固定窗口经验 0.9 分位数；DGP 跨日 iid，不启用星期选择 |
| Flat-ETO | 每患者 `[a_i,psi]` **27 维**；剩余需求森林用 `psi[23]` | D + **训练期患者 Y_i** | 七类患者 PMF + 新增连续需求条件经验 CDF + 独立边际卷积，再取 0.9 分位数 |
| Set-ETO | 完整 `A[N,4]`, `X[9]` | D | **DeepSets Mean Pool** + 固定 HurdleGamma 条件分布 NLL |
| LQR | `psi[23]` | D | 固定 L2 正则强度的线性 Pinball 0.9 |
| GBQR | `psi[23]` | D | 固定提升树结构的 Quantile/Pinball 0.9 |
| DeepQR | 完整 `A[N,4]`, `X[9]` | D | **DeepSets Mean Pool** + 直接 Pinball 0.9 |

- DAT/DeepQR/LQR/GBQR 每个 `tau` 直接拟合相应分位数；主实验只做 0.9。SAA、wSAA、Rolling-SAA、Flat-ETO、Set-ETO 可使用一次分布拟合计算多个 tau，但主结果仅 0.9。
- Flat-ETO 的患者结果仍是七类离散值；它只在训练阶段额外访问真实患者 Y_i。新版 `K_new=12` 只增加新增需求的随机订单上界，不改变 Flat-ETO 新增**连续条件经验 CDF**的训练定义。
- Flat-ETO 独立卷积可能在含潜在 U 的 DGP 下发生结构性误设；不得为其提供真实 U 来纠正。
- DeepQR 与 Set-ETO 的患者编码和 Mean Pool 类型保持一致，主差异来自直接分位数预测与参数化条件分布估计。
- 所有训练超参数运行前人为固定、归档；不因新 DGP 或模型排名执行自动搜索。

## 5. P2：七个结构条件

保留五个主条件：

1. `full`：完整 DAT；
2. `no_self_attention`：移除患者 Self-Attention 分支，其余完全保留；
3. `mean_pool`：用患者 Memory 均值池化加可训练线性输出代替动态 Cross-Attention；
4. `sum_pool`：对应的求和池化；
5. `no_day_query`：Cross-Attention Query 是可训练常量；`X` 仍通过 Day Encoder 残差进入 Day FFN。

两个辅助条件为 `mean_pool_param_matched`、`sum_pool_param_matched`，通过**只看稠密参数计数**事先调整宽度，使其与 Full DAT 相差不超过 5%，不能根据测试成本选择宽度。所有结构读取同一完整 A、X，不同时删减信息。固定 DGP，全部新训练和测试数据配对。

**理想识别方向**：有 Self-Attention 可能更易提取 `peer_i`；动态 Day Query 可能更易形成 `omega_i(A,X)`。两个假说须由结果判断；Mean/Sum Pool 与 DeepSets 也可能学好这些函数，并不被机制排除。

## 6. P3：五级嵌套信息

| 等级 | 日级授权 | 患者 Token 授权 | 研究意义 |
|---|---|---|---|
| L0 | `c1,c2,I` | Dummy | 无患者信息 |
| L1 | L0 + N、申请和备血汇总 | Dummy | 基础日级病人数和资源需求 |
| L2 | L1 + 风险均值/二阶矩、业务类型比例 | Dummy | 汇总临床信息 |
| L3 | L2 全部 | 每名患者 `(u_i,v_i,s_i)`，无其 `r_i` | 个体资源特征与类型结构 |
| L4 | 九维完整 X | 完整 `(u_i,v_i,r_i,s_i)` | 精确风险与个体特征的匹配关系 |

L0–L2 使用固定 dummy token/固定有效 Mask，不泄漏人数；L3 真实 Token/Mask 但 `r_i` 维必须固定屏蔽；完整 Day `X` 中未授权坐标被固定常数掩盖，不能内部自动拼接 `N/M`。全部信息等级使用同样 DAT 拓扑与相同分配参数预算，从头独立训练。

从生成机制看，L3 条件于风险人数之后仍不知道高风险患者对应的哪条申请；L4 则能建立 `H_align` 和 `H_peer`。因此 L3→L4 的条件 Bayes 价值可以通过风险排列枚举计算。但有限样本网络的 MeanCost 和 SL 不保证严格逐级单调。

## 7. 统计与图表

- 对同一次独立数据重复，所有方法/消融/信息组共享同一训练与测试样本，Random Seed 在随机模型之间使用预先记录的对应编号。
- 随机模型 3 Seed **先在重复内部汇总**；确定性模型只保留 1 个真实结果；再在 10 个独立重复上汇总 MeanCost 和 SL，并基于配对成本计算 `PrescriptiveP`。
- P1：主 G=300 的 9 模型 `PrescriptiveP` 图、MeanCost 图、SL_G 图；另报告五 G 全部结果与五 G 成本聚合后的 `PrescriptiveP`。
- P2：七结构相对 Full DAT 的配对 `PrescriptiveP`/MeanCost/SL_G；参数量仅为元数据，不是第 4 个评价指标。
- P3：L0–L4 三项同口径指标阶梯图；如单独呈现 Bayes 信息价值，明确标记为**机制诊断**而非实际 DAT 的结果。
- `PrescriptiveP` 使用成本的归一化，不能用 SL_G 代替分子或分母。若某方法成本优于 Oracle 的有限样本估计值，先检查采样误差，不擅自裁剪指标。

## 8. 运行冻结与验收

1. **新 DGP 先做 Oracle 信息价值校准，冻结之后才训练任何模型。**
2. 完整 A 为 4 维、X 为 9 维、传统 ψ 为 23 维、Flat 患者输入为 27 维；旧 3/8/19/22 形状全部废止。
3. 性能 `d_model=80` 时满足四维患者输入的定量逼近充分潜维条件，头数/Value 宽度匹配；每个变体实际参数计数保存为元数据。
4. 保持已确认的 9 模型、7 结构、5 信息、10 重复、随机模型 3 种子、2500/3000 日、无搜索。
5. 评价指标固定为 **PrescriptiveP、MeanCost、SL_G**，每个 G 独立评价，五 G 综合先汇总成本再归一化。
6. 医院真实数据研究完全独立；不传真实标签、权重或历史收货策略到合成模型中。
7. 任何根据新开发实验调校的生成器须另立 DGP 版本，并使用**全新未参与开发的正式测试种子**；旧 S4 和 v2 结果始终保留。

**生成器与 Oracle 的 Python 参考实现：`AXID_v3_reference.py`。**
