# 05｜AXID-DC v8：轻量 DAT 增补实验协议（仅新增训练）

> **定位**：对已完成的 `AXID-DC v8` MeanCost/SL 实验开展**事后标记的轻量架构增补**，不覆盖、不重跑原正式实验。目标是检验“较小的 DAT + 两层 Day Encoder”能否改善 Pinball(0.9) 学习、动态 Attention 消融及患者级信息利用。**这不是新 DGP，也不能依据本次正式测试结果反复调整模型并称之为未经选择的确认性实验。**
>
> **唯一 DGP**：`02_AXID_DC_v8_DGP.md`，版本 `axid_dc_v8_hb_context`。**不改任何生成公式、系数、患者集合、测试日期、库存目标或 `τ`。**

## 1. 增补范围：哪些跑、哪些绝对不重跑

复用原 v8 的 **10 个独立数据重复**，每重复原样使用 **2,500 训练日 + 3,000 测试日**；随机方法每重复 **3 个原定训练 Seed**。测试记录、输入归一化、患者 Padding/Mask、`D/I/G` 均保持一致。采用相同的 `τ=0.9`、`G={260,280,300,320,340}`。

**仅新增 6 个配置**，每个配置 `10 repeats × 3 seeds = 30` 次训练，共 **180 次训练**、**900 条按 G 评价的 seed 级指标行**：

| 新增配置 ID | 实验用途 | 唯一变化 / 输入 | 训练次数 |
|---|---|---|---:|
| `lite_full` | P1 新 DAT；P2 Full；P3 L3 | 轻量完整 SA + 动态 CA；完整 `A⁻,X` | 30 |
| `lite_no_sa` | P2 | 移除 SA（及其专属 LN），其余不变 | 30 |
| `lite_no_day_query` | P2 | CA 改为可学习常数 Query，`X` 仍经 Day MLP 残差 | 30 |
| `lite_mean_pool` | P2 | CA 改为有效患者 Memory 均值 + 输出线性映射 | 30 |
| `lite_sum_pool` | P2 | CA 改为有效患者 Memory 求和 + 输出线性映射 | 30 |
| `lite_L2` | P3 | 与 `lite_full` **完全相同的架构**，只开放 L2 汇总；固定 Dummy Token/Mask | 30 |

**全部复用、不训练**：原 v8 的 SAA、wSAA、Flat、Set-ETO、LQR、GBQR、DeepQR；原 v8 Full DAT 和八个原版 P2 变体；原 v8 的 L0、L1、L2、L3；任何现有 Oracle 结果及已完成的种子级预测。三个 `*_param_matched` **不在增补范围内**，不增加训练。`lite_full` 在 P1/P2/P3 **只训练一次、复用相同逐日预测**。

> **若只需最小容量诊断**，先执行 `lite_full` 与 `lite_L2`（共 60 次训练）；其余四个轻量 P2 结构可按本已冻结协议补充。**不能用旧版的 No-SA/Pooling 结果冒充轻量 Full 的受控 P2 消融。** 若已运行的某一任务具有完全相同的配置指纹且产物完整，则直接复用，不重复训练。

## 2. 唯一轻量 DAT 架构（冻结）

继续使用旧 v8 的**患者 3 维** `a_i=(u_i,r_i,Hb_i)` 与**日级 7 维** `X`，每个已知患者集合上限 `N⁻≤8`。**不向模型提供**生成器内部 `ζ,T,\widetilde r,ω,R,U` 或今日患者 `A⁺`。

| 模块 | 固定结构 | 参数量（标准 PyTorch 带 bias） |
|---|---|---:|
| Patient MLP | `Linear(3,32) → ReLU → Linear(32,64)` | 2,240 |
| Self-Attention | 一层 `MHA(d=64, heads=4)`，Pre-LN + residual | 16,640 |
| Day MLP | `Linear(7,16) → ReLU → Linear(16,64)` | 1,216 |
| Cross-Attention | 一层 Day 单 Query → Patient Memory，`MHA(64,heads=4)`，Pre-LN + residual | 16,640 |
| Day FFN | `Linear(64,48) → ReLU → Linear(48,64)` + residual | 6,256 |
| LayerNorm | `LN_sa`、`LN_q`、`LN_m`，各 64 维 | 384 |
| Readout | `Linear(64,1)` → `clip[0,61]` | 65 |
| **Full 合计** | | **43,441** |

张量计算合同：

\[
\begin{aligned}
E^0_i&=\phi_p(a_i),& E&=E^0+\mathrm{MHA}_{SA}(\mathrm{LN}_{sa}(E^0));\\
u&=\phi_d(X),&v&=u+\mathrm{MHA}_{CA}(\mathrm{LN}_q(u),\mathrm{LN}_m(E));\\
u^+&=v+\mathrm{FFN}_{day}(v),&\widehat f&=\operatorname{clip}_{[0,61]}(w^\top u^++b).
\end{aligned}
\]

**固定不变**：单层 SA/CA、4 Heads、无位置编码、无患者 FFN、无 Dropout、无 Day FFN 前 LN、无 Final LN、恰好三处 LN（按对应消融删除无用 LN）。Padding 必须有正确的 key padding mask；患者集合置换后预测不变。

**配置变化边界**：原 v8 主 DAT 依据原协议为约 51,105 参数、单层 Day Linear；本次轻量 DAT 为 **43,441 参数、两层 Day MLP**。因此新旧 Full 的差异同时涉及**隐藏层宽度及 Day Encoder 非线性**；可以报告总效果，**不能把改善独立归因为“参数更少”或“Day MLP”中的某一个因素**。若原工程的实际模型与原协议不同，以原 manifest/源代码确认后如实记载。

## 3. 增补 P2：只做五个轻量结构，不做 Matched

所有轻量 P2 结构使用**完整且相同的 `A⁻,X`**、同一 Day MLP 和同一 Pinball Loss。仅按下表更换指定支路：

| 结构 | 明确定义 | 预计参数（含有效 LN） |
|---|---|---:|
| `lite_full` | SA → 动态 CA | 43,441 |
| `lite_no_sa` | `E=E⁰`；保留动态 CA | 26,673 |
| `lite_no_day_query` | CA Query 用与 `X` 无关的可训练 `q₀∈R⁶⁴`；`u=DayMLP(X)` 仍通过残差进入 Day FFN | 43,505 |
| `lite_mean_pool` | SA 保留；以 `W_o(mean_valid(LN_m(E)))` 替代 CA 输出，`v=u+pool` | 30,833 |
| `lite_sum_pool` | SA 保留；以 `W_o(sum_valid(LN_m(E)))` 替代 CA 输出，`v=u+pool` | 30,833 |

数值为固定实现口径下的理论参数；实际必须用 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 核验并保存。取消参数匹配后，消融比较**同时改变结构和参数数目**，不能声称完全控制了容量。

## 4. 增补 P3：仅补轻量 L2，轻量 L3 复用 Full

保留原始 P3 权限定义，**不新增训练 L0/L1**：

| 信息级 | 输入门控 | 本次操作 |
|---|---|---|
| L0 | `X[:3]`，固定 Dummy Token/Mask | **复用旧 v8，仅作为历史参考** |
| L1 | `X[:5]`，固定 Dummy Token/Mask | **复用旧 v8，仅作为历史参考** |
| **L2** | 完整 7 维 `X`；仅一个固定 Dummy Token（不能泄漏人数或真实患者 Mask） | **训练 `lite_L2`** |
| **L3** | 完整 `X` + 真实过去患者 `A⁻`/真实 Mask | **复用 `lite_full`** |

正式的**轻量架构内部信息价值对照仅限 `lite_L2` vs `lite_full`（L3）**。旧 L0/L1 采用原版 DAT 参数化，不得与新 L2/L3 拼接成“同一架构下的四级 P3 完整梯度”并据此宣称严格控制了容量。若将来要求轻量 L0–L3 四级完整同架构结果，才需另行补跑 lite L0/L1；**不属于本次任务**。

## 5. 训练配置：只改变架构，不动其他实验条件

- 复用原 v8 的 `repeat_id`、`train/test split`、日级输入归一化、患者 Mask、随机种子表、Pinball(0.9)、优化器、学习率、Batch、Epoch、调度器、权重衰减、梯度裁剪及输出初始化规则。**以原 manifest 为唯一数值来源**，本增补不猜测、不临时改训练超参数；若无法恢复原训练配置，先标明不可比并停止该对照。
- 若原 DAT 使用训练样本无条件 0.9 分位数做输出偏置初始化，轻量网络保持相同；若原 DAT 没有，**不得只给轻量 DAT 增加这一优势**。所有轻量消融之间统一初始化规则。
- 每种结构每重复 **3 个预设 Seed 都运行并保留**，不挑最好 Seed；不按当前 v8 测试集反复调宽度、Epoch 或学习率。
- 数据及结果通过配置指纹检查：至少含 `DGP版本+生成参数哈希、repeat_id、train/test日期哈希、seed、模型结构哈希、训练配置哈希`。**仅跳过完全匹配且产物完整的任务**；旧 DAT 的权重不能作为轻量网络的初始化。

## 6. 评价、统计与无需重做的 Oracle

**先保存每个测试日的一个标量预测** `f_hat`（每重复、每 Seed、每结构一次），然后复用同一预测评价五个 `G`：

\[
q_{m,t,G}=[\widehat f_m(Z_t)+G-I_t]_+,
\quad c_{m,t,G}=0.9(G-I_t-q_{m,t,G}+D_t)_++0.1(I_t+q_{m,t,G}-D_t-G)_+.
\]

\[
SL_{m,t,G}=\mathbf1\{I_t+q_{m,t,G}-D_t\ge G\}.
\]

- **主要展示**：五 G 等权平均 MeanCost 与 `G=300` MeanCost；SL_G 辅助，特别报告 `G=340` 覆盖率——本 DGP 中 `I≤335` 且预测非负，所以 `SL_340=Pr(D≤\widehat f)`。
- **对原 v8 的增补对照**：`lite_full` vs 原版 `DAT`、`DeepQR`、`LQR` 等，原模型的结果**直接复用**，不重训。
- **轻量 P2 对照**：每个变体与 `lite_full` 配对，定义 `ΔC = C_variant − C_lite_full`，正数表示 Full 成本更低。
- **轻量 P3 对照**：`ΔC_share=C_lite_L2−C_lite_L3`，正数表示完整患者信息带来额外实现收益。
- 每个重复内部先平均 3 个 Seed，再把 **10 个独立重复**作为配对统计单位；成本差及覆盖率差报告 95% CI（优先配对簇 Bootstrap 或 t 区间，方法预先固定），不得把 30 Seed 当成 30 个独立数据重复。
- 可另保存**诊断量而非新主指标**：训练/测试 Pinball(0.9)、输出 `clip` 到 0/61 的占比、原始需求覆盖率与预测量分布，用于区分过拟合、优化不足、系统性低估。
- **Full Oracle 无需因换架构重新计算。** 原 v8 的真实测试日期和 DGP 不变，`f*(Z)=Q_{0.9}(D|A⁻,X)` 与模型参数无关。若原 v8 的 QMC Oracle 尚未完成，保持 `P=pending`；待一次性完成后，旧模型与全部轻量模型共用**同一每测试日 Oracle**及 SAA 基准，计算 `P`，无需重训。**不得用旧版本离散 Oracle 或不同数据池的 Oracle 作为参照。**

五 G 综合 Prescriptiveness 必须先合并成本差后取比值：

\[
\widehat{\mathcal P}_{m,5G}=
\frac{\sum_G(\bar C_{SAA,G}-\bar C_{m,G})}{\sum_G(\bar C_{SAA,G}-\bar C_{Oracle,G})};
\]

分母须为正；不裁剪负 P 或超过 1 的 P；10 重复配对 Bootstrap 的每次重抽样都**重新计算比值**。

## 7. 最小新增产物与验收清单

只新建增补目录，不改写原 v8 文件：

```text
outputs/axid_v8_lite_supplement/
  manifest.json                # 原 v8 数据/训练配置指纹；lite 架构和种子
  models/                      # 只存 6 个新配置的 checkpoint（按实际需要）
  predictions.*                # repeat × seed × variant × test_day 的 f_hat
  metrics.csv                  # 6 × 10 × 3 × 5 = 900 seed-G 行，MeanCost/SL_G
  paired_differences.csv       # 旧 DAT/DeepQR vs lite_full；轻量 P2；轻量 L2 vs L3
  summary.md                   # P1/P2/P3 增补表，清楚标注原结果来源及 P 状态
```

验收：

1. 新任务清单仅 `6` 配置、`180` 个训练键；P1 其他 7 个方法及原 v8 P2/P3 **零新增训练**。
2. 对新旧结果的每个 `repeat_id`，核实训练/测试数据哈希和患者/日级输入解释一致；患者重排不改变预测。
3. `lite_full` 在 P1/P2/P3 引用同一 checkpoint/逐日预测；四个轻量 P2 消融的结构定义、Dummy/Mask 权限经单元测试核验。
4. 不因轻量架构而触发任何 DGP 重新生成或 Full Oracle 重新计算；连续患者 Oracle 若未完成则明确 `pending`。
5. 补充实验结论标为 **post-hoc exploratory supplement**；原 v8 结果完整保留，不能用轻量版替换原始正式 DAT 成绩或借测试集挑选新主模型。
