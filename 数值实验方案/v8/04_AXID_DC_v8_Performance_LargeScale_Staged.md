# 04｜AXID-DC v8 大规模合成性能实验：分阶段执行协议（P3 → P2 → P1 → Oracle）

> **状态**：2026-11-14 新一轮大规模性能实验的**事先固定执行合同**；取代旧版 `04_AXID_DC_v8_Performance_Protocol.md` 的**训练/测试规模、优化预算、重复次数、阶段顺序和 P 分母**，但**不覆盖或抹除**已完成的旧 v8 实验。新实验应显式标记为 *large-scale follow-up / exploratory after v8 results*，不能冒充未经先前结果影响的首次确认性试验。
>
> **唯一 DGP**：`02_AXID_DC_v8_DGP.md`，版本 `axid_dc_v8_hb_context`。**生成公式、分布、系数、七档选择、信息边界全部不变。** 不使用 v8 Simplified、v7、v7-Final、LiteDAT 或其他版本的 DGP / Oracle。
>
> **唯一 Full DAT**：原始 v8 架构，**51,105 个可训练参数**；禁止改成 43,441 参数的 LiteDAT，禁止改 Day Encoder 为双层 MLP。
>
> **人工作业门槛**：**先完整 P3，再 P2；得到全部 DAT 系列结果后立即停止并向用户汇报；用户明确决定是否继续后，才可运行 P1 的任何非 DAT 方法（含 SAA）；Oracle / Perfect-Foresight P 统一放在最后阶段。** 不能自动越过审批门槛。

---

## 0｜冻结清单：不变与改变的边界

| 项目 | 本次正式口径 |
|---|---|
| DGP | **完全沿用** `02_AXID_DC_v8_DGP.md`；不同日 i.i.d. |
| Full DAT | **原始 v8，51,105 参数**；模型拓扑不变 |
| 可见信息 | 仅过去患者集合 `A⁻` 与日级 `X`；今日患者 `A⁺` 不可见 |
| 训练数据 | **1,024,000** 个独立合成日，所有配置共用同一数据池 |
| 测试数据 | **50,000** 个独立合成日，所有配置共用同一数据池 |
| 训练数据重复 | **1 套；无独立数据重复** |
| 模型随机 Seed | **每配置 1 个 Seed；不运行第二/第三 Seed** |
| Epoch | **1**；每个训练日仅参与一个 Batch，不做重复遍历 |
| Batch Size | **128 日/Batch**，无梯度累积、无尾批 |
| 参数更新次数 | `1,024,000 / 128 = 8,000` 次/神经网络配置 |
| 主目标 | `τ=0.9`；`G={260,280,300,320,340}`；五个 G 复用一次日级预测 |
| P3 | `L0,L1,L2,L3`，4 个 DAT 配置；**先运行** |
| P2 | `Full,No-SA,No-Day-Query,Mean-Pool,Sum-Pool`，Full 复用 L3；**其次** |
| P1 | 原 v8 八种方法；DAT 复用 L3；**非 DAT 方法等待用户放行** |
| P | **最终才计算** Bertsimas–Kallus 型 Perfect-Foresight Prescriptiveness；不要求先算条件 Oracle |
| 其他指标 | MeanCost、`SL_G`、训练 Pinball、`G=340` 覆盖率诊断 |

**无重复的统计边界**：正式结果只代表这一套固定训练数据与初始化得到的模型；允许对 50,000 个共同测试日做配对 Bootstrap，但区间**仅条件于已拟合模型**，不包含训练数据或初始化的不确定性。不能把 50,000 日说成 50,000 次模型训练重复。

**历史对比边界**：旧 v8 2,500 日、Batch 64、120 Epoch、4,800 步、10 重复×3 Seed；新方案同时改变了训练样本、新样本暴露次数、优化预算、随机性口径和部分固定优化设置，所以不能把新旧成本差**纯粹因果归因于数据量**。DAT 主模型架构及 DGP 保持不变。

## 1｜唯一数据、Seed 与归一化合同

### 1.1 生成与划分

1. 只按 v8 DGP 抽样，生成完整原始日记录：`cp, ce, I, A_minus, A_plus, Y_minus, Y_plus, D, U, J`；模型部署输入仅是授权的 `A_minus,X`，医院隐藏字段进入隔离目录，不得出现在 DataLoader 的 `__getitem__` 输入字典中。
2. **训练** `N_train=1,024,000` 天、**测试** `N_test=50,000` 天；两者独立生成，严禁重叠。仅生成**一套**，P3/P2/P1 全部复用。应提前持久化或以可复现的流式文件/分块保存；严禁每个模型临时生成一套不同训练日。
3. 训练数据按冻结的 Shuffle Seed **只随机洗牌一次**，完整遍历 1 Epoch；`drop_last=False`，但样本量可被 128 整除，因此恰好 8,000 个完整 Batch。所有 DAT 配置使用相同顺序；不同配置的模型随机流相互隔离。
4. 为保证可复现，固定如下无符号整数 Seed（每类随机源**分别初始化**，不共享会受执行顺序影响的全局 RNG 流）：

```yaml
seed_dgp_train: 2026111401
seed_dgp_test: 2026111402
seed_train_shuffle: 2026111403
seed_model_init: 2026111404
seed_baseline_forest: 2026111405
seed_test_bootstrap: 2026111406
# 每配置只有一次初始化；保持 seed_model_init 相同数值，独立重建 RNG。
# 若代码需不同 RNG 子流，用稳定的 config_id 哈希生成派生种子并写入 manifest。
```

5. 持久化 `SHA256(train_data)`、`SHA256(test_data)`、DGP 参数与代码版本、每个 config 的初始化/训练配置指纹；**如果完全相同的配置已有完整合格产物才允许跳过**。旧的 2,500 日 v8 与 LiteDAT 不能复用为本次训练结果。

### 1.2 DAT 输入归一化（严格使用 v8 原定义）

每个真实过去患者：

\[
a_i=(u_i,r_i,\mathrm{Hb}_i)\in[0,1]^3,\qquad \overline{\mathrm{Hb}}_i=1-\mathrm{Hb}_i.
\]

日级向量（**已经完成特征归一化**）：

\[
X=\left(\frac{c_p+1}{2},\frac{c_e+1}{2},\frac{I-205}{130},
\frac{N^-}{8},\frac{\sum_{A^-}u_i}{8},
\frac{\sum_{A^-}r_i}{8},\frac{\sum_{A^-}\overline{\mathrm{Hb}}_i}{8}\right)\in[0,1]^7.
\]

- 患者的第三维是 **Hb 水平**，不是 Hb 不足度；`X[6]` 才是过去患者 Hb 不足度的聚合。患者到达标记 `b_i` **不是**额外预测维度。
- 不对 `A⁻` 和 `X` 再做 `StandardScaler`、按天中心化、批内标准化、目标信息编码或利用测试集计算统计量。保留架构内原有的三个 `LayerNorm`，但没有 `BatchNorm`。
- Patient Tensor 固定填充到 8 个 Token；真实 Token 参与注意力和池化，Padding 使用严格的 `key_padding_mask`，不能作为有效患者。患者 Token 的重排不能改变标量预测。
- 输出的需求预测单位仍是 DGP 原始发血量：`D∈[0,61)`；**不做标签 z-score 或除以 61**，损失直接作用于原始需求单位。最终预测按原版架构 `clip[0,61]`。
- P1 的表格摘要 `ψ∈R^{16}` 按旧 v8 定义：7 个 `X` + 3 个患者两两乘积的**逐患者均值** + 6 个极值；不添加真实今日信息，不向 P3 的 L2 泄漏摘要交叉矩/极值。

## 2｜原始 v8 Full DAT：完全冻结架构

| 组件 | 规格 | 参数数 |
|---|---|---:|
| Patient MLP | `Linear(3,64) → ReLU → Linear(64,64)` | 4,416 |
| Self-Attention | 1 层 `MultiheadAttention(d=64, heads=4)`，有效患者 Mask，残差 | 16,640 |
| Day Encoder | **单层** `Linear(7,64)`（**不是** 7→16→64） | 512 |
| Cross-Attention | 1 层 `MHA(d=64, heads=4)`，Day 单 Query、Patient Memory，残差 | 16,640 |
| Day FFN | `Linear(64,96) → ReLU → Linear(96,64)`，残差 | 12,448 |
| LayerNorm | 恰好 `LN_sa, LN_q, LN_m`，每个 64 维 | 384 |
| Readout | `Linear(64,1)`，`clip[0,61]` | 65 |
| **合计** | **原始 v8 Full DAT** | **51,105** |

计算顺序（Pre-LN 在 SA/CA，**Day FFN 前无 LN**）：

```text
E0 = PatientMLP(A_minus)                            # [B, 8, 64]
E  = E0 + SelfAttention(LN_sa(E0), mask=valid_old)
u  = DayLinear(X)                                  # [B, 1, 64]
v  = u + CrossAttention(query=LN_q(u),
                        key_value=LN_m(E), mask=valid_old)
w  = v + DayFFN(v)
f  = clamp(Linear(w), 0, 61)                       # scalar daily quantile
```

**不允许**：LiteDAT、两层 Day Encoder、额外位置编码、患者 FFN、第四个 LN、Final LN、附加注意力层、改变 64 维或 4 头、额外 dropout、根据测试反馈改变 FFN 宽度。对模型实际执行：

```python
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert n_params == 51_105
```

`P1: DAT`、`P2: Full`、`P3: L3` 引用**同一**模型权重与 50,000 个逐日预测；**只训练一次**。

## 3｜冻结训练超参数（P3/P2 全部 DAT 统一）

**训练预算采用“先 Batch、再固定更新步数、最后确定训练日”的口径**：128 个完整医院日/步 × 8,000 步 = 1,024,000 个互不重复的训练日；1 Epoch。不能因为学习曲线未收敛而自动追加步数，也不能按正式测试成本做超参数搜索。

| 参数 | 固定值 | 说明 |
|---|---|---|
| Optimizer | `AdamW` | PyTorch 语义 |
| `betas` | `(0.9, 0.999)` | — |
| `eps` | `1e-8` | — |
| Peak LR | `3e-4` | 所有 DAT 变体一致 |
| Warm-up | **300 steps** | 从 `3e-5` 线性升至 `3e-4` |
| LR schedule | Cosine decay | 后 7,700 步，降至 `3e-5` |
| Weight decay | `1e-4` | AdamW 解耦衰减，所有可训练 Linear/MHA 权重；Bias/LN 不衰减 |
| Batch | **128 个日级样本** | 无梯度累积 |
| Epoch | **1** | 训练日不重复使用 |
| Updates | **8,000** | `optimizer.step()` 每 Batch 一次 |
| Gradient clipping | Global L2 norm **1.0** | 更新前统一裁剪 |
| Dropout | **0** | 架构不加随机 dropout |
| Loss | 日级 `Pinball(τ=0.9)` | `mean(0.9*(D−f)_+ + 0.1*(f−D)_+)` |
| 初始输出层 | `weight ~ Normal(0,0.01)`；`bias = q90_train` | `q90_train` 是同一训练集 `D` 的**无条件经验** 0.9 分位数；P3/P2 完全一致；不能读取测试集 |
| 其余层初始化 | PyTorch Linear/MHA/LN 默认初始化 | 运行前记录 framework 版本 |
| 精度 | FP32（损失与梯度为 FP32） | 不在不同配置间静默切换 AMP/TF32 |
| 早停 | **无** | 不用测试集选 epoch、seed 或 checkpoint |
| 产出权重 | **第 8,000 步最终权重** | 持久化最后步 Checkpoint |

在第 `500/1000/2000/4000/6000/8000` 步**只记录日志/中间 checkpoint（不用于选择最终模型）**。记录训练 Pinball 移动平均、全局梯度范数、预测值的 `0/61` 裁剪占比、非有限值检查。每个配置的每一步遵守：`zero_grad → forward → mean_loss → backward → clip_grad_norm_ → optimizer.step → scheduler.step`。`scheduler` 与 Warm-up 的准确函数、Step 索引需写入代码和 manifest；以第 8,000 次更新后的参数为最终模型。

**风险提示**：这种输出偏置初始化是**本次新训练配置**，不是对旧 v8 原训练结果的复刻；它保证一开始预测落在可微区域的可能性更高，但不保证最终最优或模型排名。所有 DAT 与直接 Pinball 的 P1 神经基线均使用训练集的同一 `q90_train` 初始化输出偏置，避免只给 DAT 提供该便利。

## 4｜严格执行顺序与人工审批门槛

### 阶段 0：数据与共同验收（只生成一次）

- 生成/冻结一个 `train=1,024,000`、一个 `test=50,000` 的 v8 数据池。
- 验收 v8 病人七档 PMF 归一、同日一个共享 `U`、总需求范围、输入授权、Padding Mask、置换不变性；审计 Full 51,105 参数及参数/输入摘要指纹。
- 此阶段**不得**运行 SAA/wSAA/Flat/Set-ETO/LQR/GBQR/DeepQR，不计算任何 Oracle/P。

### 阶段 1：**P3 信息共享，严格先运行**

| 训练顺序 | ID | 日级权限 | 患者 Token | 独立训练次数 |
|---:|---|---|---|---:|
| 1 | `L3_full` | 全部 7 维 `X` | 全部真实过去患者 `A⁻` | 1 |
| 2 | `L0` | `X[:3]` | 固定的 1 个有效 Dummy Token，固定 Mask | 1 |
| 3 | `L1` | `X[:5]` | 同一 Dummy/Mask | 1 |
| 4 | `L2` | 完整 7 维 `X` | 同一 Dummy/Mask | 1 |

- L0/L1 未授权的 `X` 坐标用**固定零常数**占位；Dummy Token 可取固定零向量，**不是**当日过去患者的统计值。L0–L2 患者输入固定为恰好 1 个有效 Dummy + 7 个固定填充位，Mask 不依赖 `N⁻`；不得通过输入长度、Padding、患者数量、真实病人排序泄漏授权外信息。
- L2 **只**拥有 v8 原 `X` 汇总的 7 维，不接收 P1 的 16 维 `ψ`（尤其交叉矩和极值）。
- 四组均使用**相同完整 DAT 模块配置与 51,105 参数位置**，分别从头训练（L3 作为 P2/P1 Full 复用）。Dummy 条件下部分 Attention 可能退化，须在解释中披露；不可因退化改网络宽度。
- 阶段完成后，对固定 50,000 测试日计算 **MeanCost、SL_G、测试 Pinball、G=340 覆盖率**，五个 G 共用每模型的一次预测。另报 `ΔC_{L2−L3}=C(L2)−C(L3)`（正值表示 L3 更优）；**不提前算 PF/Oracle/P**。
- 保存 P3 小结及全部模型权重、逐日预测；**继续执行 P2，而不是自动运行 P1 基线**。

### 阶段 2：**P2 DAT Attention 消融，其次运行**

| 训练顺序 | ID | 唯一修改 | 独立训练次数 |
|---:|---|---|---:|
| — | `full` | **直接复用 P3 的 `L3_full`** | **0** |
| 5 | `no_self_attention` | 移除 SA 分支及其专属 LN，保留动态 Day CA | 1 |
| 6 | `no_day_query` | CA Query 改为与 `X` 无关的可训练 64 维常数；`DayLinear(X)` 仍经残差进入后续 FFN | 1 |
| 7 | `mean_pool` | SA 保留；以**有效患者** Memory 均值 + 可训练线性投影替代 CA 输出；`X` 仍走 Day Linear 残差 | 1 |
| 8 | `sum_pool` | 同上，但改为**有效患者** Memory 求和；对非有效 Token 不求和 | 1 |

- 五种结构均可用完整 `A⁻,X`，相同 Batch/Steps/Seed/Pinball 与测试集；No-Day-Query **没有删除日级输入**，不能混淆结构与信息消融。
- 取消 `*_param_matched` 三种配置；参数数目**不匹配**，所见差异包括结构和容量变化，必须报告每个变体的实际可训练参数数，不能声称完全排除了容量解释。
- P2 计算 MeanCost、SL、G=340 覆盖率，以及 `ΔC_{ablation−full}`（正值 Full 更优）。无 Oracle/P。

### **硬性停机点 H1：DAT 完成后等待用户人工决定**

**阶段 1 + 2 合计恰好 8 次神经网络训练（4 个 P3 + 4 个 P2）；8×8,000 = 64,000 次参数更新。** Full 在三组中只训练一次。

必须向用户提供并**停止**：

- P3 的 `L0/L1/L2/L3` 五 G 成本/服务率、`L2−L3` 配对差与覆盖率；
- P2 的 `Full/No-SA/No-Day-Query/Mean/Sum` 五 G 成本/服务率、四项相对 Full 的配对差与覆盖率；
- Full DAT 训练 8,000 步曲线/最终测试预测分布、参数量及数据/模型指纹；
- **状态栏：**`P3=done, P2=done, P1 baselines=NOT RUN, PF/Oracle/P=NOT RUN`。

**在用户明确回复批准继续之前，禁止启动 P1 中任何非 DAT 方法，禁止计算 SAA、Perfect Foresight 或条件 Oracle。** 若用户决定停止，则完整保留此阶段结果，不能自动补跑或调整 DGP。

### 阶段 3：**P1 八方法；仅收到用户明确授权后执行**

| ID | 输入与目标 | 执行说明 |
|---|---|---|
| `DAT` | 完整 `A⁻,X`，Pinball(.9) | **直接复用** `L3_full`，不训练 |
| `SAA` | 训练日 `D`，经验 0.9 分位数 | 不训练神经网络；**只在获批后运行** |
| `wSAA` | 16 维 `ψ`，森林叶邻近权重 + 加权 .9 分位数 | 冻结森林超参，使用稀疏叶倒排索引，严禁创建 `50,000×1,024,000` 的稠密权重矩阵 |
| `Flat` | 患者 3 维、日级 7 维，两个独立 MSE 回归 | 仅此方法可访问训练期 `Y⁻` 与 `D_rem=D−ΣY⁻`；不做分位数后校准 |
| `Set-ETO` | 患者集合 + `X` → HurdleGamma 条件分布，NLL | 使用原 v8 定义的分布/NLL 与 0.9 分位数反演；记录具体参数化，不以测试结果调整 |
| `LQR` | 16 维 `ψ`，正则化线性 .9 Pinball | 截距不惩罚 |
| `GBQR` | 16 维 `ψ`，分位数梯度提升树 | 只用训练日，禁用基于正式测试的早停 |
| `DeepQR` | 患者集合 Mean Pool + 融合 MLP，日级 .9 Pinball | 无 SA/CA；与 DAT 使用相同的日级监督和训练预算 |

**P1 方法身份和训练标签权限完全沿用原版 v8。** Flat 包含两套独立回归器，患者回归器和剩余需求回归器**各进行 8,000 次优化更新**（两者合计 16,000 次 `optimizer.step()`），不能把两个模型合计误记为只训练 8,000 步。 P1 基线的最终实现/优化器超参须在**阶段 0 就写入 manifest 并冻结**，不能在看到 H1 DAT 排名后为任何基线挑选参数。建议从已确认的 v8 原工程配置直接导出，并仅在大规模数据造成计算不可行时采用事先登记的等价流式/稀疏实现。如下是本协议给出的单一可执行默认值（若原工程的基线结构/分布参数化不同，须在阶段 0 明确记录实际值，不能后改）：

| P1 方法 | 固定默认实现/超参数 |
|---|---|
| `SAA` | `D` 排序，第 `ceil(0.9*n)` 个顺序统计量，经验 CDF 左端分位数 |
| `DeepQR` | Patient MLP `3→64→64`、有效患者 Mean Pool、融合 MLP `71→96→1`；AdamW、Batch128、1 Epoch/8,000 Steps、LR 与 DAT 同表；输出偏置 `q90_train` |
| `Flat` | 患者回归 `3→32→1`、日级剩余需求回归 `7→32→1`；各自 AdamW、Batch128 天、8,000 Steps、MSE；两个输出分别裁剪 `0..6` 与 `0..13`，最终相加再裁剪 `0..61` |
| `LQR` | 16 维 `ψ` 加截距，平均 Pinball(.9) + `1e-4/2*||β||²`（不惩罚截距）；AdamW，Batch128，8,000 Steps，Peak LR `3e-4`（优化器本身 `weight_decay=0`，避免双重惩罚） |
| `GBQR` | Histogram Gradient Boosting；`loss=quantile, quantile=.9, learning_rate=.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=128, l2_regularization=1e-3, early_stopping=False` |
| `wSAA` | Random Forest：`n_estimators=128, max_depth=16, min_samples_leaf=128, max_features=0.75, bootstrap=True, max_samples=0.5`；森林叶邻近权重需规范化并据此计算经验加权分位数 |
| `Set-ETO` | Mean-Pool DeepSets 主干（默认 `3→64→64`, `71→96`）+ 原 v8 **HurdleGamma NLL 与反演**；AdamW、Batch128、8,000 Steps、Peak LR `3e-4`；形状/尺度参数必须有正值约束、NLL 数值稳定；如原工程缺少可确定的 HurdleGamma 定义，**该方法标记 `BLOCKED`，不得编造分布或偷换成 Gamma** |

**特别说明**：上表对基线的架构宽度/算法参数是本次新大样本协议的**运行前默认设定**，不宣称是旧 v8 已实际运行的配置；如需保留旧工程方法身份，应在阶段 0 的 manifest 中优先锁定原工程配置，且一旦进入 H1，不得再以结果为依据替换。非 DAT 的计算时间随数据量可能大幅增加（尤其 wSAA/GBQR），这只影响部署实现与资源预算，不允许暗中改变训练样本数来改善名次。

阶段 3 结束后，P1 输出原始 MeanCost/SL（五 G）并保存全部预测；**仍不得提前计算 Oracle/P**。

### 阶段 4：**最后计算 Perfect Foresight / Prescriptiveness**

本轮性能主 P 改用 **事后 Perfect Foresight**，而不再把昂贵的事前 Full Conditional Oracle 作为 P 的分母。这是**绩效定义变更，不是 DGP 变更**。

补货量必须非负，不能将 Perfect-Foresight 成本直接设为 0：

\[
\boxed{q_{t,G}^{PF}=[G+D_t-I_t]_+,\quad
 c_{t,G}^{PF}=0.1(I_t-G-D_t)_+.}
\]

对任何已完成模型：

\[
q_{m,t,G}=[\widehat f_m(Z_t)+G-I_t]_+,\qquad
c_{m,t,G}=0.9(G-I_t-q_{m,t,G}+D_t)_++0.1(I_t+q_{m,t,G}-D_t-G)_+.
\]

\[
\boxed{\widehat P_G^{PF}(m)=
\frac{\overline C_G(SAA)-\overline C_G(m)}
{\overline C_G(SAA)-\overline C_G(PF)}.}
\]

\[
\boxed{\widehat P_{5G}^{PF}(m)=
\frac{\sum_{G\in\mathcal G}[\overline C_G(SAA)-\overline C_G(m)]}
{\sum_{G\in\mathcal G}[\overline C_G(SAA)-\overline C_G(PF)]}.}
\]

`C(PF)` 在 `G=340` 的确恒为 0（因为 `I≤335`、`D≥0`）；对 `G=260/280/300/320` **可能大于 0**。不能把各 G 的 P 先分别算完再平均；不可裁剪负 P、超过 1 的 P；分母非正/接近 0 时标记不可解释并报告成本。

**所谓「Oracle 最后」有两层含义**：① **本性能协议最后**执行上述计算极轻的 `PF`（事后最优参照），不在 P3/P2/P1 前运行；② `f*(Z)=Q_.9(D|A⁻,X)` 的**事前 Conditional Oracle**是另外一种更贵的理论/辅助评价工作，不在本次 P1–P3 必跑清单。若以后需要，它也只能在阶段 4 后由用户**另行授权**计算，并单独标记 `Oracle-normalized P`，绝不可与 `PF P` 混列。

## 5｜评价与统计实现

所有配置在同一个 50,000 日测试集中，每日只预测一次 `\hat f_m(Z_t)`，五个 G 共用预测。对模型/信息等级/消融分别保存：

\[
\widehat C_G(m)=\frac1{50000}\sum_{t=1}^{50000}c_{m,t,G},\qquad
\widehat{SL}_G(m)=\frac1{50000}\sum_t\mathbf1\{I_t+q_{m,t,G}-D_t\ge G\}.
\]

本 v8 因 `I≤335`，`G=340` 时没有补货截断（非负预测），故：

\[
\boxed{C_{340}(m)=\mathbb E\rho_{.9}(D-\hat f_m(Z)),\quad SL_{340}(m)=\Pr(D\le\hat f_m(Z)).}
\]

这两者是**校准与决策的诊断**，不是模型训练后的选择标准。直接量化 `L2−L3`、`ablation−Full` 的**同日配对成本差**，正数表示右侧（L3 或 Full）更优。

如果报告置信区间，对固定拟合模型采用 `B_boot=2000` 次**日级配对 Bootstrap**；同一 Bootstrap 索引同时重抽全部方法及五个 G，报告成本差和 PF P 的 95% 百分位区间。**不得把这一条件测试区间写成对训练随机性的保证。** 所有排序、费用与预测均无选择性删改。

## 6｜最少文件产物及精确运行状态

```text
outputs/axid_v8_large_scale/
  manifest.json                    # DGP/模型/训练配置/seed/代码SHA/版本指纹；审批状态
  data/train.*                     # 1,024,000独立日，真实患者可见字段 + 独立训练标签
  data/test.*                      # 50,000独立日；隐藏信息不可进入模型
  checkpoints/p3_l3_full.*         # 首次训练DAT Full，并供P2、P1复用
  checkpoints/p3_l0.*
  checkpoints/p3_l1.*
  checkpoints/p3_l2.*
  checkpoints/p2_no_sa.*
  checkpoints/p2_no_day_query.*
  checkpoints/p2_mean_pool.*
  checkpoints/p2_sum_pool.*
  predictions/dat_family.*        # test_day × config 的预测；Full不可重复算出不同结果
  metrics/p3_mean_cost_sl.csv
  metrics/p2_mean_cost_sl.csv
  reports/H1_DAT_only_report.md    # 人工审批前必交付
  APPROVAL_PENDING                # 用户尚未明确同意继续时保持该状态
  # 以下只在H1获批后出现：
  checkpoints/p1_non_dat/...
  predictions/p1_non_dat.*
  metrics/p1_mean_cost_sl.csv
  # 以下只在全部P1完成后最后生成：
  metrics/perfect_foresight_cost.csv
  metrics/prescriptiveness_pf.csv
  reports/FINAL_P1_P2_P3.md
```

**工作流状态机**：

```text
DATA_VALIDATED
    ↓
P3_DONE → P2_DONE → H1_APPROVAL_PENDING
                         ├─ 用户选择停止 → STOP_AND_ARCHIVE
                         └─ 用户明确同意 → P1_BASELINES_DONE
                                             ↓
                                      PF_POSTPROCESS_DONE
                                             ↓
                                           FINAL
# 事前条件Oracle并不自动启动；需单独授权。
```

**验收断言**：`Full DAT 参数=51,105`；`训练日=1,024,000`；`测试日=50,000`；每个神经配置 `8,000 optimizer.step`；`P3+P2 DAT=8 次训练`；`P2 Full=P3 L3=P1 DAT` 权重文件哈希相同；L0–L2 固定 Dummy/Mask 无人数泄漏；正式测试只跑一次前向、五 G 不重复拟合；H1 前 `P1_non_DAT=NOT_RUN` 且 `PF/Oracle=NOT_RUN`。

---

## 7｜解释边界与禁止事项

1. **不以研究假说为训练/数据调参规则**：即使 P3 L3 输给 L2、P2 Full 输给 No-Day-Query、P1 DAT 输给 DeepQR，仍如实保留本次数据、训练和结果；不得根据表现修改 DGP 系数或自动二次追加训练。
2. **取消重复不是验证无随机性**：单 Seed 的结果反映给定一次训练实现；日级 Bootstrap 不是跨训练过程的稳定性证据。若未来有新的独立重复实验，应明确标为新实验，不能混入本合同。
3. **不把 P2 结构差写成独立因果作用的全部解释**：取消参数匹配后，消融同时可能改变模型容量；只有在解释中披露实际参数差异才严谨。
4. **不把 P3 有限样本结果当成 Bayes 信息价值**：即使 L3 测试成本比 L2 低，也只能说明训练算法实现的信息收益，不证明两信息集条件 Oracle 的确切成本差。
5. **不偷看未来数据**：仅真实过去患者可以进入模型；今日患者、医生中间效用/风险、共同 `U`、实际实现 `D` 仅作为生成/标签/事后评价材料。所有模型训练/测试输入授权须审计。
6. **旧 v8 与本新实验分别存档**：旧 `AXID_v8_cost_sl.xlsx` 和 LiteDAT 增补结果不能当作本次 102.4 万训练日的结果；未经实际执行不声称结果已经改善。
