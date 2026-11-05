# 04｜DAT Synthetic 性能实验正式协议：九模型、消融与信息价值（S4 Only）

> **版本**：2026-11-05 冻结版。  
> **唯一主 DGP**：**S4**；不运行 S1–S3 的正式性能实验，不做跨场景结论。  
> **唯一主分位数**：**`tau=0.9`**。可事先另设辅助 `tau`，必须用独立结果文件和图说明“辅助”，不改变主排名。  
> **训练设置**：**所有九个模型及 P2/P3 变体均只用一套事先人为确定的训练/算法超参数，不做超参搜索**。为避免误读，本 MD **不列具体学习率、Batch、Epoch、树数量、窗口长度、正则强度等数值**；它们必须在正式运行配置中冻结并随结果归档。  
> **随机设计**：10 份独立数据重复；对有随机拟合步骤的方法，每份数据使用 3 个训练/算法随机种子，**不能按测试指标挑选最佳种子**。  
> **最终性能评价仅两项**：`MeanCost`（每测试日**平均**运营成本）和 `SL_G`（期末库存达到目标 G 的服务率）；不报告总成本、需求满足率、Pinball、预测覆盖率或运行时间作为第三个性能指标。  
> **实验范围**：仅 Synthetic；医院实际数据研究完全独立，不生成历史真实收货策略。

---

## 1. 研究对象与完整运营流程

一个决策日的事前信息：

\[
A=\{a_i=(u_i,v_i,r_i)\}_{i=1}^N,\quad 1\le N\le M=8,\qquad X\in[0,1]^8.
\]

已知期初库存 `I` 也是 `X` 的一个坐标。需求 `D∈[0,B]` 在补货决策之后实现，`B=36.5`。

S4 固定生成系数：

\[
\boxed{(\lambda_{\rm ctx},\lambda_{\rm peer},\rho)=(1,1.5,0.8).}
\]

各方法给出对真实需求的 `tau=0.9` 分位数预测 `Qhat(Z)`，随后都执行**完全相同**的政策：

\[
\boxed{q_{\tau,G}(Z)=[\widehat Q_\tau(Z)+G-I]_+.}
\]

期末库存：

\[
\boxed{I_{\rm end}=I+q_{\tau,G}-D.}
\]

固定目标库存集合：

\[
\boxed{G\in\{260,280,300,320,340\}.}
\]

主图可使用预先选定的中间目标 `G=300` 作展示；完整五个 G 的数据仍应计算和存储。**改变 G 不重训模型、不重新生成 DGP。**

主 `tau=0.9` 对应归一化缺口与超目标成本权重 `b=0.9,h=0.1`。辅助 `tau` 可以使用同一生成 DGP 和相同 G 网格，但直接学习分位数的模型必须针对辅助分位数另行拟合；**不能把一套 0.9 分位数模型当作另一分位数模型使用**。

## 2. 唯一两个性能指标

### 2.1 日均运营成本 MeanCost ↓

日级成本为：

\[
\boxed{c_{\tau,G,t}=\tau[G-I_{\rm end,t}]_+ +(1-\tau)[I_{\rm end,t}-G]_+.}
\]

一个独立测试集有 `n_test=3,000` 个决策日：

\[
\boxed{\widehat{\mathrm{MeanCost}}_{\tau,G}
=\frac1{3000}\sum_{t=1}^{3000}c_{\tau,G,t}.}
\]

这就是**每天平均成本**，不是 3000 天成本之和。数值越小越好。

### 2.2 目标库存服务率 SL_G ↑

\[
\boxed{\mathrm{SL}_G=\Pr(I_{\rm end}\ge G).}
\]

测试集估计：

\[
\boxed{\widehat{\mathrm{SL}}_G
=\frac1{3000}\sum_{t=1}^{3000}\mathbf1\{I_t+q_t-D_t\ge G\}.}
\]

服务事件是**需求实现后，期末库存达到或高于目标 G**，不是患者需求完全满足率，也不是分位数覆盖率。由于：

\[
I+q-G=\max\{\widehat Q_\tau(Z),I-G\},
\]

有：

\[
I_{\rm end}\ge G\iff D\le\max\{\widehat Q_\tau(Z),I-G\}.
\]

服务率越高不一定意味着成本越低；过量补货可能提高 `SL_G`，同时增加高于 G 的库存成本。**不能只按服务率给方法排序。**

**最终性能表的两个数值列只能为 `MeanCost`、`SL_G`**；均值、区间及配对差的统计呈现属于这两项指标的汇总，不构成第三个评价维度。

---

## 3. 数据划分、重复、训练与固定配置

### 3.1 不做调参时的正式数据拆分

每次独立重复，从同一个 S4 DGP 独立生成 `5,500` 个样本：

| 划分 | 决策日数量 | 使用权限 |
|---|---:|---|
| **训练集** | **2,500** | 拟合九种基准、消融和信息级别；Flat-ETO 额外读取授权的训练患者 `Y_i` |
| **独立测试集** | **3,000** | 仅按已冻结模型计算 `MeanCost`、`SL_G` |

以前的 `2,000 训练 + 500 验证` 由于**不再进行超参数候选或验证轮次选择**，在本协议中直接作为一个 `2,500` 天正式拟合块；**不再保留正式的验证集选优步骤**。

DGP 跨日 i.i.d.；所有方法在同一独立重复内使用**同一训练集合与同一测试集合**，测试实现需求相同。测试日不得被任何模型用于参数更新、在线 SAA 更新或选择超参数/种子。

### 3.2 10 次独立数据重复与 3 个训练种子

- `replicate_id=1,...,10`：每次重新生成独立的 2,500 日训练数据和 3,000 日测试数据。这是统计推断中的**独立重复单元**。
- 在相同 `replicate_id` 下，所有方法/变体/信息级别共用同一测试日，从而形成**配对比较**。
- 对有随机性的估计器（神经网络初始化、随机森林抽样等），每个重复使用 **3 个事先固定的训练或算法种子**，分别得到完整的测试结果。
- 对 SAA、固定窗口 Rolling-SAA、确定性优化的 LQR 等无随机训练步骤的方法，只保存**一个真实结果**；不能复制为三个虚假的随机训练结果。
- 每份数据**先平均**其三个随机种子，再对十份独立数据计算跨重复统计；不能将 `10×3=30` 伪装成 30 个独立数据样本。
- 不按 `MeanCost`、`SL_G` 或测试日拟合情况挑选“最好种子”。

### 3.3 完全取消超参数候选搜索

所有方法只有运行前指定的**唯一配置**，存放在版本化配置文件，例如：

```yaml
protocol_version: s4_primary_tau09_fixed
scenario: S4
primary_tau: 0.9
G_grid: [260, 280, 300, 320, 340]
n_train: 2500
n_test: 3000
n_replicates: 10
n_stochastic_seeds: 3
hyperparameter_search: false
# 具体模型结构/训练超参数在各模型的人工固定子配置中填写，不在此文档列数值
```

以下配置项必须由工程负责人事先填写并冻结，但不需要在本 MD 写具体数值：模型 MLP 隐藏宽度、注意力头数、优化器及学习率、Batch Size、Epoch、梯度裁剪、数值稳定项、随机森林树数/叶大小/特征比例、Rolling 窗口长度、LQR 正则强度及凸优化容差、GBQR 树与提升设置、参数匹配变体宽度。

不得执行网格搜索、随机搜索、贝叶斯优化或通过验证/测试选择概率分布家族。固定轮数训练不因某个模型表现较差而临时增加 Epoch。若需要改配置，应创建独立的新运行版本并重跑所有受影响方法。

### 3.4 模型预测与 G 复用

| 方法类型 | 主 `tau=0.9` | 辅助 `tau` 如有 | G 变化 |
|---|---|---|---|
| DAT、DeepQR、LQR、GBQR | 训练一个直接分位数模型 | 另行独立拟合对应分位数 | **只重新计算 q 与指标** |
| SAA、wSAA、Rolling-SAA | 先拟合经验或加权条件 CDF | 从同一 CDF 取其他分位数 | **只重新计算 q 与指标** |
| Flat-ETO、Set-ETO | 先拟合需求条件分布 | 从同一估计分布取其他分位数 | **只重新计算 q 与指标** |

---

## 4. 通用输入：完整 A/X 与 19 维传统摘要

### 4.1 全部事前可见信息

\[
A=\{(u_i,v_i,r_i)\}_{i=1}^{N},\qquad
X=\left(\frac{c_1+1}{2},\frac{c_2+1}{2},\frac{I-225}{180},\frac NM,
\frac{\sum_i u_i}{M},\frac{\sum_i v_i}{M},\bar r,\overline{r^2}\right).
\]

DeepSets 和 DAT 使用完整的无序患者集合 `A` 及全部八维 `X`；无法直接处理变长集合的传统方法统一使用下面的固定维、**仅事前信息**摘要。

### 4.2 已确认的 19 维摘要 psi(A,X)

定义：

\[
\boxed{\psi(A,X)=\left[
X,\overline{u^2},\overline{v^2},\overline{uv},\overline{ur},\overline{vr},
\min u,\max u,\min v,\max v,\min r,\max r
\right]\in\mathbb R^{19}.}
\]

其中 `bar(g)=N^{-1} sum_i g(a_i)`。

| 组件 | 维数 | 内容 |
|---|---:|---|
| 完整日级状态 | 8 | 计划、急诊、库存、人数、两个申请摘要、风险一/二阶矩 |
| 二阶与交叉矩 | 5 | `mean(u²), mean(v²), mean(uv), mean(ur), mean(vr)` |
| 极值 | 6 | `min/max(u), min/max(v), min/max(r)` |
| **总数** | **19** | 全部来自事前可见 `A,X` |

`psi` 不包含真实 DGP 的同群核公式、潜变量 `U` 或事后结果。它是 wSAA、LQR、GBQR 以及 Flat-ETO 的新增需求估计器共同使用的**唯一固定特征函数**，不根据测试性能增删摘要。

**公平性解释**：传统方法拥有构造 `psi` 所需的同一份原始事前信息，但 19 维摘要不一定完整保存 `A` 的全部细节。DAT/DeepQR/Set-ETO 可直接处理集合；相关优势可能同时包含信息表示和归纳偏置差异。P2 的结构消融是更强的架构对照。

---

## 5. P1：九种基准方法的逐项可实现算法

本节是**九个固定算法**的正式定义，任何未列出的超参数都只在人工冻结配置中指定，不引入候选搜索。

### M1. DAT：Decision-Aware Transformer

**预测输入**：完整 `A`、八维 `X`、有效患者 Mask。无患者位置编码。

**网络**：共享患者两仿射层 ReLU → 单层 Self-Attention（Pre-LN + residual）→ 日级仿射编码 → 单 Query 的 Cross-Attention（独立 Q/Memory LN + residual）→ 日级 ReLU FFN + residual → 线性标量输出、Clip `[0,B]`。**恰好三处 LayerNorm**，没有患者 FFN、日级 FFN 前 LN 或 Final LN。精确结构见 `01_DAT_Architecture_S4.md`。

**训练数据与目标**：仅使用日级 `D`，最小化：

\[
\mathcal L_{\rm DAT}=\frac1T\sum_t\rho_{0.9}(D_t-f_\theta(A_t,X_t)).
\]

**预测**：`Qhat=f_theta(A,X)`，然后统一部署策略 `q=[Qhat+G-I]_+`。

### M2. SAA：Sample Average Approximation

**预测输入**：不利用当日 `A,X` 的需求异质性；仍然知道 `I,G` 用于决策。

训练日 `D_1,...,D_T` 构造无条件经验 CDF：

\[
\boxed{\widehat F_{\rm SAA}(d)=\frac1T\sum_{t=1}^T\mathbf1\{D_t\le d\}.}
\]

预测经验 0.9 分位数：

\[
\boxed{\widehat Q_{0.9}=\inf\{d:\widehat F_{\rm SAA}(d)\ge0.9\}.}
\]

它在所有测试日给出同一需求分位数，不需要神经网络、初始化或超参数搜索。

### M3. wSAA：Random-Forest-Weighted SAA

**预测输入**：`x=psi(A,X)∈R^19`；训练标签为日级 `D`。

1. 按人工冻结的森林配置，以 `(x_t,D_t)` 拟合**回归随机森林**；森林的作用是建立样本邻近关系，不直接将森林回归均值当成需求分位数。
2. 对测试输入 `x`，第 `b` 棵树确定其落入的叶子 `L_b(x)`；对全部训练输入按该树的分裂规则确定叶归属。
3. 定义叶节点样本数 `n_b(x)=# {t: x_t ∈ L_b(x)}`。给第 `t` 个训练日的权重：

\[
\boxed{
 w_t(x)=\frac1{B_{\rm tree}}\sum_{b=1}^{B_{\rm tree}}
 \frac{\mathbf1\{x_t\in L_b(x)\}}{n_b(x)}.
}
\]

4. 因每棵树的叶子至少包含一个训练样本，有 `w_t(x)>=0` 且 `sum_t w_t(x)=1`。使用：

\[
\boxed{\widehat F(d\mid x)=\sum_tw_t(x)\mathbf1\{D_t\le d\}.}
\]

5. 取：

\[
\boxed{\widehat Q_{0.9}(x)=\inf\{d:\widehat F(d\mid x)\ge0.9\}.}
\]

**实现约束**：森林树数、叶子大小、特征采样等人工冻结；使用 3 个事先固定的森林随机种子作数值随机性重复。计算叶权重时应以**全部训练输入**按已拟合树归属来定义叶样本数，不混用 bootstrap 抽样计数的重复权重定义。

### M4. Rolling-SAA：固定窗口经验分位数

**预测输入**：按生成序号排序后的**训练需求序列**，不使用测试实现需求更新、也不读患者 Token。

选择一个**运行前固定**的窗口长度 `W_roll<=2500`。取训练集最后 `W_roll` 个决策日：

\[
\boxed{\widehat F_{\rm roll}(d)=\frac1{W_{\rm roll}}
 \sum_{t=T-W_{\rm roll}+1}^{T}\mathbf1\{D_t\le d\}.}
\]

取该经验分布的 0.9 分位数后部署。S4 生成日期只有 i.i.d. 标识，没有真实星期季节规律，因此**不启用星期筛选**；本模型研究短窗口样本与完整 SAA 的区别，不能声称 DGP 内有可学习的时间趋势。

### M5. Flat-ETO：逐患者七类 PMF + 新增需求条件 CDF

Flat-ETO 分为**患者 PMF 估计器**、**新增需求 CDF 估计器**以及**分布组合与分位数决策**三个明确阶段。

#### (a) 逐患者离散 PMF

训练日第 `i` 名患者的输入定义为：

\[
\boxed{z_i=[a_i,\psi(A,X)]\in\mathbb R^{22}.}
\]

它由 `3` 维自身信息和 `19` 维当日总体摘要组成，包含患者自身、日级情境和当日群体简单摘要；不使用 Self-Attention。

共享患者级分类 MLP 输出 7 类 Softmax 概率：

\[
\boxed{\widehat p_{i,j}(Z)=\widehat{\Pr}(Y_i=0.5j\mid z_i),\quad j=0,\ldots,6,\qquad \sum_{j=0}^6\hat p_{i,j}=1.}
\]

**训练时允许的额外监督**：合成生成器保存每个训练日的逐患者结果 `Y_{ti}`。其他主基准正常训练只收到日级 `D_t`；Flat-ETO 可以额外以患者 `Y_{ti}` 的类别标签训练。按**逐日先平均患者损失，再对训练日平均**：

\[
\boxed{\mathcal L_{\rm patient}
=-\frac1T\sum_{t=1}^T\frac1{N_t}\sum_{i=1}^{N_t}
\log \hat p_{t,i,2Y_{ti}}.}
\]

`2Y_{ti}` 为整数类别标签 `0,...,6`。Padding 不进入损失。测试时**不得**输入实现的 `Y_i`。

#### (b) 新增需求的连续条件经验 CDF

训练日剩余需求可由额外训练监督计算：

\[
\boxed{R_t=D_t-\sum_iY_{ti}=D_{{\rm new},t}.}
\]

`R_t` 是**连续**非负需求，不能再取整到 0.5 网格，也不能让估计器读取真实 `U`、`J` 或 DGP 参数。对 `x_t=psi(A_t,X_t)` 与 `R_t` 拟合另一个固定配置的随机森林，依据 §M3 相同的叶邻近规则得到权重 `w_t^new(x)`，定义：

\[
\boxed{\widehat F_{\rm new}(d\mid x)=\sum_tw_t^{\rm new}(x)\mathbf1\{R_t\le d\}.}
\]

#### (c) 组合总需求分布与决策

先把各患者预测 PMF 按需求网格做离散卷积：

\[
\boxed{\widehat p_A=\widehat p_1*\cdots*\widehat p_N.}
\]

若 `p_A,k` 是总患者需求等于 `0.5k` 的预测概率，则构造全天需求条件 CDF：

\[
\boxed{\widehat F_D(d\mid Z)
=\sum_k\widehat p_{A,k}(Z)\,
\widehat F_{\rm new}(d-0.5k\mid\psi(A,X)).}
\]

再定义：

\[
\boxed{\widehat Q_{0.9}(Z)=\inf\{d:\widehat F_D(d\mid Z)\ge0.9\}.}
\]

这是一种**给定事前信息后采用独立边际卷积的估计—再优化方法**。S4 真实 DGP 含未观测共同冲击 `U`，患者结果与新增需求通常条件相关；Flat-ETO 的该独立卷积是明确的、可能发生结构性误设的基线假设，不能把真实 DGP 的 `U` 直接提供给它来修补。

患者 PMF 与新增需求 CDF 都是**估计得到的**，不能用 Oracle 真实概率替代。所有网络/森林参数事先固定，不根据测试表现替换组件。

### M6. Set-ETO：DeepSets + **固定 HurdleGamma** 条件分布

**预测输入**：完整患者 `A` 与日级 `X`。患者共享编码器 `phi_set` 按 Mask 后均值池化：

\[
\boxed{h_A=\frac1N\sum_{i=1}^N\phi_{\rm set}(a_i),\qquad h=[h_A,X].}
\]

集合编码器和融合主干采用 DeepSets，与 DeepQR 保持尽可能接近的结构容量。**Set-ETO 和 DeepQR 都已确定为 DeepSets，不使用 DAT 的 Self/Cross-Attention。**

Set-ETO 最后输出 HurdleGamma 分布参数：

\[
\boxed{\pi_0(Z)\in(0,1),\qquad k(Z)>0,\qquad\vartheta(Z)>0.}
\]

其中：

\[
\Pr(D=0\mid Z)=\pi_0(Z),\quad
D\mid(D>0,Z)\sim\operatorname{Gamma}(k(Z),\vartheta(Z)).
\]

`pi0` 用 Sigmoid 保证概率合法，形状 `k` 和尺度 `vartheta` 用 Softplus 加统一数值稳定项保证正值。只使用日级 `D` 最大化条件对数似然（最小化 NLL）：

\[
\boxed{\mathcal L_{\rm Set}=-\frac1T\sum_{t=1}^T\log p_\theta(D_t\mid A_t,X_t).}
\]

对 `d>0`，其密度是 `(1-pi0)*GammaPDF(d;k,vartheta)`；对 `d=0`，其概率质量为 `pi0`。由预测条件分布求 0.9 分位数：

\[
\boxed{\widehat Q_\tau(Z)=
\begin{cases}
0,&\tau\le\pi_0(Z),\\
F^{-1}_\Gamma((\tau-\pi_0(Z))/(1-\pi_0(Z));k(Z),\vartheta(Z)),&\tau>\pi_0(Z).
\end{cases}}
\]

最后裁剪到 `[0,B]`。**只采用 HurdleGamma 一种预先指定的正式分布族**，不在 HurdleGamma 和 Normal 之间按验证/测试结果择优；这是九基准中的**唯一 Set-ETO 方法**。

由于 DGP 有连续余量 `J>0` 几乎处处，真实需求在零点并无正概率质量；HurdleGamma 是有意保留的**可能误设**的参数化分布基线。

### M7. LQR：L2 正则化线性分位数回归

**输入**：`x=psi(A,X)∈R^19`；训练标签 `D`。

拟合：

\[
\boxed{\hat Q_\tau(x)=\operatorname{clip}_{[0,B]}(\beta_0+\beta^\top x).}
\]

以预先固定的 `lambda_LQR>=0` 求解凸问题：

\[
\boxed{\min_{\beta_0,\beta}\frac1T\sum_t
\rho_\tau(D_t-\beta_0-\beta^\top x_t)+\lambda_{\rm LQR}\|\beta\|_2^2.}
\]

截距 `beta0` 不正则；数值解的优化器和容差在代码配置中固定。主任务 `tau=0.9`，辅助 `tau` 需重新求解对应损失。若求解器是确定性的，不需要人为复制三份相同结果。

### M8. GBQR：Gradient-Boosted Quantile Regression

**输入**：`x=psi(A,X)∈R^19`；训练标签 `D`。

采用以 Pinball/Quantile 为目标的梯度提升树：

\[
\boxed{F_K(x)=F_0(x)+\sum_{m=1}^K\eta\,h_m(x),}
\]

其中 `F0` 为初始常数分位数，`h_m` 为迭代拟合的回归树，`eta` 为预先冻结的学习率。可使用支持 quantile loss 的稳定梯度提升回归器，模型训练目标对应：

\[
\min_F\sum_t\rho_\tau(D_t-F(x_t)).
\]

每个 `tau` 单独训练，预测裁剪到 `[0,B]`。树叶数、树深、迭代数、正则条件均事先固定；不能用测试表现挑选更适合 DAT 结果的配置。

### M9. DeepQR：DeepSets + 直接条件分位数

**输入**：完整 `A,X`，与 Set-ETO 一致。

患者共享网络提取 `phi_set(a_i)`，对有效患者作 Mask Mean Pool：

\[
\boxed{h_A=\frac1N\sum_{i=1}^N\phi_{\rm set}(a_i).}
\]

将 `[h_A,X]` 交给融合 MLP，直接输出需求条件分位数：

\[
\boxed{\hat Q_\tau(Z)=\operatorname{clip}_{[0,B]}(g_\tau([h_A,X])).}
\]

用日级 `D` 的 Pinball 损失直接训练：

\[
\boxed{\mathcal L_{\rm DeepQR}=\frac1T\sum_t\rho_\tau(D_t-\hat Q_\tau(Z_t)).}
\]

DeepQR **不**估计需求分布族；Set-ETO 使用相同类型的 DeepSets 编码，却通过 HurdleGamma NLL 学习整个条件分布。两者的差异应解释为**学习目标及分布假设的差异**，不只是池化方式差异。

### 5.10 九方法输入与监督对照表

| P1 方法 | 预测输入 | 训练标签 | 主要估计对象 | 主 `tau=0.9` 的输出 |
|---|---|---|---|---|
| DAT | 完整 `A,X` | 日级 `D` | Self/Cross-Attention 分位数函数 | 直接 `Qhat` |
| SAA | 历史训练 `D` | 日级 `D` | 无条件经验 CDF | 经验分位数 |
| wSAA | 19 维 `psi(A,X)` | 日级 `D` | 森林条件样本权重与经验 CDF | 加权分位数 |
| Rolling-SAA | 固定历史窗口 `D` | 日级 `D` | 窗口经验 CDF | 窗口分位数 |
| Flat-ETO | 患者 22 维 `[a_i,psi]`、新增需求 19 维 `psi` | **日级 D + 训练期患者 Y_i** | 七类患者 PMF + 新增连续 CDF | 卷积 CDF 分位数 |
| Set-ETO | 完整 `A,X`，DeepSets | 日级 `D` | HurdleGamma 条件分布 | 分布分位数 |
| LQR | 19 维 `psi` | 日级 `D` | 线性 Pinball | 直接分位数 |
| GBQR | 19 维 `psi` | 日级 `D` | 提升树 Pinball | 直接分位数 |
| DeepQR | 完整 `A,X`，DeepSets | 日级 `D` | 非线性集合分位数函数 | 直接分位数 |

**必须如实披露**：Flat-ETO 在**训练**时拥有逐患者结果监督，而其他主基准没有；传统固定维摘要不一定保留完整患者 Token 的全部信息。无论方法如何估计，最终都执行同一个目标 G 补充政策。

---

## 6. P1 九模型基准比较

- **DGP**：固定 S4。
- **主 `tau`**：0.9。
- **目标**：`G=260,280,300,320,340`；G 只作用于部署与评价。
- **组数**：恰好 **9 个方法**，对应 §5 的 M1–M9。
- **数据**：每重复 2,500 训练 + 3,000 独立测试；10 次独立数据重复；随机估计器 3 个训练/算法种子。
- **超参数**：九个方法各一套事先人为固定的配置；不搜索候选。

结果表仅含两项：

| Method | MeanCost ↓ | SL_G ↑ |
|---|---:|---:|
| DAT | 待真实运行 | 待真实运行 |
| SAA | 待真实运行 | 待真实运行 |
| wSAA | 待真实运行 | 待真实运行 |
| Rolling-SAA | 待真实运行 | 待真实运行 |
| Flat-ETO | 待真实运行 | 待真实运行 |
| Set-ETO | 待真实运行 | 待真实运行 |
| LQR | 待真实运行 | 待真实运行 |
| GBQR | 待真实运行 | 待真实运行 |
| DeepQR | 待真实运行 | 待真实运行 |

对同一重复的同一测试日比较全部方法，保留配对关系。不能因为 DAT 未胜出就重选某个基线分布族或调整 DGP。主图可在 `G=300,tau=0.9` 展示方法排名，再为其他 G 展示敏感性；全部结果仍只包含两项性能指标。

---

## 7. P2：DAT 的七个结构条件（五主 + 两辅助）

**DGP 固定 S4，主 `tau=0.9`，输入完整 `A,X`。** 除指定的模块替换之外，其他模型组件、训练数据及人为固定的优化规则应保持一致。

### 7.1 五个主结构变体

| 变体 ID | 唯一结构操作 | 保留哪些能力 |
|---|---|---|
| `full` | 完整 DAT | Self-Attention、日级动态 Cross-Attention、Day FFN |
| `no_self_attention` | Self-Attention 残差输出设为零/移除该计算支路 | 保留患者编码、动态 Cross-Attention、全部 X 与 Day FFN |
| `mean_pool` | 使用有效患者 Memory 的均值池化 + 可学习输出映射代替 Cross-Attention | 保留 Self-Attention、完整 X、日级残差与 FFN |
| `sum_pool` | 使用有效患者 Memory 的求和池化 + 可学习输出映射代替 Cross-Attention | 同上 |
| `no_day_query` | 将 Cross-Attention 的 Query 改为与 X 无关的可训练常数 Query | X 仍通过 Day Linear 的残差主支路和 Day FFN 进入模型 |

记有效 Mask 为 `m_i∈{0,1}`，`M_i=LN_m(E_i)`，池化定义为：

\[
\boxed{\mathrm{MeanPool}=W_o\left(\frac{\sum_i m_iM_i}{\sum_i m_i}\right),\quad
\mathrm{SumPool}=W_o\left(\sum_i m_iM_i\right).}
\]

对应日级输出依然为 `v=u+Pool`，再执行相同 `Day FFN` 和线性 Clip。任何消融都不得同时删除某些 `A/X/I` 可见坐标，避免结构效应与信息效应混淆。

### 7.2 两个容量匹配辅助变体

- `mean_pool_param_matched`
- `sum_pool_param_matched`

通过**仅依据模型参数计数**人工固定这些变体的隐藏宽度，目标使稠密参数位置数与 Full DAT 的差异不超过 `5%`。这不涉及验证/测试结果，因此**不属于超参数搜索**。保存真实参数量和最终固定宽度作为审计元数据，不将参数量当成第三个性能指标。

P2 共**7 个结构条件**，没有额外增加双重消融；更改场景或信息输入的变体不能混入本结构主组。每种变体对同一重复从头训练，随机模型按三个 Seed 运行，最后只报告 MeanCost 与 SL_G。

---

## 8. P3：五级事前信息价值实验

**DGP 固定 S4，主 `tau=0.9`；网络模块结构与分配参数预算保持一致，每级从头训练。** 这组只改变可以看到的信息，不改变需求发生机制。

| 等级 | 授权日级特征 | 授权患者信息 | 应隐藏的信息 |
|---|---|---|---|
| **L0** | `c1,c2,I` | 只有固定 Dummy Token | 真实人数、全部逐患者信息及患者摘要 |
| **L1** | L0 + `N/M,sum(u)/M,sum(v)/M` | 仍为固定 Dummy Token | 逐患者信息、风险均值及二阶矩 |
| **L2** | L1 + `mean(r),mean(r²)` | 仍为固定 Dummy Token | 逐患者特征及其对应关系 |
| **L3** | 完整 L2 日级摘要 | 真实患者逐个 `(u_i,v_i)` | 每名患者自己的 `r_i`，但 L2 授权的风险摘要保留 |
| **L4** | 完整八维 `X` | 真实患者逐个 `(u_i,v_i,r_i)` | 无 |

**门控合同**：

1. 所有级别 Day Encoder 都接收同样形状的八维张量；被隐藏坐标以固定不含样本信息的常数屏蔽，不能改变模型参数预算。
2. **L0–L2 必须使用一个固定的 Dummy Token 和固定有效 Mask**，绝不能让患者 Token 的真实长度、Padding 或 `mask.sum()` 透露 N。模型不能无条件自动追加人数比例。
3. **L3** 才可以使用实际患者 Token/Mask，但逐患者风险坐标固定隐藏，同时保留 L2 已授权的风险均值、二阶矩。
4. **L4** 才允许全部患者坐标；五级均使用相同训练日、测试日和人工冻结的训练规则。
5. 五级从头训练，不能把 L4 权重迁移给 L0–L3，也不能用测试指标在等级之间选最佳模型并称为统一方法。

解释：L3→L4 研究的是**单名患者的风险分数与其申请/备血特征对应关系**所带来的附加价值，不是从“无临床风险”跳到“有临床风险”。有限样本模型的成本可能不按信息等级严格单调下降，服务率也不必单调上升；不能将网络实现收益等同于 Bayes 最优信息价值。

---

## 9. 跨 Seed 与独立重复的统计汇总

固定一个 `method/variant/information_level` 和 `G`，第 `r` 份独立数据、第 `s` 个训练 Seed 得到：

\[
C_{r,s}=\widehat{\mathrm{MeanCost}}_{r,s},\qquad L_{r,s}=\widehat{\mathrm{SL}}_{G,r,s}.
\]

先在**同一数据重复**内平均真实存在的随机 Seed：

\[
\boxed{\bar C_r=\frac1{S_r}\sum_{s=1}^{S_r}C_{r,s},\qquad
\bar L_r=\frac1{S_r}\sum_{s=1}^{S_r}L_{r,s}.}
\]

随机估计器 `S_r=3`；确定性估计器 `S_r=1`。之后在 10 个独立数据重复之间求平均：

\[
\boxed{\bar C=\frac1{10}\sum_{r=1}^{10}\bar C_r,\qquad
\bar L=\frac1{10}\sum_{r=1}^{10}\bar L_r.}
\]

置信区间和方法对比的配对差都应以**10 份独立数据重复**为统计单位，不能把 30 次训练当成 30 份独立样本。对于比较方法 A 与 B 的配对成本差：

\[
\Delta_r^{\rm cost}=\bar C_{r,A}-\bar C_{r,B};
\]

服务率同理。10 次独立重复不保证细微差异一定得到狭窄置信区间；如算力允许追加重复，应为所有受比较方法按预定方案统一追加，而不是根据“哪个方法没赢”选择性扩充。

## 10. 执行伪代码与计算复用

```python
for replicate_id in range(10):
    train, test = generate_independent_s4_data(
        n_train=2500, n_test=3000, data_seed=DATA_SEEDS[replicate_id]
    )

    # P1: nine methods; P2: seven DAT structural variants;
    # P3: five information levels. All use same train/test for this replicate.
    for specification in PREDECLARED_SPECS:
        seeds = TRAIN_SEEDS if specification.has_stochastic_fit else [None]
        for train_seed in seeds:
            model = fit_predeclared_fixed_config(
                specification, train, tau=0.9, seed=train_seed
            )
            qhat = predict_tau_quantile(model, test, tau=0.9)
            qhat = clip(qhat, lo=0, hi=36.5)
            for G in [260, 280, 300, 320, 340]:
                q = clip(qhat + G - test.I, lo=0)
                final_inventory = test.I + q - test.D
                daily_cost = (0.9 * clip(G-final_inventory, lo=0)
                              + 0.1 * clip(final_inventory-G, lo=0))
                save_row(replicate_id, specification.id, train_seed, 0.9, G,
                         MeanCost=mean(daily_cost),
                         SL_G=mean(final_inventory >= G))

aggregate_seeds_within_replicate_then_aggregate_10_replicates()
```

说明：分布方法可复用同一个拟合对象获取辅助多个 `tau`，但主循环只使用 `tau=0.9`；如启用辅助任务，其配置和结果必须另存。不能让 `G` 引起额外模型拟合。

## 11. 结果文件、图和权限核查

### 11.1 最终结果 Schema

```text
experiment, scenario_id='S4', tau, G,
method, structure_variant, information_level,
replicate_id, train_seed, n_test,
MeanCost, SL_G
```

其中 `structure_variant`、`information_level` 对不适用方法可以为空。`MeanCost` 和 `SL_G` 是唯一两个**评价指标列**。训练配置指纹、参数量、种子、异常状态等放在单独 `metadata`，不是第三个性能指标。

### 11.2 主图

| 实验 | 成本主图 | 服务率主图 |
|---|---|---|
| P1 | `P1_S4_MeanCost_by_method.png` | `P1_S4_SL_by_method.png` |
| P2 | `P2_S4_MeanCost_ablation.png` | `P2_S4_SL_ablation.png` |
| P3 | `P3_S4_MeanCost_information.png` | `P3_S4_SL_information.png` |

主图 `tau=0.9`；按 G 分面或对预设中央 `G=300` 单独展示，其余 G 使用另一同指标图。跨独立重复展示均值与预先选定的不确定性区间。辅助 `tau` 的图必须明确标注为附加分析。

### 11.3 数据权限

- 生成器保留潜变量和事后结果，预测器**不得**读取潜在 `U`、测试日 `Y_i`、`J`、`D_new` 或真实 Oracle 分位数作为输入。
- Flat-ETO 仅在训练集额外读取 `Y_i`；不能让其他方法偷偷读取同样标签并宣称监督条件一致。
- P3 的 L0–L2 严禁通过 Token 数、Mask、归一化逻辑泄漏实际 N。
- 本 Synthetic 实验不读取医院实际数据、不伪造医院历史收货记录。

## 12. 正式运行前验收清单

| ID | 条件 | 必须满足 |
|---|---|---|
| P01 | 场景 | **只有 S4**；其他场景不在正式运行列表 |
| P02 | 主分位数 | **tau=0.9**；辅助 tau 与主结果分离 |
| P03 | 数量尺度 | 原库存 `I=315+20c1+U(-70,70)`、原 G 五档、`B=36.5` 均保留 |
| P04 | 数据划分 | 每个独立重复 **2500 训练 + 3000 测试**，不进行验证集超参选优 |
| P05 | 统计结构 | **10 个独立重复；随机方法 3 Seed**；确定性方法不伪造重复 |
| P06 | 调参 | **无超参数搜索**；所有配置人工固定，具体数值只写运行配置 |
| P07 | 指标 | 最终只有 **MeanCost** 与 **SL_G**，都使用 `mean()` 而不是总成本 `sum()` |
| P08 | 方法数 | P1 恰好九种固定方法，无额外替换的第十模型 |
| P09 | DeepSets | Set-ETO、DeepQR 都用有效患者 Mean Pool 的 DeepSets |
| P10 | Set-ETO | 固定 **HurdleGamma**，不在 Normal 与 Gamma 中按数据选优 |
| P11 | 传统摘要 | `psi(A,X)` 恰好 19 维；Flat 患者输入恰好 22 维 |
| P12 | Flat-ETO | 患者七类 PMF + 新增连续需求 CDF + 卷积，额外训练监督 `Y_i` 已披露 |
| P13 | P2 结构组 | 五个主变体 + 两个参数匹配辅助变体，输入信息保持相同 |
| P14 | P3 信息组 | L0–L4 五级、固定 Dummy Token 防 N 泄漏、逐患者风险屏蔽严格 |
| P15 | G 重用 | 同一模型对五个 G **只重新算政策和评价** |
| P16 | 配对比较 | 同一数据重复所有方法共享同一测试集、同一 D 实现 |
| P17 | 选择偏差 | 不按测试指标挑初始化/随机种子/训练轮数/模型家族 |
| P18 | 数据隔离 | 完全无医院真实数据、真实模型权重或历史真实收货基线 |

---

**最终执行口径：S4 Only × 主 `tau=0.9` × 五个 G × [P1 九模型 / P2 七结构 / P3 五信息级别] × 10 独立数据重复 × 随机模型 3 Seed；全部超参数预先人为固定、无搜索；正式报告只有每天平均成本 `MeanCost` 与期末达到 G 的服务率 `SL_G`。**
