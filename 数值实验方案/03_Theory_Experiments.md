# DAT Synthetic 理论实验协议：T1、T2、T3

> **文档状态**：理论实验实施规范（2026-11-04 整理版）  
> **依据**：`synthetic_protocol.md` 第 4–5、7–8 节；`DAT理论保证(3).pdf` 命题 9、定理 10、定理 17、推论 18/19/24。  
> **范围**：仅使用第二号文档的 Synthetic DGP；不调用真实医院数据。  
> **核心区分**：**T1 看随训练样本 \(T\) 的统计学习；T2 看随参数量 \(P\) 的可达到逼近；T3 看样本和模型容量共同增长的风险/运营收敛。** 数值 AdamW 不是理论中已认证的精确 ERM。

## 1. 理论目标与三个实验的不同纵轴

令 \(Z=(A,X)\)、\(D\in[0,B]\)，分位数风险为

\[
R_\tau(f)=\mathbb E\rho_\tau(D-f(Z)),\qquad
\rho_\tau(e)=\tau[e]_++(1-\tau)[-e]_+.
\]

设 \(f^\star(Z)=Q_\tau(D\mid Z)\) 是 DGP 的真实条件 \(\tau\)-分位数，\(\mathcal F_P\) 是抽样前固定的紧 DAT 函数类。

理论中的两项误差：

\[
\boxed{\begin{aligned}
\mathcal E_{P,T}^{\mathrm{est}}&=R_\tau(\hat f_{P,T})-R_\tau(f_P^\dagger),\\
\mathcal E_P^{\mathrm{app}}&=R_\tau(f_P^\dagger)-R_\tau(f^\star),
\end{aligned}}
\]

其中 \(f_P^\dagger\) 是**类内总体最优解**，\(\hat f_{P,T}\) 是**精确 ERM**。于是

\[
R_\tau(\hat f_{P,T})-R_\tau(f^\star)=
\mathcal E_{P,T}^{\mathrm{est}}+\mathcal E_P^{\mathrm{app}}.
\]

实际 AdamW 得到的函数记为 \(\tilde f_{P,T}\)，不要未经认证就写成 \(\hat f_{P,T}\)。

### 1.1 三组理论实验的最终作图口径

| 实验 | 固定什么 | 横轴 | **正文主纵轴** | 关键解释 |
|---|---|---|---|---|
| **T1a：可实现校准** | 常数预测子类 \(P=1\)；C0 DGP | 训练日数 \(T\) | **精确估计超额 Pinball 风险**与 \(1-\delta\) 理论上界 | 真正可以求精确 ERM，逼近误差为零 |
| **T1b：完整 DAT** | \(P=5801,R=100\)；S2/S3 | 训练日数 \(T\) | **相对 Oracle 的总超额 Pinball 风险** | 固定逼近误差和未认证优化误差可能形成平台；不能称纯估计误差 |
| **T2：逼近诊断** | DGP 和 Oracle 拟合池规模固定 | 实际参数数 \(P\) | **Oracle Quantile L1**：预测与 oracle 分位数的平均绝对距离 | 训练出的一个可达到逼近器，并非整个类的精确最优逼近误差 |
| **T3：联合增长** | S2/S3 内的 DGP 固定 | 训练日数 \(T\) | **相对 Oracle 的总超额 Pinball 风险** | 与固定宽度 32/128 的轨道比较；另画运营遗憾 |

> **必须避免混淆**：T1b 的总超额风险包含固定类的逼近差距；T2 的 Oracle L1 是函数值距离，不是 Pinball 风险。T3 则考察实际训练流程在增长容量下的总风险轨迹。

## 2. 共同数据、模型、种子及评估规则

### 2.1 共同配置

| 项 | 固定值或规则 |
|---|---|
| DGP 规模 | \(M=2,d_a=3,d_x=7,B=18.5\)，DAT 潜维 \(d=20\) |
| 理论任务分位数 | **\(\tau=0.9\)** |
| DGP 场景 | T1 完整 DAT：S2/S3；T2：S2/S3/S4；T3：S2/S3；C0 仅 T1 校准 |
| 训练样本量网格 | \(T\in\{500,1000,2000,4000,8000,16000\}\) |
| 预先登记扩展 | 32,000、64,000、128,000（单独标记扩展，不在试验中途静默增加） |
| 正式独立数据重复 | **200**（按原始理论实验协议） |
| Smoke/耗时试运行 | 20 份，不与正式 200 份混合 |
| 初始化 | 固定 3 个候选初始化，按规定的训练目标选择；不是 3 份独立数据重复 |
| 总体评价输入池 | 起始 50,000 个独立 \(Z\)，对需求做条件积分而不是只用抽到的单个 \(D\) |
| MHA 头数、LayerNorm | SA4、CA4、三个 LN、`eps=1e-5` |
| Dropout、Weight Decay | 均为 0 |
| 模型选择测试泄漏 | 不使用测试集或 Oracle 风险来选择普通训练的初始化、轮次或超参数 |

200 是独立模拟**数据重复**数量，不是 200 次来自同一数据集的随机初始化。对同一重复，应保持对应实验中的公共输入池和条件风险评价池，以利于配对比较。

### 2.2 正常 DAT 数值训练（T1 完整 DAT、T3）

原协议固定的数值优化规则：

```yaml
optimizer: AdamW
learning_rate: 0.0003
batch_size: 128
max_epochs: 300
weight_decay: 0.0
grad_clip_norm: 1.0
dropout: 0.0
training_loss: raw_pinball
checkpoints: by_lowest_full_training_pinball
initializations: 3
parameter_constraint: global_euclidean_ball
```

每一步优化器更新后对**全部网络参数拼接向量**作欧氏半径投影。不得使用 oracle、测试成本或测试样本挑选训练 checkpoint；理论轨道不把性能实验的 500 日验证集拼入拟合样本。

实际拟合天数严格等于横轴 \(T\)。训练结果记为 \(\tilde f\)，只有具备全局最优认证时才记为精确 \(\hat f\)。

### 2.3 条件期望风险的计算

DGP 提供 \(p_k(z)=\Pr(D_0=0.5k\mid Z=z)\)，可计算

\[
\phi_z(y)=\mathbb E[\rho_\tau(D-y)\mid Z=z]
=\sum_kp_k(z)\,2\int_0^{0.5}\rho_\tau(0.5k+j-y)\,dj.
\]

给定独立评价输入 \(Z_1^{\mathrm{eval}},\ldots,Z_{N_e}^{\mathrm{eval}}\)，\(N_e=50000\)，对**已训练函数** \(f\) 计算

\[
\widehat R_\tau^{\mathrm{cond}}(f)
=\frac1{N_e}\sum_{j=1}^{N_e}
\phi_{Z_j^{\mathrm{eval}}}(f(Z_j^{\mathrm{eval}})).
\]

再用同一池计算 \(\widehat R_\tau^{\mathrm{cond}}(f^\star)\)，相减形成总超额风险估计。此法消除了给定 \(Z\) 后单次随机抽 \(D\) 带来的评价噪声，但仍有**抽样 \(Z\)** 的 Monte Carlo 误差，S4 还另有正态积分误差。

## 3. T1：随样本量 \(T\) 的有限样本与高概率界实验

### 3.1 实验目标

- **T1-C0**：在可精确求 ERM 且逼近误差为零的子类中，数值检查估计误差分布、理论上界及越界频率。
- **T1-DAT**：固定完整 DAT 容量与参数球，观察更多训练日带来的**总风险**变化；其中逼近误差在函数类固定时不变，但实际优化误差可能存在。

### 3.2 T1-C0：可实现的常数校准环境

C0 不是 S1–S4 的替换。它只用于估计保证的精确校准：

- 固定 \(N=M\)；所有已知患者和新增订单的需求概率固定为：`p_issue=0.35`、`p_quantity=0.4`、`p_new=0.3`、`p_new_quantity=0.4`。
- 不让需求分布依赖可见患者信息、\(c_1,c_2\) 或 \(U\)；保留 \(J\sim U(0,0.5)\)。
- 真正的条件分位数为常数 \(q_\tau^\star\)，可由 C0 的离散 PMF 卷积和连续余量公式算得。

选择常数预测子类

\[
\mathcal F_{\mathrm{const}}=\{f_c(Z)\equiv c:0\le c\le B\}.
\]

其中只有一个可训练的**自由偏置参数**；在实现中固定其他 DAT 参数使输出恒为该偏置即可。用于理论界的维数 **\(P=1\)**，参数到预测的 Lipschitz 常数可取 **\(\Lambda_P=1\)**，参数半径须覆盖 \([0,B]\)（可用半径 \(B\) 或原合同 \(R=100\)，一旦确定不可随 \(T\) 改变）。

对一组 \(T\) 个独立需求 \(D_1,\ldots,D_T\)，选取经验下 \(\tau\)-分位数

\[
\boxed{\hat q_{\tau,T}=\inf\{y:\widehat F_T(y)\ge\tau\}.}
\]

它是常数预测子类上的 Pinball **精确 ERM**。由于 \(f^\star\in\mathcal F_{\mathrm{const}}\)，有 \(\mathcal E^{\mathrm{app}}=0\)。

因此每次数据重复 \(r\) 的 T1 纵轴：

\[
\boxed{
Y_{T,r}^{\mathrm{C0}}=
R_\tau(\hat q_{\tau,T}^{(r)})-R_\tau(q_\tau^\star)
=\mathcal E_{1,T}^{\mathrm{est}}.
}
\]

C0 分布与 \(Z\) 无关，理论总体风险可直接按固定混合 PMF 解析积分，无需把实验估计误差混入评价随机 \(D\) 噪声。

### 3.3 T1 的高概率界如何画出来

记 \(L_\tau=\max(\tau,1-\tau)=0.9\)，输出跨度上界 \(\Delta_P\le B\)。原理论命题 9 形式为

\[
\boxed{
\Pr\left(\mathcal E_{P,T}^{\mathrm{est}}\le\varepsilon_{P,T}(\delta)\right)
\ge1-\delta.
}
\]

对 \(\Delta_P>0\)，可用闭式保守上界

\[
\boxed{
\begin{aligned}
\varepsilon_{P,T}^{\mathrm{closed}}(\delta)
=\min\Bigg\{L_\tau\Delta_P,\;
&6L_\tau\Delta_P\sqrt{\frac PT}
\left[\sqrt{\log\left(1+\frac{8R_P\Lambda_P}{\Delta_P}\right)}
+\frac{\sqrt\pi}{2}\right]\\
&+L_\tau\Delta_P\sqrt{\frac{2\log(1/\delta)}T}\Bigg\}.
\end{aligned}
}
\]

若使用原文更精细的积分界，则定义

\[
H_P(s)=P\log(1+2R_P\Lambda_P/s),
\]

\[
\mathscr J_{P,T}=\min\left\{\frac{\Delta_P}{2},
\inf_{0\le a\le\Delta_P/4}\left[4a+\frac{12}{\sqrt T}
\int_a^{\Delta_P/4}\sqrt{H_P(s)}\,ds\right]\right\},
\]

并计算

\[
\varepsilon_{P,T}(\delta)
=\min\left\{L_\tau\Delta_P,\;
2L_\tau\mathscr J_{P,T}+L_\tau\Delta_P\sqrt{\frac{2\log(1/\delta)}T}\right\}.
\]

这两种都是以**确定性合法包络** \(R_P,\Lambda_P,\Delta_P\) 计算出的上界；不能根据误差结果拟合 \(\Lambda_P\) 来制造看似紧的界。原参数球的 C0 情形可用 \(P=1,\Lambda_P=1,\Delta_P=B\)。

对 \(\delta\in\{0.10,0.05\}\)，每个 \(T\) 有 200 份独立误差 \(Y_{T,r}^{\mathrm{C0}}\)，计算经验越界率

\[
\boxed{
\widehat v_\delta(T)=\frac1{200}\sum_{r=1}^{200}
\mathbf1\{Y_{T,r}^{\mathrm{C0}}>\varepsilon_{1,T}(\delta)\}.
}
\]

该概率保证针对**真实抽样重复的总体越界概率**，而不是要求有限 200 次重复必然恰好出现 10 次或 20 次越界。理论界可能退化为平凡界 \(L_\tau B\)，观察到零次越界并不说明界紧。

### 3.4 T1-C0 的最终主图

| 图 | 横轴 | 纵轴 | 内容 |
|---|---|---|---|
| `T1_C0_error_bound` | **训练日数 \(T\)**，对数刻度 | **精确估计超额 Pinball 风险** | 200 重复的均值/中位数及 90%、95% 经验分位数；叠加理论 90%、95% 高概率上界 |
| `T1_C0_exceedance` | **训练日数 \(T\)** | **经验越界率 \(\hat v_\delta(T)\)** | \(\delta=0.10/0.05\) 两条经验曲线及对应参考线 \(y=\delta\) |

纵轴的风险单位是归一化 Pinball 损失单位，不是单日需求数量，也不是运营成本。

### 3.5 T1-DAT：固定宽度下的完整学习曲线

S2/S3 分别固定：

\[
H_{\mathrm{pat}}=H_{\mathrm{ffn}}=32,
\quad P=5801,
\quad R_P=100,
\quad d=20.
\]

只改变训练日数 \(T\)。按 §2.2 训练，使用 DGP oracle **只做评价**，计算

\[
\boxed{
Y_{T,r}^{\mathrm{DAT}}=
R_\tau(\tilde f_{P,T}^{(r)})-R_\tau(f^\star).
}
\]

该值包含：（i）固定容量的逼近差距，（ii）有限样本统计误差，（iii）未认证的优化误差。**不能**将它直接写为 \(\mathcal E_{P,T}^{\mathrm{est}}\)，也不能把命题 9 的仅估计项闭式界直接当作它的正式上界。

图 `T1_DAT_total_excess`：横轴 \(T\)，纵轴 **总超额 Pinball 风险**，分别显示 S2、S3，含独立重复区间。不要求曲线每一点单调，也不保证平台一定可观察。

### 3.6 T1 验收

- C0 的标签生成独立于 \(Z\)，并正确使用原常数概率。
- C0 数值 ERM 与直接样本分位数结果一致；类内逼近误差确实为 0。
- 完整 DAT 始终 \(P=5801,R_P=100\)，不随 \(T\) 增长。
- 200 独立重复、3 初始化是不同统计层次；按协议聚合。
- 理论上界的常数在观察结果前预设；越界频率使用正确的 \(\delta\)。
- 正常训练只使用 \(D\)，不使用 oracle 训练标签。

## 4. T2：随参数量 \(P\) 的构造性逼近诊断

### 4.1 实验要识别什么

定理 10 在人数有界、目标 Hölder、宽度条件下证明**存在** DAT 逼近器，使

\[
\inf_{f\in\mathcal F_P}\|f-f^\star\|_{\infty,K}
\le CP^{-\gamma},\qquad
\gamma=\frac\alpha{Md_{\mathrm{eff}}}.
\]

对于理论 DGP，\(\alpha_{\mathrm{Holder}}=1,M=2,d_{\mathrm{eff}}=16\)，于是 \(\gamma=1/32\)。该指数是保守**存在性充分界**，不预测实际拟合曲线的准确斜率。

T2 训练一个可实现的具体网络 \(\tilde f_P\)，数值衡量它与 Oracle 的距离；这不是求取网络类上的未知全局最优。

### 4.2 固定的数据规模和变化的模型容量

**原协议固定**：

- 场景：S2、S3、S4。
- 每个独立数据重复，生成 **10,000 个独立完整信息输入 \(Z\)** 及其机制 Oracle 分位数 \(f^\star(Z)\) 作为无观测需求噪声的拟合样本。
- 使用 **50,000 个独立评价输入**；同一数据重复内不同宽度共同使用这些评价输入。
- 固定最多 300 epoch；以**完整训练平方误差**选择 checkpoint 和三个初始化中的最优者。
- 不从 T2 把权重迁移给 T1、T3 或 P1–P3。

变化的是患者编码器和日级 FFN 的**共同隐藏宽度 \(H\)**：

| \(H\) | 16 | 32 | 64 | 128 | 256 | 512 |
|---:|---:|---:|---:|---:|---:|---:|
| \(P=p(H)=3721+65H\) | 4,761 | 5,801 | 7,881 | 12,041 | 20,361 | 37,001 |

所有其他结构固定（潜维 20、头数 4/4、Attention 层数 1/1、三处 LN）。以实际 \(P\) 为横轴，而不是只用 \(H\)。

每档参数范数使用预先确定的增长包络：

\[
\boxed{R_P=100(P/5801)^2.}
\]

因此 T2 严格上是“**随参数预算及预定范数包络共同增长的可达到逼近诊断**”，并非只改 \(P\) 不改 \(R_P\) 的纯因子实验。

### 4.3 拟合目标和两项核心数值定义

用 Oracle 分位数作平方误差监督：

\[
\boxed{
\min_{\theta:\|\theta\|_2\le R_P}
\frac1{10000}\sum_{i=1}^{10000}
\big(f_\theta(Z_i)-f^\star(Z_i)\big)^2.
}
\]

在独立 50,000 个 \(Z\) 上计算：

\[
\boxed{
\mathrm{OracleL1}(P)=\frac1{50000}\sum_j
|\tilde f_P(Z_j)-f^\star(Z_j)|,
}
\]

\[
\mathrm{OracleL2}(P)=\left[\frac1{50000}\sum_j
(\tilde f_P(Z_j)-f^\star(Z_j))^2\right]^{1/2}.
\]

还可以记录以下**理论诊断**，不属于性能实验 P1–P3 的第三指标：

\[
\mathrm{EvalMax}(P)=\max_{1\le j\le50000}
|\tilde f_P(Z_j)-f^\star(Z_j)|,
\]

\[
\mathrm{AttainableExcessRisk}(P)=
R_\tau(\tilde f_P)-R_\tau(f^\star).
\]

请注意：`EvalMax` 只是**测试点最大值**，不是整个紧定义域上的一致范数 \(\|\cdot\|_{\infty,K}\)。`AttainableExcessRisk` 是当前训练模型达到的风险差，真正类内逼近误差不大于这一风险差，但你并未计算到类内全局最小值。

### 4.4 T2 最终作图

| 图 | 横轴 | 主纵轴 | 曲线 |
|---|---|---|---|
| `T2_oracle_L1_vs_P` | **实际总参数量 \(P\)**（建议对数） | **Oracle L1**（需求数量单位） | S2/S3/S4 |
| `T2_oracle_L2_vs_P` | \(P\) | Oracle L2 | S2/S3/S4，辅助 |
| `T2_attainable_risk_vs_P` | \(P\) | 可达到 Pinball 超额风险 | S2/S3/S4，辅助 |
| `T2_pointmax_vs_P` | \(P\) | 测试点最大绝对误差 | S2/S3/S4，辅助 |

**不画未校准的绝对数值理论曲线 \(CP^{-1/32}\) 作为真实置信上界**，因为常数 \(C\) 未知；可在辅助图中只展示标注清楚的参考斜率。

### 4.5 T2 数据规模与优化平台判别

10,000 个 Oracle 输入是原协议已固定的主实验配置，而不是理论证明它足够覆盖高维输入。若观察到大宽度平台，应另建**事先命名的敏感性实验**（例如统一增加 Oracle 输入或训练 epoch），不得只对表现不好的某个 \(P\) 档位临时追加训练并混入主曲线。

对 S4，大量 Oracle 查询涉及正态积分。应缓存每份输入的 PMF 或分位数，并在训练前验证积分精度；缓存不是通过真实实现 \(U\) 生成的“更准确标签”。

### 4.6 T2 验收

- 每条 \(P\) 曲线内 \(M,B\)、S 场景、DGP 参数、拟合池样本量及优化预算固定。
- 使用 Oracle **函数值**作拟合标签，而不是当天实际 \(D\)。
- 随 \(H\) 增大，患者 MLP 与日级 FFN 同时增长；参数数目严格匹配表格。
- 同一重复不同 \(P\) 共用评价输入，模型从头初始化。
- 数据评价集完全独立；最终误差不可用于选择 checkpoint。
- 不能把有限 50,000 点的最大误差称为全域 \(L^\infty\) 误差。

## 5. T3：随 \(T\) 与 \(P_T\) 共同增长的风险/运营一致性诊断

### 5.1 增长规则与理论联系

在 S2、S3 各自固定的 DGP 下，采用预先确定的隐藏宽度

\[
\boxed{H(T)=\max\left\{1,\left\lfloor32\sqrt{T/2000}\right\rfloor\right\}.}
\]

两块增长 MLP 使用相同 \(H(T)\)，所以

\[
P_T=3721+65H(T)=O(\sqrt T).
\]

采用相同参数半径规则 \(R_{P_T}=100(P_T/5801)^2\)，固定潜维 20、SA/CA 头数、LN、模块数量。

由于理论保证 \(R_P\Lambda_P\) 至多为 \(P\) 的固定多项式，可得复杂度条件的**趋势形式**：

\[
\boxed{\frac{P_T\log(e+P_T)}T
=O\left(\frac{\log T}{\sqrt T}\right)\longrightarrow0.}
\]

同时 \(H(T)\to\infty\)，两块可增长 MLP 的宽度均发散。它与定理 10 的增长筛类构造方向相符，但不表示有限 300 epoch 的优化器自动满足推论 24 的精确 ERM 条件。

**本增长规则是原协议设定，不是推论 19 中平衡上界后得到的速率最优 \(P_T\) 选择。**

### 5.2 对照轨道与数据配对

每个 S2/S3 场景、每个数据重复都运行三个独立容量轨道：

1. `fixed32`：任何 \(T\) 都使用 \(H=32\)。
2. `fixed128`：任何 \(T\) 都使用 \(H=128\)。
3. `growing`：对每个 \(T\) 使用上述 \(H(T)\)。

**同一独立数据重复**必须先生成一条足够长的无限 i.i.d. 样本流的有限前缀，再对各 \(T\) 取嵌套前缀：

\[
S_{500}\subset S_{1000}\subset S_{2000}\subset\cdots.
\]

每条轨道、每个 \(T\) 都从头按预先固定的规则训练，按完整训练 Pinball 选择 checkpoint/初始化，不能因为某个测试点未下降就延长某条轨道训练。

### 5.3 T3 的两项核心纵轴

**A. 总超额风险**：

\[
\boxed{
\mathrm{ExcessRisk}_{T}^{\mathrm{track}}
=R_\tau(\tilde f_{P_T,T})-R_\tau(f^\star).
}
\]

**B. 运营遗憾**（理论诊断，不与性能实验仅报告 AC/SL 的规定冲突）：

\[
\boxed{
\mathrm{OperationalRegret}_{G,T}^{\mathrm{track}}
=C_G(\tilde f_{P_T,T})-C_G^\star.
}
\]

其中

\[
q_f=[f(Z)+G-I]_+,
\]

\[
C_G(f)=\mathbb E\left[\tau(G-I-q_f+D)_+
+(1-\tau)(I+q_f-D-G)_+\right],
\]

\(C_G^\star\) 使用完整信息 oracle 分位数诱导的最优可行策略来计算。由于合成成本中 \(b=\tau,h=1-\tau\)，有 \(b+h=1\)，故理论风险转移为

\[
\boxed{0\le C_G(f)-C_G^\star
\le R_\tau(f)-R_\tau(f^\star).}
\]

> **为使绘图合同可运行，新增且明确标记的执行约定**：T3 运营遗憾主图选择原 \(G\) 网格中央值 **\(G=300\)**，另在独立辅助图中展示原五个 \(G\) 的曲线/敏感性；原始 MD 没有指定 T3 的单一主图 \(G\)。这只是图表选择，不改变 DGP 或理论。

运营遗憾可用条件风险积分计算，从而避免给定 \(Z\) 后单次需求抽样噪声。所有轨道在同一个评价 \(Z\) 池上比较。

### 5.4 T3 最终作图

| 图 | 横轴 | 主纵轴 | 对照 |
|---|---|---|---|
| `T3_risk_convergence` | **训练日数 \(T\)** | **总超额 Pinball 风险** | fixed32、fixed128、growing |
| `T3_operational_regret` | **训练日数 \(T\)** | **相对 oracle 策略的运营遗憾**（默认 \(G=300\)） | 三容量轨道 |
| `T3_complexity_proxy` | \(T\) | \(P_T\log(e+P_T)/T\) | 三轨道，辅助 |
| `T3_P_and_norm` | \(T\) | 实际 \(P_T\) 与全局参数范数（分图显示） | 辅助审计图 |

注意同一个 \(\tau\) 任务的风险上界在一个概率事件上同时控制所有**确定性 \(G\)**；但这不意味着所有 \(G\) 的数值曲线相同，也不意味着有限 Monte Carlo 误差下不会看到微小表观违背。

### 5.5 T3 验收

- S2、S3 各自 DGP 保持固定，不得随 \(T\) 改变 \(M\)、场景参数或需求上界。
- Growing 轨道只增长 Patient MLP 与 Day FFN，不增加潜维、层数和头数。
- 三条轨道在同一重复内共用嵌套前缀，评价使用同一独立输入池。
- 记录实际 \(P_T\)、全局参数范数与 \(P_T\log(e+P_T)/T\)；该比例是**复杂度代理**，不是完整高概率界数值。
- 固定容量曲线是否出现平台是实验问题，不预设结果；不能将有限曲线称为数学意义的“一致性证明”。

## 6. 实验统计、输出表格、随机种子

### 6.1 统计单位与聚合

**独立数据重复为统计单位**。在重复 \(r\) 内，三个初始化按事先规定的训练目标挑选一个，不得把它们当作三份独立的 \(Z,D\) 样本。

对每个 \(T\)、\(P\) 和场景，保存逐重复结果及在 200 份独立重复上的均值、中位数、90%/95% 分位数；在适合的图中加入独立重复置信区间。不得用反复重采样同一数据集的种子数量冒充统计重复数。

### 6.2 建议的中间记录字段

```text
experiment_id, run_id, scenario_id, replicate_id, train_seed,
T, H_pat, H_ffn, P_actual, R_ball, theta_norm,
tau, G_if_applicable,
train_objective, selection_rule,
excess_pinball_risk, oracle_L1, oracle_L2, eval_point_max,
operational_regret, complexity_proxy,
confidence_delta, theoretical_bound, bound_exceeded,
oracle_integration_accuracy, evaluation_MC_accuracy
```

并非每组都填写全部字段：T1 必须有概率界及越界指标；T2 必须有 Oracle 拟合误差；T3 必须有总体超额风险及运营遗憾。Oracle **评价指标**可保存，不等于把 Oracle 作为训练特征。

### 6.3 主要图表验收一览

| 实验 | 横轴 | 纵轴 | 类型 |
|---|---|---|---|
| T1-C0 | \(T\) | 真正估计超额 Pinball 风险 + 上界 | **高概率有限样本校准** |
| T1-C0 | \(T\) | 经验上界越界率（对比 \(\delta\)） | **概率校准** |
| T1-DAT | \(T\) | 完整 DAT 总超额 Pinball 风险 | **数值学习曲线** |
| T2 | \(P\) | Oracle Quantile L1 | **可达到函数逼近诊断** |
| T3 | \(T\) | 总超额风险（固定/增长） | **风险收敛诊断** |
| T3 | \(T\) | 运营遗憾（固定/增长） | **运营收敛诊断** |

## 7. 运行流程与可复现性

原协议约定的统一入口：

```bash
python scripts/run_synthetic_experiments.py --family all --smoke --name check
python scripts/run_synthetic_experiments.py --family theory --name theory_main
```

单组运行 `--group` 名称应从当前 `synthetic/README.md` 或实际 CLI 帮助读取，不能在本文中凭空假设已有特定 group flag 值。

**运行步骤**：

1. 先执行 `check_protocol.py`，核对 DAT 参数量、LayerNorm 数目、置换与 Mask 测试、投影范数。
2. 进行 20 次 Smoke/耗时试运行，验证数据权限和 Oracle 积分误差；这些输出不得混入正式统计。
3. 固定协议、配置文件版本、代码 commit、随机种子体系，再运行正式独立重复。
4. 保存逐重复结果、数值积分精度、图表与原始运行清单。
5. `--resume` 应校验已完成文件和运行配置指纹；代码/配置改变时禁止把旧结果合并入新运行。
6. Git 只保留代码和协议；生成需求、模型权重、缓存、试验原始明细和大型结果保留本机。

## 8. 论文解释边界

- T1-C0 直接检验**精确 ERM 的可实现校准子类**，不等于验证整个 DAT 的全局优化性质。
- T1-DAT 的总风险差含逼近与优化成分；固定类下 \(T\to\infty\) 时不要求风险趋于零。
- T2 用的是**Oracle 函数拟合**，目标是隔离需求随机标签噪声；不能被包装成正常训练中的 Oracle 辅助预训练。
- T3 的增长类条件来自**函数类表达能力和复杂度控制**；AdamW 的有限迭代一致性并未由原定理给出。
- S1–S4 的趋势可能被需求均值和方差的改变共同影响；不要把所有跨场景差异解释为 DAT 模块的单一贡献。
- 以上实验不存在“实验已经证明定理正确”的逻辑；应称为与定理相对应的**数值诊断、有限样本校准和收敛趋势证据**。

---

**定稿口径**：T1 使用严格区分的 C0 精确估计误差与完整 DAT 总风险，T2 使用固定 10,000 Oracle 输入的容量曲线，T3 使用 \(H(T)\propto\sqrt T\) 的增长筛类与固定宽度对照。理论实验不增加性能实验之外的其他真实医院任务。
