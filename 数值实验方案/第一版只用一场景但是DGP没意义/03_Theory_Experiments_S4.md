# 03｜DAT Synthetic 理论实验规范：T1、T2、T3（S4 Only）

> **状态**：2026-11-05 理论数值实验正式合同。  
> **主 DGP**：T1-DAT、T2、T3 **全部只用 S4**；T1-C0 单独使用常数可实现校准 DGP。  
> **主分位数**：固定 `tau=0.9`。其他分位数仅为独立标记的辅助实验，不进入主结论。  
> **训练配置**：优化器、学习率、Batch、轮数、梯度裁剪及模型规格均在正式运行前人为固定并保存；**不做超参数搜索**。本文件只写数学必要的模型规模、数据规模、实验变量和统计规则，不逐项列训练超参数具体值。  
> **解释边界**：数值神经网络训练通常不是精确 ERM；理论定理不是由有限曲线“证明”的。医院真实数据与本组实验完全隔离。

---

## 1. 三组实验对应的数学问题

令：

\[
R_\tau(f)=\mathbb E[\rho_\tau(D-f(Z))],\qquad
\rho_\tau(e)=\tau[e]_++(1-\tau)[-e]_+.
\]

真实完整信息 Bayes 预测为：

\[
\boxed{f^\star(Z)=Q_{0.9}(D\mid Z;\mathrm{S4}).}
\]

对于抽样前固定的紧 DAT 函数类 `F_P`：

\[
f_P^\dagger\in\operatorname*{argmin}_{f\in\mathcal F_P}R_\tau(f),\qquad
\hat f_{P,T}\in\operatorname*{argmin}_{f\in\mathcal F_P}R_{\tau,T}(f).
\]

`hat f` 表示数学上的**精确 ERM**；实际固定数值训练所得模型记为 `tilde f`，在没有全局最优证书时不要混用。

定义：

\[
\mathcal E_{P,T}^{\rm est}=R_\tau(\hat f_{P,T})-R_\tau(f_P^\dagger),\quad
\mathcal E_P^{\rm app}=R_\tau(f_P^\dagger)-R_\tau(f^\star).
\]

精确 ERM 有：

\[
R_\tau(\hat f_{P,T})-R_\tau(f^\star)=\mathcal E_{P,T}^{\rm est}+\mathcal E_P^{\rm app}.
\]

| 实验 | 唯一主变化 | 横轴 | **正文主纵轴** | 理论定位 |
|---|---|---|---|---|
| **T1-C0** | 常数类训练日数 `T` | `T` | 精确估计超额 Pinball 风险及其高概率界 | 可实现精确 ERM、逼近误差为 0 |
| **T1-DAT** | 固定 DAT 的训练日数 `T` | `T` | 相对 S4 Oracle 的**总**超额 Pinball 风险 | 固定容量学习曲线；含逼近和数值优化影响 |
| **T2** | DAT 总参数预算 `P` | `P` | Oracle Quantile L1 距离 | 可达到的构造性逼近能力诊断 |
| **T3** | 训练日数 `T` 同时驱动 `P_T` 增长 | `T` | 总超额 Pinball 风险；辅图为运营遗憾 | 风险与运营收敛趋势 |

**注意**：T2 的纵轴是需求分位数**预测值差**；T1、T3 的主要纵轴是 Pinball **风险差**，两者不能混为一个误差。

## 2. 所有理论实验共享的环境

### 2.1 DGP、结构及主分位数

理论 DGP 使用：

\[
M=2,\ d_a=3,\ d_x=7,\ B=18.5,\ d_{\rm model}=20,\ \tau=0.9.
\]

S4 机制参数：

\[
(\lambda_{\rm ctx},\lambda_{\rm peer},\rho)=(1,1.5,0.8).
\]

生成器及真实条件分位数详见 `02_Synthetic_DGP_S4.md`。T1-C0 是本规则的**专用例外**，使用已知常数条件分位数环境。

对 S4 的任一 `Z`，必须**先在给定 U 下做联合离散卷积，再将 PMF 对 U 正态积分**，得到数值 Oracle。不能把实现的随机 `U` 输入到 Oracle，也不能把数值积分当成无误差解析计算。

### 2.2 正式实验规模与种子层级

按原始理论协议保留以下**正式目标规模**；若算力不足，可另设明确标记的 Pilot，不得将 Pilot 重复数写成正式结果：

| 项目 | 规则 |
|---|---|
| T1/T3 的训练日数 | `500, 1000, 2000, 4000, 8000, 16000` |
| 更大样本扩展 | 仅在事前登记后独立报告 |
| 统计数据重复 | **200 份相互独立的 DGP 训练样本** |
| 非凸模型数值初始化 | 每份训练样本预先设定 **3 次初始化**，按其**训练目标**选用，不按测试结果选用 |
| 正常风险评价 | 约 **50,000** 个独立完整信息 `Z`，使用条件期望 Pinball 风险 |
| T2 Oracle 拟合输入 | 每份重复 **10,000** 个独立 `Z` 与其 `f*(Z)` |
| T2 独立评价输入 | **50,000** 个独立 `Z` |
| 数据及随机状态 | 分离 `data_seed`、`initialization_seed`、`evaluation_seed`、数值 Oracle 积分配置 |

注意：`200 次数据重复 × 3 次初始化` **不意味着 600 份独立训练数据**；统计区间的基本独立单元仍然是 200 个数据重复。

为了控制计算量，正式运行可以采用事先定义的分阶段/断点续跑机制；不得在测试结果出来后为某个容量、某个 T 档位单独追加训练以改善曲线。

### 2.3 总体风险的条件期望评价

S4 Oracle 的离散主体 PMF 为：

\[
p_k(z)=\Pr(D_0=0.5k\mid Z=z),\quad D=D_0+J,\quad J\sim U(0,0.5).
\]

给定预测 `y`：

\[
\boxed{\phi_z(y)=\mathbb E[\rho_\tau(D-y)\mid Z=z]
=\sum_kp_k(z)\,2\int_0^{0.5}\rho_\tau(0.5k+j-y)\,dj.}
\]

内层积分可以分段解析计算；总体风险通过独立输入池近似：

\[
\widehat R_\tau^{\rm cond}(f)
=\frac1{N_e}\sum_{j=1}^{N_e}\phi_{Z_j}(f(Z_j)).
\]

对模型与 Oracle **使用同一个评价输入池**。这消除了给定输入后单次随机需求实现造成的评价噪声，但仍有 `Z` 的 Monte Carlo 误差以及 S4 积分数值误差。

---

## 3. T1：随样本量 T 的有限样本统计实验

### 3.1 T1-C0：可精确求解的高概率界校准

C0 条件需求分布不依赖 `A,X`；它通过固定用血发生概率、用血数量概率和新增订单概率产生连续需求（具体分布在 DGP 文档 §5）。因此：

\[
f^\star(Z)=q_\tau^\star\quad\text{恒为一个常数}.
\]

选择：

\[
\mathcal F_{\rm const}=\{f_c(Z)\equiv c:0\le c\le B\}.
\]

这个校准子类只有一个自由参数，故**校准中使用 `P=1`，不是完整 DAT 的参数数目**。输入和 `Z` 无关，参数预测 Lipschitz 常数可取 `Lambda=1`；参数半径取覆盖 `[0,B]` 的预先固定数值。

用 `T` 天需求计算样本分位数：

\[
\boxed{\hat q_{\tau,T}=\inf\{y:\hat F_T(y)\ge\tau\}.}
\]

这是常数类上精确的 Pinball ERM，且 Oracle 就属于该类，所以逼近误差为零：

\[
\boxed{Y_{T,r}^{\rm C0}=R_\tau(\hat q_{\tau,T}^{(r)})-R_\tau(q_\tau^\star)=\mathcal E_{1,T}^{\rm est}.}
\]

因为 C0 的条件需求分布是固定分布，这个总体风险可由该分布的 PMF 与均匀余量**直接数值/解析积分**，不依赖有限 `Z` 评价池。

#### 高概率上界及经验越界率

取 `L_tau=max(tau,1-tau)=0.9`，函数类的预测跨度上界 `Delta<=B`。对 `Delta>0`，可用原理论命题 9 的闭式上界（且再取平凡界）：

\[
\boxed{
\begin{aligned}
\epsilon_{P,T}(\delta)=\min\Bigg\{L_\tau\Delta,\ &6L_\tau\Delta\sqrt{P/T}
\left[\sqrt{\log(1+8R_P\Lambda_P/\Delta)}+\frac{\sqrt\pi}{2}\right]\\
&+L_\tau\Delta\sqrt{2\log(1/\delta)/T}\Bigg\}.
\end{aligned}}
\]

在 C0 中代入 `P=1`、`Lambda=1`、合法固定 `R`、`Delta=B`（保守合法）。也可使用原命题更细致的截断 Dudley 积分上界，但需要整个实验统一使用同一个预先固定的版本。

对 `delta∈{0.10,0.05}`，利用 200 份独立训练数据，计算：

\[
\boxed{\hat v_\delta(T)=\frac1{200}\sum_{r=1}^{200}
\mathbf1\{Y_{T,r}^{\rm C0}>\epsilon_{1,T}(\delta)\}.}
\]

理论含义是单次随机训练样本抽取时，真实越界概率至多为 `delta`；200 次独立重复上的观察频率是一个随机估计值，**不必**恰好等于 5% 或 10%。由于界可能保守、甚至退化为平凡界，零次越界不能解释为界紧。

#### T1-C0 最终图

| 图名 | 横轴 | 纵轴 | 内容 |
|---|---|---|---|
| `T1_C0_estimation_bound` | 训练日数 `T` | **精确估计超额 Pinball 风险** | 200 次重复的均值/90%/95% 经验分位数，叠加相应高概率界 |
| `T1_C0_bound_exceedance` | 训练日数 `T` | **经验越界比例** | `delta=0.05/0.10` 的越界比例及水平参考线 `y=delta` |

### 3.2 T1-DAT：固定容量下的完整学习曲线

**DGP：仅 S4**，不再进行 S2/S3 的跨场景比较。

固定 DAT 架构与参数球，例如沿用理论基准：

\[
H_{\rm patient}=H_{\rm dayFFN}=32,\quad P=5801,\quad R_P=100.
\]

这里的 `H=32`、`P=5801`、`R=100` 是已定理论函数类身份的一部分，**不是重新搜索的训练超参数**。对每一个 `T`，只改变训练样本数，使用同一个 S4 需求机制和预设训练规则。

记实际神经优化训练出的函数为 `tilde f_{P,T}`。在独立评价池上计算：

\[
\boxed{Y_{T,r}^{\rm DAT}=R_\tau(\tilde f_{P,T}^{(r)})-R_\tau(f^\star).}
\]

该指标是**总超额风险**，其中可含固定逼近差距、统计波动和未认证优化误差。它不等于命题 9 中针对精确 ERM 的纯估计误差，不能直接用仅估计项的界宣称覆盖该曲线。

| 图名 | 横轴 | 纵轴 | 分组 |
|---|---|---|---|
| `T1_DAT_total_excess_S4` | 训练日数 `T` | `R_tau(tilde f)-R_tau(f*)` | 仅 S4，跨独立重复的均值及区间 |

**预期问题**：随着样本量增大，总风险差是否减小、是否在固定类容量下出现平台；平台并非数学上必然在有限区间出现。

---

## 4. T2：随参数预算 P 的 Oracle 函数逼近诊断

### 4.1 与定理 10 的联系

显式逼近定理在固定 `M`、目标 Hölder 正则性、潜维和 Value 宽度条件下建立存在性保证：

\[
\boxed{\inf_{f\in\mathcal F_P}\|f-f^\star\|_{\infty,K}\le CP^{-\gamma},\quad
\gamma=\frac{\alpha}{M d_{\rm eff}}.}
\]

在本理论环境 `M=2`、`d_eff=16`、S4 条件分位数各人数切片可以取 Hölder 指数 `alpha=1`，故：

\[
\boxed{\gamma=1/32.}
\]

这只是一个保守存在性速率；实验不会直接求网络类上的精确全局最优逼近误差，也不以该幂次强制拟合实验斜率。

### 4.2 变量、固定项及数据规模

- **DGP：仅 S4**；所有容量档位的条件分位数目标相同。
- **拟合输入数**：每个独立重复 `10,000` 个完整信息 `Z`，以数值 Oracle `f*(Z)` 为目标标签；**不使用随机实现的需求 D**。
- **评价输入数**：独立 `50,000` 个 `Z`，不同容量使用同一测试输入池。
- **变化量**：患者编码器和 Day FFN 的共同隐藏宽度 `H`；Attention 层数、潜维、头数、输入和 DGP 均固定。
- **参数类**：采用预先声明的随 `P` 增长的闭参数球；原理论筛类可采用 `R_P=100(P/5801)^2`。这是理论容量协议，不是从测试性能搜索得到的正则参数。

标准实现的参数计数：

\[
\boxed{P(H)=3721+65H.}
\]

| 共同隐藏宽度 `H` | 16 | 32 | 64 | 128 | 256 | 512 |
|---:|---:|---:|---:|---:|---:|---:|
| 实际参数量 `P` | 4,761 | 5,801 | 7,881 | 12,041 | 20,361 | 37,001 |

T2 横轴必须是**实际 `P`**，不能把模型宽度 `H` 直接冒充参数预算。保持同一固定拟合池规模才可以解释成随模型容量变化的可达到拟合曲线。

### 4.3 训练与评价

专用 T2 的训练目标为：

\[
\boxed{\min_{\theta\in\Theta_P}\frac1{10000}\sum_{i=1}^{10000}
[f_\theta(Z_i)-f^\star(Z_i)]^2.}
\]

这是为了隔离实际随机需求标签噪声的**Oracle 拟合诊断**；与正常 DAT 的真实 `D` 上 Pinball 训练不同。非凸模型的初始化和训练规则事先固定，不用 Oracle 评价池选最优初始化或训练参数。T2 训练权重不得用于其他组。

主纵轴：

\[
\boxed{\mathrm{OracleL1}(P)=\frac1{50000}\sum_{j=1}^{50000}
|\tilde f_P(Z_j)-f^\star(Z_j)|.}
\]

可保存作为辅助理论诊断的：

\[
\mathrm{OracleL2}(P)=\left\{\frac1{50000}\sum_j[\tilde f_P(Z_j)-f^\star(Z_j)]^2\right\}^{1/2},
\]

\[
\mathrm{EvalPointMax}(P)=\max_j|\tilde f_P(Z_j)-f^\star(Z_j)|,
\]

\[
\mathrm{AttainableExcessRisk}(P)=R_\tau(\tilde f_P)-R_\tau(f^\star).
\]

必须如实称呼：`OracleL1` 是**训练所得网络与 Oracle 的平均绝对预测差**；`EvalPointMax` 不是全定义域的 sup 范数；`AttainableExcessRisk` 不是未知的类内最优逼近误差。

| 图名 | 横轴 | 主纵轴 |
|---|---|---|
| `T2_S4_oracleL1_by_P` | 实际参数预算 `P`（建议对数轴） | `OracleL1(P)` |
| `T2_S4_oracleL2_by_P`（辅助） | `P` | `OracleL2(P)` |
| `T2_S4_attainableRisk_by_P`（辅助） | `P` | 可达到 Pinball 风险差 |

不把未知常数 `C` 的 `CP^{-1/32}` 画成具有绝对数值含义的可信界；若作斜率参考线必须标注只是参考指数。

---

## 5. T3：T 与 P_T 同时增长的一致性诊断

### 5.1 固定 S4 与预定增长规则

T3 只使用 S4。训练日数使用 T1 相同的预先固定网格。

共同隐藏宽度按：

\[
\boxed{H(T)=\max\{1,\lfloor32\sqrt{T/2000}\rfloor\}.}
\]

标准实现下：

\[
P_T=3721+65H(T)=O(\sqrt T).
\]

参数球半径仍按预先声明的 `R_{P_T}` 增长；潜维 20、模块数量、Attention 头数不增长。因此：

\[
\frac{P_T\log(e+P_T)}{T}=O\!\left(\frac{\log T}{\sqrt T}\right)\to0.
\]

该式用于展示统计复杂度条件的渐近趋势，**不是**完整的有限样本风险上界数值；固定数值优化算法也未因此自动获得精确 ERM 的一致性保证。

### 5.2 三条预设轨道

| 轨道 | 隐藏宽度 | 含义 |
|---|---|---|
| `fixed32` | 对所有 `T` 固定 `H=32` | 小型固定容量 |
| `fixed128` | 对所有 `T` 固定 `H=128` | 较大固定容量 |
| `growing` | `H=H(T)` | 随数据量增长的筛类 |

在每个独立重复中，先生成一条 S4 的足够长 i.i.d. 样本流，再取嵌套训练前缀：

\[
S_{500}\subset S_{1000}\subset S_{2000}\subset\cdots.
\]

三条轨道在同一重复、同一 `T` 使用**同一训练前缀与独立评价输入**，各模型从头开始训练；不从小容量预训练迁移到大容量模型，也不根据评价曲线临时延长某个训练过程。

### 5.3 两类主诊断

**(a) 总超额 Pinball 风险：**

\[
\boxed{\mathrm{ExcessRisk}_{T,\mathrm{track}}
=R_\tau(\tilde f_{P_T,T})-R_\tau(f^\star).}
\]

**(b) 运营遗憾：** 主图使用**预先固定的**目标 `G=300`，这是原目标网格的中央值；其他 G 仅为独立辅助分析。

\[
q_f(Z)=[f(Z)+G-I]_+,\quad
C_G(f)=\mathbb E\left[\tau[G-(I+q_f-D)]_+ +(1-\tau)[(I+q_f-D)-G]_+\right].
\]

\[
\boxed{\mathrm{OperationalRegret}_{G,T}=C_G(\tilde f_{P_T,T})-C_G(f^\star).}
\]

为了计算条件运营成本，定义：

\[
y_f(Z)=I+q_f(Z)-G=\max\{f(Z),I-G\}.
\]

于是每个输入的条件运营成本正好为：

\[
\boxed{\mathbb E[c_{\tau,G}(q_f,D,I)\mid Z]=\phi_Z(y_f(Z)).}
\]

因此可以在**同一独立评价输入池**用条件 PMF 算出模型及 Oracle 的期望成本再求差，减少额外的需求抽样噪声。

在合成归一化成本 `b=tau,h=1-tau` 下，理论风险转移有：

\[
\boxed{0\le C_G(f)-C_G(f^\star)\le R_\tau(f)-R_\tau(f^\star).}
\]

若数值求积或独立输入池有限，极小表观偏差可能来自数值误差，必须检查求积精度。

| 图名 | 横轴 | 纵轴 | 曲线 |
|---|---|---|---|
| `T3_S4_risk_convergence` | 训练日数 `T` | 总超额 Pinball 风险 | fixed32 / fixed128 / growing |
| `T3_S4_operational_regret` | 训练日数 `T` | 相对 Oracle 的目标成本遗憾 | fixed32 / fixed128 / growing |
| `T3_S4_complexity_proxy`（辅助） | `T` | `P_T log(e+P_T)/T` | fixed / growing 分轨迹 |

需要区分：理论实验中使用运营遗憾作为**定理诊断**，并不要求性能实验 P1–P3 把它作为第三个指标。性能实验最终仍只有 MeanCost 与 SL_G。

---

## 6. 结果聚合与统计可信度

1. **统计单位是独立数据重复**，不是训练初始化数、Oracle 积分节点数或多个 `G` 结果。
2. 在 T1-DAT、T2、T3 中，同一份训练数据的多个初始化按照预先固定的**训练目标**确定拟合结果，不准看测试/Oracle 评价结果挑选。
3. 对 200 个独立重复报告均值、适当的分位数和不确定性区间。T1-C0 应记录完整的超界事件，而不只是均值。
4. 同一重复内、不同 T/P/轨道使用尽量配对的评价输入池，减少无关抽样差异。
5. 理论模拟中一条函数逼近曲线的平滑、单调或 `T→∞` 的极限不能由有限计算精确证明；报告数值趋势时明确这一点。

### 推荐逐重复结果字段

```text
experiment_id, replicate_id, data_seed, initialization_seed,
scenario_id, tau, T, model_track, H_patient, H_dayFFN,
P_actual, parameter_radius, parameter_norm,
train_objective, selection_rule,
oracle_L1, oracle_L2, evaluation_point_max,
excess_pinball_risk, operational_regret,
confidence_delta, high_probability_bound, bound_exceeded,
complexity_proxy, oracle_quadrature_accuracy, evaluation_MC_accuracy
```

某些字段只适用于对应实验，如 T1 才填高概率界、T2 才填 OracleL1；不能把空字段补成伪造的零。

## 7. 必做的理论实验验收

| 编号 | 验收项 |
|---|---|
| T01 | T1-DAT/T2/T3 全部 S4；C0 是独立专用校准，而非另一个性能场景 |
| T02 | 所有主实验 `tau=0.9`，辅助分位数使用单独运行标识 |
| T03 | `M=2,B=18.5,d_model=20`；完整信息 `X` 含 `N/M`，没有重复拼接 |
| T04 | C0 精确样本分位数、`P=1`、逼近误差 0，高概率界及越界率计算正确 |
| T05 | T1-DAT 固定 `P=5801` 和同一紧参数球，只改变 `T` |
| T06 | T2 每档 10,000 个 Oracle 拟合输入、50,000 个独立评价输入；变化量只有容量/范数包络 |
| T07 | T2 `P(H)=3721+65H` 与实际代码稠密参数计数一致 |
| T08 | T3 `fixed32/fixed128/growing` 三轨道在每次重复内共用嵌套 i.i.d. 前缀 |
| T09 | S4 Oracle 先卷积后对隐藏 `U` 积分，记录积分精度 |
| T10 | T1-DAT/T3 的正常训练只能使用真实需求 `D`；Oracle 标签只用于 T2 特殊拟合 |
| T11 | 不搜索超参数；初始化/训练规则由拟合目标确定，绝不利用测试性能选模型 |
| T12 | 正式重复数和任何 Pilot 的实际重复数分别报告，不混淆 |

## 8. 运行及复现规则

```bash
python scripts/run_synthetic_experiments.py --family all --smoke --name s4_smoke
python scripts/run_synthetic_experiments.py --family theory --name theory_s4_main
```

上述为**预期统一入口样式**；实际命令是否支持这些标志取决于仓库实现，不代表已确认所有脚本存在。正式代码需自动保存配置快照、代码版本、独立种子、每次结果以及 S4 数值积分精度。`--resume` 必须比较配置指纹；不同协议的旧结果不得混入新版汇总。

---

**一句话总结：T1 固定类研究 T 的统计影响；T2 固定 S4 和拟合池研究 P 的表达能力；T3 固定 S4，研究 T 与 P_T 联合增长时风险和运营遗憾的趋势。**
