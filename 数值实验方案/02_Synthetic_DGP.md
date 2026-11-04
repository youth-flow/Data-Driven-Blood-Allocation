# 合成 DGP：AXID 数据生成机制与 Oracle 计算规范

> **文档状态**：Synthetic DGP 冻结版（2026-11-04 整理）  
> **依据**：最初上传的 `synthetic_protocol.md` 第 1–3、7–8 节；已确认**不调整原始库存、需求和目标库存数量尺度**。  
> **范围**：理论实验 T1–T3 和性能实验 P1–P3 共用的合成生成机制；**不使用任何医院真实数据、历史补货记录、真实模型权重或历史收货策略**。  
> **不可混淆**：DGP 是生成**真实随机需求**的概率机制，不是 DAT，不是神经网络 Teacher。性能实验只有 `AC` 与 `SL_G` 两个最终评价指标，定义见第四号文档。

## 1. 固定设计原则

1. **每天独立同分布**：不同日的全部外生随机性独立；日期/序号仅用作 ID，不输入可预测的趋势或滚动日期。
2. **决策时信息隔离**：DAT 只知道当天事前患者集合 \(A\) 与日级信息 \(X\)，不观察未来共同冲击 \(U\)、患者实际用血 \(Y_i\)、新增患者实际需求或连续扰动 \(J\)。
3. **两种人数规模，同一生成公式**：理论环境最大可见人数 \(M=2\)，性能环境 \(M=8\)。除必要的 \(M,B,d_{\mathrm{mod}}\) 配置不同，生成公式不变。
4. **DGP 不随 \(\tau,G\) 改变**：\(\tau\) 是成本不对称/预测分位数，\(G\) 是期末库存目标。它们只影响训练任务、补货策略或评价，不能回流影响 \(D\) 的生成。
5. **连续需求主版本**：先产生半单位网格需求 \(D_0\)，再加入独立 \(J\sim U(0,0.5)\)。离散版 \(J=0\) 仅为单列稳健性扩展，不混入主结果。
6. **数量尺度冻结**：库存为 \(I=315+20c_1+U(-70,70)\)，性能目标为 \(G\in\{260,280,300,320,340\}\)；不采用曾讨论的低库存缩放方案。
7. **同一随机重复的场景配对**：S1–S4 共用已生成的 \(A,X,I\)，改变需求机制参数；改变机制后重新抽取对应的结果随机量。

## 2. 固定配置表

| 参数 | 理论环境 | 性能环境 | 含义 |
|---|---:|---:|---|
| \(M\) | 2 | 8 | 决策前最多可见的患者/订单数 |
| \(d_a\) | 3 | 3 | 每名患者输入维度 |
| \(d_x\) | 7 | 7 | 不含人数 \(N/M\) 的日级维度 |
| 完整 \(X\) 维数 | 8 | 8 | 实际输入 Day Encoder 的维度 |
| \(K_{\mathrm{new}}\) | 4 | 4 | 新增订单数量上界；不要与需求目标 \(G\) 混淆 |
| \(B\) | 18.5 | 36.5 | 决策日总需求的确定性上界 |
| DAT \(d_{\mathrm{mod}}\) | 20 | 64 | 模型配置，不参与需求概率公式 |
| \(\tau\) | 理论统一 0.9 | 0.5、0.6、0.7、0.8、0.9 | 分位数成本；0.95 为尾部扩展 |
| \(G\) | 只在理论运营图中指定 | 260、280、300、320、340 | 期末目标库存 |

为避免与理论文稿 \(\alpha\)（Hölder 指数）及 \(\gamma\)（逼近率指数）冲突，本文件在解释中将**原始 MD 中的 DGP 参数** `alpha`、`gamma` 称为 \(\lambda_{\mathrm{ctx}}\)、\(\lambda_{\mathrm{peer}}\)。仅是**记号替换**，场景数值、随机分布和已有配置键无需因此改变。

## 3. 一个决策日的完整生成过程

以下随机量属于同一天；除了明确的条件依赖以外，各抽样调用使用独立随机性。不同天之间全部独立。

### 3.1 生成计划情境、急诊压力和期初库存

首先独立抽取

\[
c_1,c_2\overset{i.i.d.}{\sim}\operatorname{Uniform}(-1,1),
\qquad e_I\sim\operatorname{Uniform}(-70,70).
\]

定义

\[
\boxed{I=315+20c_1+e_I,\qquad 225\le I\le405.}
\]

- \(c_1\)：计划业务强度，影响当天可见人数、库存及部分需求概率。
- \(c_2\)：急诊压力，影响订单条件需求及新增需求。
- \(I\)：**决策前可见**的期初库存；在当天需求生成中不被需求结果反向决定。
- 不执行从前一天剩余库存到下一天库存的递推。
- 条件于 \(c_1,c_2\) 后，独立库存扰动 \(e_I\) **不提供额外的条件需求信息**，但 \(I\) 是补货决策的必要状态量。

### 3.2 生成人数 \(N\)

记 Sigmoid 函数 \(s(x)=1/(1+e^{-x})\)。令

\[
\boxed{N=1+\operatorname{Binomial}\big(M-1,s(-0.2+0.4c_1)\big).}
\]

于是对每个日级样本有

\[
1\le N\le M.
\]

对于固定环境 \(M\)，人数分布受 \(c_1\) 影响；场景 S1–S4 **不改变**该人数生成公式。

### 3.3 生成可见患者 \(a_i=(u_i,v_i,r_i)\)

对每个 \(i=1,\ldots,N\)，先抽取

\[
r_i\sim\operatorname{Beta}(2,3).
\]

再条件于 \(r_i\)，分别抽取

\[
\boxed{
\begin{aligned}
u_i&=\frac{\operatorname{Binomial}(12,0.25+0.25r_i)}{12},\\
v_i&=\frac{\operatorname{Binomial}(16,0.15+0.20r_i)}{16}.
\end{aligned}}
\]

> **记号提示**：这里的 \(u_i\) 是患者的申请量归一化特征，不是日级编码向量 \(u=\phi_{\mathrm{day}}(X)\)。实现中可以分别命名为 `patient_u` 与 `day_embedding`。

每名患者输入：

\[
\boxed{a_i=(u_i,v_i,r_i)\in[0,1]^3.}
\]

| 变量 | 实际解释 | 数据可见时间 |
|---|---|---|
| \(6u_i\) | 物理申请量 | 决策前 |
| \(8v_i\) | 备血量 | 决策前 |
| \(r_i\) | 已知临床风险评分 | 决策前 |
| \(Y_i\) | 真实发生的用血需求 | 决策后；不属于 \(a_i\) |

协议**没有**规定 \(8v_i\ge6u_i\)，也没有规定实际 \(Y_i\) 必须小于单名患者申请量。不能在数据清洗中擅自添加此类不等式约束。

构造无序多重集合

\[
\boxed{A=\{a_1,\ldots,a_N\}.}
\]

患者行的输入顺序不影响后续生成的条件分布与 Oracle；不能用排序位置生成结果。

### 3.4 构造完整八维日级输入 \(X\)

\[
\boxed{
X=\left(
\frac{c_1+1}{2},
\frac{c_2+1}{2},
\frac{I-225}{180},
\frac NM,
\frac{\sum_i u_i}{M},
\frac{\sum_i v_i}{M},
\frac1N\sum_i r_i,
\frac1N\sum_i r_i^2
\right).
}
\]

所有坐标在 \([0,1]\) 内；第 4 坐标为显式人数比例 \(N/M\)。

| 0-based 索引 | 名称 | 可从哪些事前量计算 |
|---:|---|---|
| 0 | `c1_normalized` | \(c_1\) |
| 1 | `c2_normalized` | \(c_2\) |
| 2 | `inventory_normalized` | \(I\) |
| 3 | `count_ratio` | \(N/M\) |
| 4 | `application_summary` | \(\sum_i u_i/M\) |
| 5 | `reserve_summary` | \(\sum_i v_i/M\) |
| 6 | `risk_mean` | \(N^{-1}\sum_i r_i\) |
| 7 | `risk_second_raw_moment` | \(N^{-1}\sum_i r_i^2\) |

从输入可恢复原单位库存：\(I=225+180X_3\)（这里数学下标从 1 开始；代码 `X[:,2]`）。

这里同时提供逐患者输入和部分患者汇总，**不构成未来数据泄漏**，但会影响 P3 信息层级的解释。各层级授权范围见第四号文档。

### 3.5 构造同群风险 \(\mathrm{peer}_i\)

当 \(N=1\) 时，直接取

\[
\mathrm{peer}_1=0.
\]

当 \(N>1\) 时，对 \(j\ne i\) 定义

\[
w_{ij}=\exp\left(-\frac{(u_i-u_j)^2}{2(0.25)^2}\right)
\]

及

\[
\boxed{
\mathrm{peer}_i=\frac{\sum_{j\ne i}w_{ij}r_j}{\sum_{j\ne i}w_{ij}}.
}
\]

由于 \(u_i,u_j\in[0,1]\) 且权重严格为正，\(N>1\) 时分母不为零；有限人数还保证分母具有统一正下界。

这是一个**给定可见患者特征后即可确定的统计关系**；不涉及结果 \(Y_j\)，也不能被描述为经过识别的临床因果效应。

### 3.6 可见患者的用血发生与用血数量

对患者 \(i\)，定义发生概率 Logit

\[
\boxed{
\begin{aligned}
\ell_i={}&-2+0.35\log(1+6u_i)+0.15\log(1+8v_i)+1.2r_i\\
&+\lambda_{\mathrm{ctx}}(0.8c_1u_i+0.6c_2r_i)
+\lambda_{\mathrm{peer}}r_i\mathrm{peer}_i.
\end{aligned}}
\]

用血数量概率参数

\[
\boxed{
\pi_i=s\left(-0.5+0.3\log(1+6u_i)+0.15\log(1+8v_i)+0.7r_i\right).
}
\]

生成**当天未来不可观测**的共同冲击

\[
U\sim\mathcal N(0,1).
\]

给定 \((A,X,U)\) 后，用独立随机抽样生成

\[
\boxed{
\begin{aligned}
B_i&\sim\operatorname{Bernoulli}(s(\ell_i+\rho U)),\\
V_i&=0.5\big(1+\operatorname{Binomial}(5,\pi_i)\big),\\
Y_i&=B_iV_i.
\end{aligned}}
\]

因此

\[
Y_i\in\{0,0.5,1,1.5,2,2.5,3\}.
\]

\(B_i\) 是患者是否实际发生用血的二元指示量，不是总需求上界 \(B\)。\(V_i\) 为发生用血后的数量，\(Y_i\) 才是患者实际需求结果。

- S1–S3：\(\rho=0\)。给定完整输入 \((A,X)\) 后，患者结果没有通过 \(U\) 引入额外随机关联；S3 的概率本身仍依赖其他患者的可见特征。
- S4：\(\rho>0\)。给定 \((A,X)\) 后，\(U\) 未被观察到，患者结果与新增需求可存在条件相关性。

### 3.7 新增订单和新增需求

新增订单数量：

\[
\boxed{L\sim\operatorname{Binomial}\big(4,s(-0.7+0.4c_2+0.3c_1+\rho U)\big).}
\]

新增数量概率：

\[
\pi_{\mathrm{new}}=s(-0.4+0.5c_2+0.2\rho U).
\]

新增离散需求：

\[
\boxed{D_{\mathrm{new},0}=0.5\big[L+\operatorname{Binomial}(5L,\pi_{\mathrm{new}})\big].}
\]

其中 \(0\le L\le4\)，所以 \(0\le D_{\mathrm{new},0}\le12\)。再独立产生

\[
J\sim\operatorname{Uniform}(0,0.5),\qquad
D_{\mathrm{new}}=D_{\mathrm{new},0}+J.
\]

\(L\)、\(D_{\mathrm{new},0}\) 和 \(J\) 的实际值均不属于决策前输入。

### 3.8 总需求标签与有界性

\[
\boxed{D=\sum_{i=1}^N Y_i+D_{\mathrm{new},0}+J.}
\]

由于每名可见患者 \(Y_i\le3\)，新增离散需求 \(D_{\mathrm{new},0}\le12\)，连续余量 \(J\le0.5\)，有

\[
0\le D\le3N+12+0.5\le3(M+4)+0.5.
\]

定义

\[
\boxed{B=3(M+4)+0.5.}
\]

所以理论环境 \(B=18.5\)，性能环境 \(B=36.5\)。**不可**将实际连续需求四舍五入、截断尾部或按数据集最大观察值重新定义 \(B\)。

## 4. 四个机制场景：S1–S4

| 场景 | 原配置键 `alpha` | 原配置键 `gamma` | `rho` | 机制含义 |
|---|---:|---:|---:|---|
| S1 | 0 | 0 | 0 | 患者自身特征与基本日级需求机制 |
| S2 | 1 | 0 | 0 | 增加患者 × 可见日级情境交互 |
| S3 | 1 | 1.5 | 0 | 再增加患者构成/同群影响 |
| S4 | 1 | 1.5 | 0.8 | 再增加不可观测共同冲击 |

原配置键 `alpha` 的数学含义是本文 \(\lambda_{\mathrm{ctx}}\)，原配置键 `gamma` 的含义是 \(\lambda_{\mathrm{peer}}\)。**不得**把它们误读为 DAT 理论中的 Hölder \(\alpha\) 和逼近率 \(\gamma\)。

场景 S1 并非“患者信息无价值”，而是没有另外三种增强机制。场景 S3 并非结果之间给定输入后的共同随机冲击；只有 S4 额外包含这种条件依赖。

在同一独立数据重复内，建议先生成共享的 \((A,X,I)\) 池，再分别调用四种机制产生结果，以支持场景内的配对比较。不同场景之间的结果均值和方差也可能改变，不能把每个跨场景性能差全部归因于某一模型模块。

## 5. Oracle 条件分位数：逐项可实现的算法

### 5.1 定义离散主体需求

定义

\[
D_0=\sum_{i=1}^N Y_i+D_{\mathrm{new},0}.
\]

由于 \(D_0\) 仅取半单位网格，令

\[
\boxed{p_k(Z)=\Pr(D_0=0.5k\mid Z),\qquad k=0,1,\ldots,6(N+4).}
\]

这里 **Oracle 只能条件于完整事前信息 \(Z=(A,X)\)**，不能条件于实际结果 \(Y_i,L,J\)，也不能在 S4 中偷看实际抽到的 \(U\)。

### 5.2 每个可见患者的网格 PMF

固定 \(Z,U=u\)，定义

\[
s_i(u)=s(\ell_i+\rho u).
\]

则

\[
\Pr(Y_i=0\mid Z,U=u)=1-s_i(u),
\]

并且对 \(j=1,\ldots,6\)：

\[
\boxed{
\Pr(Y_i=0.5j\mid Z,U=u)
=s_i(u)\binom5{j-1}\pi_i^{j-1}(1-\pi_i)^{6-j}.
}
\]

这个长度为 7 的概率数组就是 \(Y_i\) 的 PMF（以半单位为一个整数索引）。

### 5.3 新增需求 PMF

固定 \(Z,U=u\)，令

\[
a(u)=s(-0.7+0.4c_2+0.3c_1+\rho u),\quad
b(u)=s(-0.4+0.5c_2+0.2\rho u).
\]

对新增订单数 \(\ell=0,\ldots,4\) 和附加 Binomial 取值 \(j=0,\ldots,5\ell\)：

\[
\Pr(L=\ell\mid Z,u)=\binom4\ell a(u)^\ell[1-a(u)]^{4-\ell},
\]

\[
\Pr(W=j\mid L=\ell,Z,u)=\binom{5\ell}j b(u)^j[1-b(u)]^{5\ell-j}.
\]

此时 \(D_{\mathrm{new},0}=0.5(\ell+j)\)。因此新增需求的第 \(k\) 个 PMF 槽为

\[
\boxed{
\Pr(D_{\mathrm{new},0}=0.5k\mid Z,u)
=\sum_{\ell=0}^4\mathbf1_{\{0\le k-\ell\le5\ell\}}
\Pr(L=\ell\mid Z,u)
\Pr(W=k-\ell\mid L=\ell,Z,u).
}
\]

### 5.4 用卷积获得总离散需求分布

记 \(*\) 为以半单位索引进行的离散卷积；给定 \(Z,U=u\)，各结果由分开的随机抽样生成，故

\[
\boxed{
p_{\cdot}(Z,u)
=\mathrm{PMF}_{Y_1}(u)*\cdots*\mathrm{PMF}_{Y_N}(u)
*\mathrm{PMF}_{D_{\mathrm{new},0}}(u).
}
\]

- S1–S3：\(\rho=0\)，该卷积不再依赖 \(u\)，直接输出 \(p_k(Z)\)。
- S4：\(\rho=0.8\)，必须对不可见的正态共同冲击积分：

\[
\boxed{
p_k(Z)=\int_{-\infty}^{\infty}p_k(Z,u)\varphi(u)\,du,
}
\]

其中 \(\varphi\) 是标准正态密度。

可使用预先固定的 Gauss–Hermite 求积或显式尾部截断加数值积分；必须记录节点数、尾部概率、收敛检查和数值稳定性。**有限节点数值计算结果应称为数值 Oracle，而不是误差为零的精确 Oracle**。

### 5.5 加入连续余量 \(J\) 并求条件分位数

定义

\[
C_k(Z)=\sum_{j=0}^kp_j(Z),\qquad C_{-1}=0.
\]

因为 \(J\sim\operatorname{Uniform}(0,0.5)\) 且与 \(D_0\) 独立，在区间 \([0.5k,0.5(k+1)]\) 内：

\[
F_Z(y)=\Pr(D\le y\mid Z)
=C_{k-1}(Z)+2p_k(Z)(y-0.5k).
\]

取最小的 \(k\) 使 \(C_k(Z)\ge\tau\)，在区间内部解 \(F_Z(Q_\tau)=\tau\)，得到

\[
\boxed{
Q_\tau(D\mid Z)
=0.5k+0.5\frac{\tau-C_{k-1}(Z)}{p_k(Z)}.
}
\]

如果数值累计概率非常接近 1，应先处理舍入容差和非负 PMF 归一化。不可对分位数结果做需求单位的事后四舍五入。

### 5.6 条件风险及总体风险

定义 Pinball 损失

\[
\rho_\tau(e)=\tau[e]_++(1-\tau)[-e]_+.
\]

给定 \(Z=z\)，预测 \(y\) 的**条件风险**为

\[
\boxed{
\phi_z(y)=\sum_kp_k(z)\,2\int_0^{0.5}\rho_\tau(0.5k+j-y)\,dj.
}
\]

每项是分段线性函数在有限区间上的积分，可解析计算。总体风险

\[
R_\tau(f)=\mathbb E_Z[\phi_Z(f(Z))]
\]

仍需对 \(Z\) 的分布积分；协议使用独立 \(Z\) 池（起始 50,000 个）数值估计，并报告 Monte Carlo 精度。每次对比必须使用相同的独立评价输入池。

**权限限制**：T2 可以读取完整信息 Oracle 作为拟合标签；T1/T3 的正常 DAT 训练只能读取抽样需求 \(D\)。P3 低信息级别的 Bayes Oracle 不能简单用完整信息 \(Q_\tau(D\mid A,X)\) 代替；当前性能实验只报告 `AC` 与 `SL_G`，所以无须把低信息 Oracle 作为强制第三项评价。

## 6. 理论条件与 DGP 的关系

| 理论条件 | DGP 如何满足 |
|---|---|
| 每天观测 \((Z_t,D_t)\) i.i.d. | 不滚动库存；不同日外生随机量独立 |
| 患者人数 \(1\le N\le M\) | 定义中 \(N=1+\operatorname{Binomial}(M-1,p)\) |
| 患者特征位于 \([0,1]^3\) | \(u_i,v_i,r_i\) 构造 |
| 日级特征位于 \([0,1]^8\) | 支持已知的固定仿射归一化，包含人数比例 |
| 需求 \(D\in[0,B]\) | \(B=3(M+4)+0.5\) |
| 患者排列不变 | 同群核仅依赖数值与患者集合，无行顺序 |
| 条件分位数正则性 | 有限人数、紧输入、平滑严格正概率及连续 \(J\)，支持切片上 Lipschitz 的充分机制 |
| 可选局部二次风险上界 | 条件密度在各区间为 \(2p_k\le2\)，具有统一密度上界 |

这些是该模拟机制支持理论假设的原因；**不表示**实际有限宽 DAT 已被证明达到最坏情形速率，也不表示实际优化器找到全局 ERM。

## 7. C0：T1 专用可实现校准机制（不替换 S1–S4）

为了能真正观测固定类的**估计误差**，T1 另设 C0 校准子环境：

- 固定可见人数 \(N=M\)。
- 用血发生概率固定：`p_issue=0.35`。
- 用血数量二项概率固定：`p_quantity=0.4`。
- 新增订单发生参数固定：`p_new=0.3`（即 \(L\sim\mathrm{Binomial}(4,0.3)\)）。
- 新增需求数量二项概率固定：`p_new_quantity=0.4`。
- 关闭所有日级情境、同群和共同冲击对需求分布的影响。
- 保留独立 \(J\sim U(0,0.5)\)，使需求保持连续版。

此时条件需求分布不依赖 \(A,X\)，故 \(Q_\tau(D\mid Z)=q_\tau^\star\) 为一个常数。使用仅含单个输出偏置的常数预测子类，\(f^\star\) 可实现，\(\mathcal E^{\mathrm{app}}=0\)；训练样本分位数是真正的精确 ERM。

**C0 是 T1 校准环境，不是第五个性能场景。** 不能把其验证结果解释为完整 5,801 参数 DAT 已获精确 ERM 认证。

## 8. 数据记录结构与可见性

建议每个模拟日保留下列**生成器内部**信息：

```text
sample_id, scenario_id, M, N,
c1, c2, inventory_raw,
patient_features: [N, 3],
day_features: [8],
demand_total: scalar,
patient_outcomes: [N],     # 仅准许授权的 Flat-ETO 训练使用
latent_U, L, J, D_new,     # 只能用于生成器内部核查，禁止预测器读取
rng_metadata, split_id
```

数据权限分层：

| 参与者 | 允许读取 |
|---|---|
| DAT / DeepQR 等正常分位数预测器 | 当前信息等级授权的 \(A,X\)、训练标签 \(D\) |
| Flat-ETO | 同样的事前输入 + 方案明示允许的训练集订单结果 \(Y_i\) |
| T2 Oracle 拟合器 | 当前完整 \(A,X\) 及由 DGP 计算的 \(Q_\tau(D\mid A,X)\) |
| Oracle 评价器 | 已知 DGP 参数，给定 \(Z\) 后积分/卷积；不能使用真实日内 \(U\) |
| 生成器内部 | 所有外生及实现随机量，用于复现/检查 |

**禁止**把 `latent_U`、`J`、`patient_outcomes` 或实际 `D_new` 无意中加入训练特征矩阵；也不能由测试结果反推并向模型开放这些变量。

## 9. 必做的生成器和 Oracle 验收

1. **支持与一致性**：\(1\le N\le M\)，\(a_i\in[0,1]^3\)，\(X\in[0,1]^8\)，\(D\in[0,B]\)。
2. **库存恢复**：`I == 225 + 180 * X[2]`（在浮点容差内）。
3. **人数恢复**：完整信息 `X[3] == N/M`；Padding Mask 的真实人数一致。
4. **条件 PMF**：每个 \(p_k\ge0\)，\(\sum_kp_k=1\)（求积和浮点容差内）。
5. **Oracle 公式**：\(0\le Q_\tau\le B\)，\(|F_Z(Q_\tau)-\tau|\) 在预设数值容差内。
6. **S4 积分稳定性**：节点数或截断区域加密后，分位数与条件风险差异可忽略到预先规定的数值精度。
7. **重排对称性**：患者排列改变不得改变 \(p_k\)、Oracle、总需求的**条件分布**。
8. **未来信息隔离**：正常预测器看不到 \(U,L,Y_i,J,D_{\mathrm{new}}\) 的实际值。
9. **场景配对**：同一重复内 S1–S4 的可见 \(A,X,I\) 完全一致，需求机制使用对应参数。
10. **目标尺度不变**：\(I\) 分布不随 \(G\) 变化，改变 \(G\) 不重新生成 \(A,X,D\)。
11. **连续版本**：`J > 0` 几乎处处，不能在训练数据中把 \(D\) 四舍五入为半单位网格。
12. **独立种子**：区分数据生成、模型训练和 S4 数值 Oracle 求积用的随机种子。

由于原库存/目标尺度较大，可能存在 \(I-G\ge B\) 的日期，此时对全部 \([0,B]\) 内预测器都不会补货、目标服务也自动达成。这是**保留尺度的既定运营情境**，不是数据生成错误；不能在看过模型结果后删除这些日期。

## 10. 最简生成与 Oracle 伪代码

```text
function generate_one_day(M, scenario, rng):
    c1, c2 ~ independent Uniform(-1,1)
    eI ~ Uniform(-70,70)
    I = 315 + 20*c1 + eI

    N = 1 + Binomial(M-1, sigmoid(-0.2 + 0.4*c1))
    for i in 1..N:
        r[i] ~ Beta(2,3)
        u[i] ~ Binomial(12, 0.25+0.25*r[i])/12
        v[i] ~ Binomial(16, 0.15+0.20*r[i])/16
    A = multiset(zip(u,v,r))
    X = build_day_features(c1,c2,I,A,N,M)  # precisely 8 dims

    peer = masked_gaussian_peer(u,r)         # N=1 -> 0
    ell, pi = patient_probability_parameters(A,c1,c2,peer,scenario)
    U ~ Normal(0,1)
    for i in 1..N:
        B_i ~ Bernoulli(sigmoid(ell_i + rho*U))
        V_i = 0.5*(1 + Binomial(5,pi_i))
        Y_i = B_i*V_i
    L ~ Binomial(4, sigmoid(-0.7+0.4*c2+0.3*c1+rho*U))
    pi_new = sigmoid(-0.4+0.5*c2+0.2*rho*U)
    D_new0 = 0.5*(L + Binomial(5*L,pi_new))
    J ~ Uniform(0,0.5)
    D = sum(Y) + D_new0 + J
    return authorized_inputs(A,X,I), label(D), internal_audit(U,Y,L,J)

function oracle_quantile(Z, tau, scenario):
    if scenario.rho == 0:
        p = convolution_pmf(Z, U_unused)
    else:
        p = gaussian_integral_over_U(convolution_pmf(Z,U))
    p = validated_nonnegative_normalized_PMF(p)
    C = cumulative_sum(p)
    k = smallest_index(C[k] >= tau)
    return 0.5*k + 0.5*(tau - C[k-1])/p[k]
```

---

**冻结结论**：原始 AXID 随机生成公式、原库存尺度、\(B\)、\(G\)、四个场景和 Oracle 原理均不变。以后如需新增高维、离散版或其他分布族，必须另建有名称的独立实验协议，不能悄悄覆盖本 DGP。
