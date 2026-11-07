# 02｜Synthetic DGP：S4 完整需求机制与 Oracle 计算规范

> **状态**：2026-11-05 正式生成协议。  
> **主机制**：**仅 S4**，即 `lambda_ctx=1`、`lambda_peer=1.5`、`rho=0.8`。不运行 S1、S2、S3 的主实验，不做跨场景优劣比较。  
> **唯一例外**：理论 T1-C0 使用一个**独立的常数需求分位数校准环境**，目的是精确 ERM 与高概率界校准，**不是另一性能 DGP**。  
> **主任务**：`tau=0.9`；其他分位数仅作预先登记的辅助分析。保留原库存规模、需求规模及 `G` 网格。  
> **约束**：只生成合成数据；不读取任何医院真实表、历史收货策略或真实训练权重。

---

## 1. 基本框架与两种规模

一天独立生成一组 

\[
(Z,D,I),\qquad Z=(A,X),\quad A=\{(u_i,v_i,r_i)\}_{i=1}^{N}.
\]

- `A`、`X`、`I`：决策前可见信息；`I` 已经内含于 `X`，原值另存只为运营计算。
- `D`：决策后实现的当日总需求，训练时作为日级标签。
- `Y_i`：逐患者事后实际需求，**仅 Flat-ETO 的训练组件可作为额外标签读取**。
- `U,L,J,D_new`：事后或潜在随机变量，只有生成器/Oracle 可以访问，预测器不可见。
- 不同日的全部外生随机量相互独立，所以日级 `(Z_t,D_t)` 为 i.i.d.；日期只作为 ID，不引入星期周期、滚动库存或在线历史需求更新。

| 设置 | 理论 T1–T3 | 性能 P1–P3 |
|---|---:|---:|
| 最大可见患者数 `M` | 2 | 8 |
| 患者特征维度 `d_a` | 3 | 3 |
| 日级其余维度 `d_x` | 7 | 7 |
| 完整日级输入维度 | 8 | 8 |
| 最多新增订单 `K_new` | 4 | 4 |
| 需求上界 `B` | 18.5 | 36.5 |
| DAT 模型潜维（非生成参数） | 20 | 64 |

**重要**：在理论和性能规模之间，只有 `M` 及由其决定的需求上界、模型尺寸等不同；S4 概率公式及其系数相同。给定一条规模曲线时不得同时改变 `M`。

## 2. S4 的逐日生成顺序

定义 Sigmoid：

\[
\sigma(x)=\frac1{1+e^{-x}}.
\]

### 2.1 外生日级情境与期初库存

独立抽样：

\[
c_1,c_2\overset{i.i.d.}{\sim}\mathrm{Uniform}(-1,1),\qquad
\varepsilon_I\sim\mathrm{Uniform}(-70,70).
\]

定义：

\[
\boxed{I=315+20c_1+\varepsilon_I,\qquad I\in[225,405].}
\]

- `c1` 解释为计划业务强度，`c2` 解释为急诊压力。
- `I` 是决策前已知的库存；**不**由前一天库存递推。
- 给定 `c1,c2`，独立库存扰动不额外影响需求分布，但库存影响后续补充策略。
- 本协议保留这个原始数量尺度，**不**改为此前讨论的低库存版本。

### 2.2 决策前可见患者人数

\[
\boxed{N=1+\mathrm{Binomial}(M-1,\sigma(-0.2+0.4c_1)).}
\]

因此 `1<=N<=M`，理论所需的非空有限多重集合条件成立。

### 2.3 每名患者的三个事前坐标

对 `i=1,...,N`，独立抽样基础随机量，先取：

\[
 r_i\sim\mathrm{Beta}(2,3).
\]

条件于 `r_i`，分别取：

\[
\boxed{
\begin{aligned}
u_i&=\mathrm{Binomial}(12,0.25+0.25r_i)/12,\\
v_i&=\mathrm{Binomial}(16,0.15+0.20r_i)/16.
\end{aligned}}
\]

患者输入：

\[
\boxed{a_i=(u_i,v_i,r_i)\in[0,1]^3.}
\]

`6u_i` 和 `8v_i` 分别是合成协议的物理申请量和备血量，`r_i` 是事前临床风险评分。没有强制 `8v_i>=6u_i`，也没有要求实际 `Y_i` 必然不超过申请量；不要擅自通过数据清洗添加这些限制。患者行顺序不具有业务含义。

### 2.4 八维日级输入（固定次序）

\[
\boxed{
X=\left(
\frac{c_1+1}{2},\frac{c_2+1}{2},\frac{I-225}{180},
\frac NM,\frac{\sum_i u_i}{M},\frac{\sum_i v_i}{M},
\frac1N\sum_i r_i,\frac1N\sum_i r_i^2
\right)\in[0,1]^8.
}
\]

| 0-based | 键 | 含义 |
|---:|---|---|
| 0 | `c1_scaled` | `(c1+1)/2` |
| 1 | `c2_scaled` | `(c2+1)/2` |
| 2 | `inventory_scaled` | `(I-225)/180` |
| 3 | `count_ratio` | `N/M` |
| 4 | `sum_u_scaled` | `sum(u_i)/M` |
| 5 | `sum_v_scaled` | `sum(v_i)/M` |
| 6 | `risk_mean` | `mean(r_i)` |
| 7 | `risk_second_moment` | `mean(r_i**2)` |

`d_x=7` **不包含**第 4 个数学坐标 `N/M`，而模型收到的完整 `X` 总是 8 维。所有特征是已知支持上的固定变换，不能根据训练样本均值/方差重新定义理论输入域。原单位库存从 `I=225+180*X[2]` 恢复。

### 2.5 患者同群风险

当 `N=1` 时定义 `peer_1=0`。当 `N>1` 时：

\[
 w_{ij}=\exp\!\left[-\frac{(u_i-u_j)^2}{2(0.25)^2}\right],\quad j\ne i,
\]

\[
\boxed{\mathrm{peer}_i=\frac{\sum_{j\ne i}w_{ij}r_j}{\sum_{j\ne i}w_{ij}}.}
\]

由于 `u_i∈[0,1]` 且核权重为正，`N>1` 时分母严格为正。该量由事前患者特征确定，不是从其他患者的事后实际需求中计算出来的。

### 2.6 发生用血的 Logit 与用血数量参数

S4 的固定机制参数为：

\[
\boxed{\lambda_{\rm ctx}=1,\qquad \lambda_{\rm peer}=1.5,\qquad\rho=0.8.}
\]

这里的 `lambda_ctx`、`lambda_peer` 对应最初 MD 的 DGP 系数 `alpha`、`gamma`，**不是**逼近定理的 Hölder 指数与速率指数。

对患者 `i`：

\[
\boxed{
\begin{aligned}
\ell_i={}&-2+0.35\log(1+6u_i)+0.15\log(1+8v_i)+1.2r_i\\
&+\lambda_{\rm ctx}(0.8c_1u_i+0.6c_2r_i)
+\lambda_{\rm peer}r_i\,\mathrm{peer}_i,\\
\pi_i={}&\sigma[-0.5+0.3\log(1+6u_i)+0.15\log(1+8v_i)+0.7r_i].
\end{aligned}}
\]

再独立生成同一天共享、但在决策时不可见的：

\[
 U\sim\mathcal N(0,1).
\]

给定 `A,X,U` 后，通过相互独立的抽样生成：

\[
\boxed{
\begin{aligned}
B_i&\sim\mathrm{Bernoulli}(\sigma(\ell_i+\rho U)),\\
V_i&=\tfrac12[1+\mathrm{Binomial}(5,\pi_i)],\\
Y_i&=B_iV_i.
\end{aligned}}
\]

于是：

\[
\boxed{Y_i\in\{0,0.5,1,1.5,2,2.5,3\}.}
\]

解释：`B_i` 是是否发生用血，`V_i` 是发生用血后的数量，`Y_i` 是实现的患者需求。**S4 给定完整事前信息后，各患者结果一般并不独立**，因为未观测 `U` 仍是共同随机来源；只有再条件于 `U` 才可以直接使用独立卷积。

### 2.7 决策后新增需求

\[
\boxed{L\sim\mathrm{Binomial}(4,\sigma[-0.7+0.4c_2+0.3c_1+\rho U]).}
\]

\[
\pi_{\rm new}=\sigma[-0.4+0.5c_2+0.2\rho U].
\]

\[
\boxed{D_{\rm new,0}=\tfrac12[L+\mathrm{Binomial}(5L,\pi_{\rm new})].}
\]

生成独立连续余量：

\[
J\sim\mathrm{Uniform}(0,0.5),\qquad D_{\rm new}=D_{\rm new,0}+J.
\]

`L`、`D_new,0`、`J` 的实际值在决策时都未知；`L<=4`，故 `D_new,0<=12`。

### 2.8 总需求与硬上界

\[
\boxed{D=\sum_{i=1}^NY_i+D_{\rm new,0}+J.}
\]

由于每名可见患者最多用 3 单位、新增订单最多 4 名且每名最多 3 单位、`J<=0.5`：

\[
0\le D\le3N+12+0.5\le3(M+4)+0.5.
\]

因此：

\[
\boxed{B=3(M+4)+0.5}
\]

对应理论环境 `B=18.5`、性能环境 `B=36.5`。总需求**保留连续值**，不按半单位四舍五入，不用样本最大值重设 `B`。

## 3. 为什么主实验只用 S4？

S4 **同时**包含：

1. 患者特征与可见日级情境的交互（`lambda_ctx=1`）；
2. 患者群体内部的相似度加权风险作用（`lambda_peer=1.5`）；
3. 真实需求中的未观测共同冲击（`rho=0.8`）。

所以 T1-DAT/T2/T3 和 P1/P2/P3 均在**同一个完整 S4 DGP** 下运行，避免跨机制训练组合的额外计算。仍允许改变理论/性能人数上界 `M`，但不能在同一实验曲线里改变它。

只用 S4 时，**不能**声称通过跨场景比较分别识别三种机制的独立贡献；P2 识别的是在固定 S4 环境下移除特定架构模块的结果。

## 4. 完整信息 Oracle：从 PMF 到真实条件分位数

Oracle 是：

\[
\boxed{f_\tau^\star(Z)=Q_\tau(D\mid A,X),}
\]

**不能条件于当天实现的 `U`**，因为共同冲击在决策前不可见。它不是一套网络 Teacher，更不使用训练集反向拟合 DGP 参数。

### 4.1 离散主体需求

\[
D_0=\sum_iY_i+D_{\rm new,0},\qquad D=D_0+J.
\]

令：

\[
p_k(Z)=\Pr(D_0=0.5k\mid Z),\qquad k=0,1,\ldots,6(N+4).
\]

固定 `Z,U=u`，定义 `s_i(u)=sigma(ell_i+rho*u)`。

每名可见患者的离散 PMF：

\[
\boxed{
\begin{aligned}
\Pr(Y_i=0\mid Z,u)&=1-s_i(u),\\
\Pr(Y_i=0.5j\mid Z,u)&=s_i(u)\binom5{j-1}\pi_i^{j-1}(1-\pi_i)^{6-j},\quad j=1,\ldots,6.
\end{aligned}}
\]

### 4.2 新增需求的离散 PMF

令：

\[
a(u)=\sigma[-0.7+0.4c_2+0.3c_1+\rho u],\quad
b(u)=\sigma[-0.4+0.5c_2+0.2\rho u].
\]

给定 `u`：

\[
\Pr(L=\ell\mid Z,u)=\binom4\ell a(u)^\ell[1-a(u)]^{4-\ell},\quad 0\le\ell\le4.
\]

给定 `L=ell`：

\[
\Pr(W=j\mid L=\ell,Z,u)=\binom{5\ell}j b(u)^j[1-b(u)]^{5\ell-j},\quad0\le j\le5\ell.
\]

因为 `D_new,0=0.5(L+W)`，新增需求 PMF 的第 `k` 个元素为：

\[
\boxed{
\Pr(D_{\rm new,0}=0.5k\mid Z,u)
=\sum_{\ell=0}^4
\mathbf1\{0\le k-\ell\le5\ell\}
\Pr(L=\ell\mid Z,u)\Pr(W=k-\ell\mid L=\ell,Z,u).
}
\]

### 4.3 条件卷积，随后积分掉共同冲击

**先在给定 `U=u` 的条件下**卷积：

\[
\boxed{
 p_{\cdot}(Z,u)
 =p_{Y_1}(\cdot\mid Z,u)*\cdots*p_{Y_N}(\cdot\mid Z,u)
 *p_{\rm new,0}(\cdot\mid Z,u).
}
\]

`*` 是以 `0.5` 为一步的离散 PMF 卷积。由于 S4 的 `U` 未被观察，必须再计算：

\[
\boxed{p_k(Z)=\int_{-\infty}^{\infty}p_k(Z,u)\varphi(u)\,du,\qquad U\sim N(0,1).}
\]

推荐数值实现采用 Gauss–Hermite 求积，节点 `x_j` 和权重 `w_j` 选自标准 `e^{-x^2}` 权函数时：

\[
p_k(Z)\approx\frac1{\sqrt\pi}\sum_{j=1}^{n_q}w_j\,p_k(Z,\sqrt2 x_j).
\]

积分阶数及误差容差由 Oracle 数值规范在正式运行前固定；比较加密节点后的 `p_k`、分位数和条件风险，报告数值收敛。**有限精度的结果称数值 Oracle，而不宣称无误差解析 Oracle**。

> **禁止的错误顺序**：先对每个患者边际 PMF 各自积分掉 `U`，再把边际当成独立项卷积。这样会丢掉 S4 的共同冲击相关结构。

### 4.4 连续余量产生线性 CDF

令累计网格概率：

\[
C_k(Z)=\sum_{j=0}^kp_j(Z),\qquad C_{-1}=0.
\]

在 `y∈[0.5k,0.5(k+1)]` 上：

\[
F_{D\mid Z}(y)=C_{k-1}(Z)+2p_k(Z)(y-0.5k).
\]

找最小 `k`，使 `C_k(Z)>=tau`。在 `p_k(Z)>0` 的条件下，分位数为：

\[
\boxed{Q_\tau(D\mid Z)=0.5k+0.5\frac{\tau-C_{k-1}(Z)}{p_k(Z)}.}
\]

**主任务取 `tau=0.9`**。辅助 `tau` 如被预先登记，可复用同一个条件 PMF 计算其分位数，不需重新生成 DGP。返回值在 `[0,B]` 内，不能再按半单位取整。

### 4.5 用于理论评价的条件 Pinball 风险

对预测值 `y`，定义：

\[
\rho_\tau(e)=\tau[e]_++(1-\tau)[-e]_+,\quad
\phi_Z(y)=\mathbb E[\rho_\tau(D-y)\mid Z].
\]

由 `D=D_0+J` 得到：

\[
\boxed{
\phi_Z(y)=\sum_kp_k(Z)\,2\int_0^{0.5}
\rho_\tau(0.5k+j-y)\,dj.
}
\]

内层可分段解析积分；总体风险 `R_tau(f)=E_Z[phi_Z(f(Z))]` 仍需以独立 `Z` 评价池作 Monte Carlo 近似。**性能 P1–P3 不将 Oracle 风险作为第三个最终性能指标。**

## 5. T1-C0：唯一允许的专用理论校准环境

为了直接计算精确 ERM 的估计误差，单独使用 `C0`：

- 令 `N=M`，但患者特征、日级情境仍可按既有支持生成。
- 每名可见患者的用血发生概率固定为 `p_issue=0.35`。
- 发生用血后的数量分布使用固定二项概率 `p_quantity=0.4`。
- 新增订单数服从 `Binomial(4,p_new)`，其中 `p_new=0.3`。
- 新增数量二项概率 `p_new_quantity=0.4`。
- 需求分布不依赖 `A,X`，也不受同群或未观测共同冲击影响。
- 保留 `J~Uniform(0,0.5)`。

于是：

\[
Q_\tau(D\mid Z)=q_\tau^\star
\]

是常数。只训练 `[0,B]` 中的一个常数预测值时，样本分位数为精确 ERM，类内逼近误差为 0。**C0 仅用于 T1a 高概率上界校准，不得混入 S4 的 T1-DAT、T2、T3 或 P1–P3 主样本。**

## 6. 输出结构与数据权限

推荐生成器内部记录：

```text
sample_id, split_id, replicate_id, M, scenario_id='S4',
c1, c2, inventory_raw, N,
patient_features[N,3], day_features[8],
demand_total,
patient_outcomes[N],          # 仅 Flat-ETO 授权训练标签
latent_U, L, D_new0, J,      # 仅生成器/Oracle 审计
rng_metadata
```

| 使用者 | 预测时可见信息 | 训练标签特权 |
|---|---|---|
| DAT、DeepQR、Set-ETO | `A,X`（P3 按授权遮蔽） | 日级 `D` |
| wSAA、LQR、GBQR | 由 `A,X` 构造的固定 19 维摘要 | 日级 `D` |
| SAA、Rolling-SAA | 历史训练需求（日期顺序仅供固定窗口） | 日级 `D` |
| Flat-ETO | 22 维患者输入、19 维日级摘要 | 日级 `D` 和训练期逐患者 `Y_i` |
| Oracle/生成器 | 完整 DGP 参数与已观测 `Z`；内部可审计潜变量 | 不把 `latent_U` 作为决策时信息 |

未来信息 `U,L,J,Y_i,D_new` 不作为任何模型的预测特征；Flat-ETO 仅可在**训练阶段**读取自己授权的 `Y_i` 监督，不可在测试日提前读取。

## 7. 与理论条件的对应

- 每日观测 i.i.d.、人数 `1<=N<=M`、患者特征与日级特征有界、需求 `D<=B`。
- 患者集合无序；`peer_i` 由数值关系而非行位置生成，故目标条件分布置换不变。
- 固定 `N` 后，离散 PMF 的所有网格点具有严格正的条件概率；相应概率函数在紧输入域上连续并有界。加入连续 `J` 后，条件 CDF 在支持上严格递增，条件分位数可在各人数切片上满足所需的 Lipschitz/Hölder 条件。
- 条件密度在每个网格单元内为 `2p_k(Z)<=2`，可用于检验局部二次风险的充分条件。
- S4 的同一天多个患者需求条件相关，但**不同天完整观测仍独立同分布**；这不违反理论的日级 i.i.d. 假设。

上述条件是定理的适用依据，不等于实际优化算法已达到精确 ERM。

## 8. 必须通过的 DGP／Oracle 验收

1. `scenario_id == 'S4'`，除 T1-C0 专用生成器之外没有 S1–S3 正式分支。
2. 各规模下 `M,N,d_a,d_x,B` 与硬支持一致；`X` 恰好八维，`X[3]=N/M`。
3. `I=225+180*X[2]`，库存生成分布不随 `G`、`tau` 或模型改变。
4. 患者行任意重排，S4 的**条件 PMF**与 Oracle 分位数不变。
5. `p_k(Z)>=0`，且 `sum_k p_k(Z)=1`（预设的浮点容差内）。
6. `F_{D|Z}(Q_tau(Z))≈tau` 且 `0<=Q_tau<=B`。
7. 逐步加密正态积分节点，Oracle 分位数和条件风险在预设数值容差内稳定。
8. 正常模型不能读取事后 `U,L,J,D_new,Y_i` 的预测时值；只有 Flat-ETO 的训练级监督例外。
9. 所有主训练标签是实际连续 `D`，不四舍五入；T2 才可使用 Oracle 分位数作为特殊训练目标。
10. 生成器、模型训练、数值积分及评价随机性分别管理，配置指纹随结果保存。
11. 不按照测试表现改变需求系数或挑选容易使 DAT 获胜的库存/目标尺度。

## 9. 生成／Oracle 伪代码

```python
def generate_day(M, rng):
    c1, c2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
    I = 315 + 20*c1 + rng.uniform(-70, 70)
    N = 1 + rng.binomial(M-1, sigmoid(-0.2+0.4*c1))
    A = [sample_patient(rng) for _ in range(N)]  # (u,v,r) as §2.3
    X = build_8_day_features(c1,c2,I,A,N,M)
    peer = compute_peer_without_order(A)
    ell, pi = patient_parameters(A, c1,c2,peer, 1, 1.5)
    U = rng.normal(0,1)  # latent; never sent to predictors
    Y = [rng.binomial(1, sigmoid(e+0.8*U))
         * 0.5*(1+rng.binomial(5,p)) for e,p in zip(ell,pi)]
    L = rng.binomial(4, sigmoid(-0.7+0.4*c2+0.3*c1+0.8*U))
    pn = sigmoid(-0.4+0.5*c2+0.2*0.8*U)
    Dnew0 = 0.5*(L+rng.binomial(5*L,pn))
    J = rng.uniform(0,0.5)
    D = sum(Y)+Dnew0+J
    return (A,X,I,D), {'Y_train_only':Y, 'U_private':U, 'L_private':L, 'J_private':J}

def full_information_oracle(Z, tau=0.9):
    # For every quadrature node u:
    #   compute all conditional patient PMFs and new-order PMF,
    #   convolve them to get p[k|Z,u].
    # Integrate each p[k|Z,u] against N(0,1) density.
    pmf = integrate_joint_convolution_over_normal_U(Z)
    pmf = check_and_normalize_numeric_pmf(pmf)
    cumulative = pmf.cumsum()
    k = first_index(cumulative >= tau)
    prev = 0 if k == 0 else cumulative[k-1]
    return 0.5*k + 0.5*(tau-prev)/pmf[k]
```

---

**结论**：本协议冻结的是 **原始 AXID 生成公式 + 原库存与 `G` 数量尺度 + 仅 S4 + 单一主 `tau=0.9`**。T1-C0 是特殊校准例外；所有性能对比不使用真实医院数据。
