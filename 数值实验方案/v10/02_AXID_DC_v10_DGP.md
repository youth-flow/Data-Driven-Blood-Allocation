# AXID-DC v10｜单一 G 的 MDCEV 合成数据生成机制

> 场景：`AXID-DC-v10`；`schema_version=generator_version=10`。正式实现见
> `synthetic/src/axid_synthetic/generator_v10.py`。本 DGP 不读取测试结果，也不得依据
> DAT 或基准方法的排名回调参数。

## 1. 时序、信息和共同目标

血液中心观察

\[
Z_t=(A_t^-,c_{p,t},c_{e,t},I_t)
\]

后预测当天总发血量的条件 0.9 分位数。随后新增患者集合 (A_t^+) 到达，医院医生在
完整患者集合上实施联合分配。本轮补货 (q_t) 不进入医生机制，因此生成的 (D_t) 不依赖
被评价方法的决策。

医院效用锚点和血液中心目标共同固定为

\[
G=256,qquad \kappa=1,qquad 0\le Y_i\le10.
\]

## 2. 日级状态和人数

原始人数满足

\[
\widetilde N^-\sim NB(6.92,0.106),\qquad
\widetilde N^+\sim NB(4.91,0.146),
\]

其中 (E[N]=r(1-p)/p)。可执行支持为

\[
N^-=\min\{138,\max(1,\widetilde N^-)\},\qquad
N^+=\min\{76,\widetilde N^+\}.
\]

\[
c_p\sim Beta(6.7,1.4),\qquad c_e\sim Beta(2.85,1.81),
\]

\[
I\sim0.39TN_{[40,519]}(231,57^2)+0.61TN_{[40,519]}(301,44^2).
\]

按顺序 ((N^-,N^+,I,c_p,c_e)) 使用 Gaussian copula：

\[
\Sigma=
\begin{pmatrix}
1&.206&-.229&.302&.217\\
.206&1&-.132&.004&.139\\
-.229&-.132&1&.129&-.305\\
.302&.004&.129&1&.564\\
.217&.139&-.305&.564&1
\end{pmatrix}.
\]

## 3. 患者特征

申请量 (u\in\{0,1,2,3,4,5,6,8,10\})，概率为

\[
(0.165,0.005,0.381,0.030,0.388,0.002,0.017,0.003,0.009).
\]

低、中、高风险层的风险分布分别为

\[
Beta(2,8),\quad Beta(5,5),\quad Beta(8,2).
\]

(A^-) 的层权重为 ((.73,.07,.20))，(A^+) 为 ((.35,.10,.55))。Hb 单位为
g/L，均截断在 ([42,170])：

\[
Hb\mid L\sim.23TN(103,24^2)+.77TN(138,15^2),
\]
\[
Hb\mid M\sim.75TN(61,9^2)+.25TN(111,25^2),
\]
\[
Hb\mid H\sim.82TN(59,9^2)+.18TN(108,27^2).
\]

## 4. \(\gamma\)、\(\phi\) 与联合分配

定义

\[
\ell_i=\frac{\log(1+u_i)}{\log 11},\qquad
d_i=\min\left\{4,\left[\frac{70+10r_i-Hb_i}{10}\right]_+\right\}.
\]

\[
\gamma_i=0.7(0.5+1.5\ell_i),
\]

\[
\log\phi_{it}=-1.78+0.15\ell_i+1.20r_i+0.45d_i
+0.40c_{p,t}\ell_i+0.60c_{e,t}r_i+\eta_t+\varepsilon_{it},
\]

其中

\[
\eta_t\sim N(0,.08^2),\qquad
\varepsilon_{it}\sim Gumbel(-.12\gamma_E,.12).
\]

医生求解

\[
\max_{R,\mathbf Y}
256\log(R/256)+\sum_i\gamma_i\phi_i\log(1+Y_i/\gamma_i)
\]

满足 (R+\sum_iY_i=I)、(R>0)、(0\le Y_i\le10)。唯一解为

\[
Y_i=\min\left\{10,\gamma_i(\phi_i/\lambda-1)_+\right\},
\]

其中

\[
256/\lambda+\sum_iY_i=I.
\]

患者之间通过共同影子价格 (\lambda) 竞争；(c_p\ell_i,c_er_i) 是显式日状态—患者
交互，库存 (I) 通过共同预算改变零/正发血和正剂量。最终 (D=\sum_iY_i\in[0,519])。

## 5. 输入合同

患者 token 为

\[
(u/10,\ r,\ (Hb-42)/128)\in[0,1]^3.
\]

日输入仍为 7 维：

\[
X=\left(c_p,c_e,\frac{I-40}{479},\frac{N^-}{138},
\frac{\sum u}{1380},\frac{\sum r}{138},
\frac{\sum(170-Hb)/128}{138}\right).
\]

正式数据采用连续患者数组加 day offsets 的 CSR/memmap 格式，不生成三千万行 CSV。

## 6. 参数识别与校准口径

本版采用**结构预设 + 经验分布估计 + 透明矩校准**，不是“全部参数由真实数据完全反演”。

目标函数存在尺度不识别：同时把 (\kappa) 和所有 (\phi_i) 乘以同一个正数不会改变
最优分配。因此 (G=256) 是共同业务锚点，(\kappa=1) 是效用尺度归一化，不能称为
数据估计结果。

若真实数据完整观测同日 choice set、逐患者 (Y_i)、准确资源预算 (I) 和保留量
(R=I-\sum_iY_i)，可通过角点/封顶 MDCEV 似然或模拟矩估计 (\gamma) 形状和
\(\log\phi=x^\top\beta+\eta+\varepsilon\) 的相对斜率。但当前 processed 数据存在：

- 没有原生连续风险 (r)，必须由事前临床字段预先构造；
- 新增患者级 (A^+) 不完整；
- (R) 和影子价格未直接观测；
- 期初库存、同日入库和其他离库不完全等于本 DGP 的硬预算；
- `real_demand=0` 表示当天未发，不必然是最终拒绝；同一 pending request 可跨日重复出现。

因此 v10 的六变量、人数和相关结构以真实数据聚合规律为主要依据；函数形式、交互项和
符号先由业务与理论冻结；可识别的相对斜率可作辅助估计；截距、\(\gamma) 基准尺度及
随机效用尺度用预先声明的矩校准和敏感性情景处理。所有校准必须在性能模型训练前冻结，
严禁根据 DAT 是否胜出反调 DGP。
