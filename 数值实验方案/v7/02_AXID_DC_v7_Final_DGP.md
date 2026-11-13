# 02｜AXID-DC v7-Final：连续患者画像、优先级竞争与动态匹配的需求生成机制

> **版本识别：`axid_dc_v7_continuous_final`。** 本文按要求沿用“v7 最终修订版”称呼，但**不是**已于此前完成正式实验的 `AXID-DC v7`（8 类离散画像）原始 DGP。这里将患者 `u,r` 改为连续、增加今日/过去的医生优先级差异，并替换日级关注公式；属于**新生成机制**。原 v7 的模型结果、Oracle 和 Prescriptiveness **一律不得与本版混合或移植**。
>
> 研究性质：**两级供应链的受控合成行为模型**，不是对真实输血科医生偏好的临床估计。正式 P1/P2/P3 排名未知；冻结此方案后使用全新数据检验。

## 1. 研究对象、信息边界与决策顺序

- **医院输血科（需求生成端）**：发血时观察既有患者 `A⁻`、今日新增患者 `A⁺`、日级业务状态 `c=(c_p,c_e)` 及共同可见的事前库存信号 `I`；依据相同的随机效用机制为每名患者决定拟发血量 `Y_i`，汇总总需求 `D`。**医院不知道本次血液中心实际补货 `q`**。
- **血液中心（库存补充端）**：订货时只观察 **过去已登记患者** `A⁻`、当天业务状态、库存 `I` 和由 `A⁻` 构造的日级向量 `X`；不知道今日患者 `A⁺`、共同冲击 `U`、个体发血结果 `Y_i` 或需求 `D`；决定 `q` 使期末库存贴近目标 `G`。
- **时间顺序**：生成 `(c,I,A⁻)` → 中心根据 `Z=(A⁻,X)` 决定 `q` → 今日患者 `A⁺` 出现 → 医院观察全部患者后形成随机需求 `D` → 以同一 `D` 比较各策略。
- **需求解释**：`D` 为医院的**拟发血/申请需求**，不是接受本次 `q` 硬约束后的实际满足量。医生的 `I` 是事前区域库存/安全储备压力信号，不施加 `Σ_i Y_i ≤ I` 硬约束。`G,q` 不进入需求生成式。

记 `σ(z)=1/(1+exp(-z))`，`[z]₊=max(z,0)`。每个决策日独立同分布；同日存在共享的隐变量 `U`。

中心的统一策略为

\[
\boxed{q_{f,G}(Z)=[f(Z)+G-I]_+,\qquad Z=(A^-,X),\qquad I_{\mathrm{end}}=I+q_{f,G}-D.}
\]

主分位数 `τ=0.9`；固定 `G∈{260,280,300,320,340}`；需求预测上界 `B=61`。

## 2. 阶段 A｜运营环境与过去、今日两类患者

### A1. 日级状态和库存

\[
\boxed{c_p,c_e\overset{\mathrm{iid}}\sim\mathrm{Unif}(-1,1),\qquad
\varepsilon_I\sim\mathrm{Unif}(-50,50),\qquad
I=270+15c_p+\varepsilon_I\in[205,335].}
\]

- `c_p`：计划手术压力；`c_e`：急诊压力；`I`：事前共享的库存/安全储备状态。
- 本模型**不**建立跨日期滚动队列或补货对未来到达的反馈，保留 iid 训练/测试假设。

### A2. 两类患者人数

\[
\boxed{
N^-=3+\mathrm{Binom}\!\left(5,\sigma(-0.1+0.4c_p)\right),\quad
N^+=\mathrm{Binom}\!\left(2,\sigma(-0.4+0.5c_e)\right).}
\]

其中 `3≤N⁻≤8`、`0≤N⁺≤2`、全日 `3≤N=N⁻+N⁺≤10`。`A⁻` 全部是**过去**患者且对中心可见；`A⁺` 全部是**今天**患者且在中心订货时不可见。不存在“今日早到可见”的第三类患者。

### A3. 共享患者画像生成规律

令医院端患者画像为

\[
\boxed{a_i^{\mathrm{hosp}}=(u_i,r_i,s_i,b_i),\qquad
b_i=\begin{cases}0,&i\in A^-,\\1,&i\in A^+;\end{cases}}
\]

**两批患者使用完全相同的**临床画像抽样律；给定 `c`，不同患者独立生成：

\[
\boxed{
\begin{aligned}
u_i&\sim\mathrm{Beta}(2,2),\\
r_i&\sim\mathrm{Beta}(2,3),\\
s_i&\sim\mathrm{Bernoulli}\!\left(\sigma(-0.15+0.6c_e-0.35c_p)\right).
\end{aligned}}
\]

其中 `u_i∈(0,1)` 是预计用血强度，`r_i∈(0,1)` 是临床风险，`s_i=0/1` 是计划/急诊类型；`b_i` 表示是否今天到达。`u_i` 不设为实际发血量的硬上限。`u,r` 相互独立用于分离个体风险与用血强度的配对效应；`s` 可受当天环境影响。

**输入细节**：中心只能看到 `A⁻={(u_i,r_i,s_i):i∈A⁻}`，不额外输入 `b_i`，因为对所有可见患者恒有 `b_i=0`。到达时间标记仅用于医院医生对两类患者采用有差异的优先级/等待规则。

## 3. 阶段 B｜输血科医生的四类判断

所有下面的变量均由**完整医院患者集合** `A^{all}=A⁻∪A⁺` 计算，**不得直接作为 DAT 或基线输入**。

### B1. 临床优先级（包括过去患者的等待优势）

\[
\boxed{\zeta_i=2r_i+u_i+0.5s_i+0.5(1-b_i).}
\]

`ζ_i` 衡量医生排序时的相对重要性：临床风险为主要因素，兼顾预计用血、急诊属性及过去患者的等待/既定治疗安排。旧患者得到 `0.5` 的附加优先级，但不保证胜过当天高度紧急的新患者。

### B2. 患者间优先级挤出——Self-Attention 对应机制

\[
\boxed{
 g(I)=\sigma\!\left(\frac{270-I}{12}\right),\qquad
 T_i=-\frac{g(I)}{10}\sum_{j\in A^{\mathrm{all}},\,j\ne i}
u_j[\zeta_j-\zeta_i]_+\le0.}
\]

- `T_i`：患者 `i` 因其他**更高优先级**患者竞争而遭受的软性发血挤出；竞争者 `j` 的预计用血越高，挤出越大。
- `g(I)`：库存越低，竞争越强；除以固定 `10` 是量纲稳定化而非实际容量约束。
- 当一名更高优先级患者加入时，原低优先级患者的 `T_i` 不增加。该机制刻画病例间资源优先级，并非患者之间的生理因果影响。

### B3. 日级状态—患者画像匹配——Cross-Attention 对应机制

\[
\boxed{
\begin{aligned}
m(c_p)&=0.5+0.3c_p\in[0.2,0.8],\\
d_i&=(1-s_i)c_p+s_ic_e,\\
e_i^{\mathrm{match}}&=-\frac{(u_i-m(c_p))^2}{2(0.18)^2}+1.5d_i+0.3u_i,\\
\omega_i&=\frac{\exp(e_i^{\mathrm{match}})}{\sum_{j\in A^{\mathrm{all}}}\exp(e_j^{\mathrm{match}})}.
\end{aligned}}
\]

- `m(c_p)`：当天业务状态对应的**重点预计用血强度画像**；`0.18` 是匹配带宽，调节关注权重随差距下降的速度。低计划压力下中心落在较低用血强度是**受控合成假设**，并非临床规范。
- `d_i`：根据业务类型选取适用的计划/急诊压力；`1.5` 调节类型与日级压力的匹配；`0.3u_i` 保留同一业务类型内对预计用血强度的轻微偏好。
- `ω_i`：归一化患者关注权重 `ω_i>0, Σ_iω_i=1`，并非“整天只给一人发血”的选择概率。即使所有患者类型相同，`m(c_p)` 变化仍可能改变患者权重。

### B4. 库存稀缺惩罚与共同冲击

\[
\boxed{\lambda(I)=0.25+0.65g(I),\qquad U\sim N(0,1).}
\]

`λ(I)` 是医生面临低库存时的**软稀缺惩罚**（不是硬容量约束的最优影子价格）；高临床风险患者承担较少惩罚。`U` 在同一天的所有患者之间共享，且对血液中心不可见。

## 4. 阶段 C｜医生离散选择与总需求

### C1. 唯一发血强度评分

\[
\boxed{
\begin{aligned}
h_i={}&-0.5+0.8u_i+0.65r_i+0.15s_i+0.4(1-b_i)\\
&+\underbrace{6T_i}_{\text{高优先级患者挤出}}
+\underbrace{5\omega_i r_i}_{\text{日级匹配 × 临床风险}}\\
&-\underbrace{\lambda(I)(1-0.65r_i)}_{\text{库存稀缺}}
+0.35U.
\end{aligned}}
\]

这里四条路径分别对应：患者自身及等待、其他患者、日级状态选择患者、库存稀缺；`6` 和 `5` 为受控合成机制强度，而非真实医院估计系数。`d_i` 不另作直接加性项。

固定原患者特征、当天状态、库存和 `U`，若新增更高优先级患者，则原患者 `T_i` 下降，`ω_i` 被 Softmax 稀释，因 `r_i≥0` 故 `h_i` 不会上升；其发血期望与发血概率不增加。

### C2. 七档随机效用离散选择

\[
\boxed{
\begin{gathered}
\mathcal Y=\{0,1,2,3,4,5,6\},\\
V_i(y)=\begin{cases}
0,&y=0,\\
-0.7+0.5yh_i-0.105y^2,&y=1,\ldots,6,
\end{cases}\\
\Pr(Y_i=y\mid A^{\mathrm{all}},c,I,U)
=\frac{\exp V_i(y)}{\sum_{v=0}^6\exp V_i(v)}.
\end{gathered}}
\]

- `Y_i=0` 表示不发血；1–6 表示标准化发血量。
- 该 Logit 由各行动独立的单位尺度 Gumbel 效用扰动诱导，`-0.7` 为发血启动门槛，负二次项表示边际效用递减。
- **条件独立性**：给定 `A^{all},c,I,U`，不同患者的行动 Gumbel 扰动独立；边际上患者选择因共享 `U`、患者关系及隐藏今日患者而相关。实现用 `logsumexp` 或减最大值的 Softmax 保证稳定。

### C3. 全日总需求

\[
\boxed{D_0=\sum_{i\in A^{\mathrm{all}}}Y_i\in\{0,\ldots,60\},\qquad
J\sim\mathrm{Unif}(0,1),\qquad D=D_0+J\in[0,61].}
\]

`J` 与其余随机量独立，仅提供分位数反演所需的连续扰动；**不单设新的 `D_new` 模型**。两类患者采用完全相同的医生离散选择规则。

## 5. 阶段 D｜血液中心输入、条件 Oracle、补货与评价

### D1. 七维日级向量

\[
\boxed{
X=\left(\frac{c_p+1}{2},\frac{c_e+1}{2},\frac{I-205}{130},
\frac{N^-}{8},\frac{\sum_{i\in A^-}u_i}{8},
\frac{\sum_{i\in A^-}r_i}{8},\frac{\sum_{i\in A^-}s_i}{8}\right)\in[0,1]^7.}
\]

完整可用信息 `Z=(A⁻,X)`，**不含**今日患者、医生评分、真实需求、共享冲击、隐藏新患者的真实人数（真实 `N⁺` 也不能输入）。

### D2. 正确的事前 Full Oracle

\[
\boxed{f^*(Z)=Q_{0.9}(D\mid A^-,X).}
\]

令 `p_k(Z)=Pr(D_0=k|Z)`。正确边际化为

\[
\boxed{
p_k(Z)=\mathbb E_{N^+,A^+,U\mid Z}
\left[\left(\bigast_{i\in A^-\cup A^+}
 p_{Y_i\mid A^-\cup A^+,c,I,U}\right)_k\right],
\quad k=0,\ldots,60.}
\]

**Oracle 信息约束**：不能使用测试日实现的 `N⁺,A⁺,U,Y_i,D` 去计算该日分位数。今日患者一旦加入，医生对**全部**患者的 `T_i,ω_i,h_i` 需重新计算；新增患者不仅贡献自身需求，还可能挤出过去患者。

**建议的数值路线（正确且避免不必要的重复）**：

1. 令 `p_+(c_e)=σ(-0.4+0.5c_e)`，对 `n=0,1,2` 使用**精确 Binomial(2,p_+) 人数权重**。
2. 给定 `n`，对其 `s_j∈{0,1}` **精确枚举** `2^n` 种类型并赋予 Bernoulli 权重；对每名今日患者的连续 `(u_j,r_j)` 使用**随机化 Sobol/QMC**，按 Beta 分布逆 CDF 映射。最大连续积分维数为 **4**，不用模拟日期的真实 `A⁺` 充当 Oracle。
3. 在每个隐藏组合及每个正态 **Gauss–Hermite** 节点 `U=u` 下，计算完整患者的选择概率并**先联合卷积**全部 `Y_i` 的七类 PMF；再对 `U` 积分，最后对隐藏画像求期望。**禁止先对各患者的 `U` 分别积分再卷积。**
4. 连续积分是近似而非有限状态精确枚举。正式主算可预先固定 `25` 个 GH 节点和**经过精度验收的** QMC 点数（例如从 128/256 点启动）；使用更密 QMC、独立 scramble 和少量输入上的 16/25/32 GH 对照核验分位数与 Oracle 成本稳定性。不能未经验收声称 256 点足够。
5. 对每个测试 `Z` 仅计算**一次** `f*(Z)`，五个 `G` 及所有 P1/P2/P3 方法复用；若精度不足只增加积分预算、不改变 DGP 或事后选择模型。

由于 `J~Unif(0,1)`，若 `C_{k-1}(Z)=Σ_{j<k}p_j(Z)<0.9≤C_k(Z)=Σ_{j≤k}p_j(Z)`，则

\[
\boxed{f^*(Z)=k+\frac{0.9-C_{k-1}(Z)}{p_k(Z)}.}
\]

### D3. 统一运营策略与主评价指标

\[
q_{f,G}=[f(Z)+G-I]_+,\qquad I_{\mathrm{end}}=I+q_{f,G}-D,
\]

\[
\boxed{C_G(f)=\mathbb E\left[0.9(G-I_{\mathrm{end}})_++0.1(I_{\mathrm{end}}-G)_+\right].}
\]

**主指标**（Oracle-normalized Prescriptiveness，**不同于**以事后 perfect-foresight 为参照的原始 Bertsimas–Kallus 系数）：

\[
\boxed{
\mathcal P_G(f)=\frac{C_G(\mathrm{SAA})-C_G(f)}{C_G(\mathrm{SAA})-C_G(f^*)},
\qquad C_G(\mathrm{SAA})>C_G(f^*).}
\]

五 `G` 综合必须**先汇总成本差，再取比值**：

\[
\boxed{
\mathcal P_{5G}(f)=\frac{\sum_{G\in\mathcal G}[C_G(\mathrm{SAA})-C_G(f)]}
{\sum_{G\in\mathcal G}[C_G(\mathrm{SAA})-C_G(f^*)]}.}
\]

**辅助指标**：`MeanCost=C_G(f)`，`SL_G=Pr(I_end≥G)`。高服务率可能伴随过量备血，不能仅凭服务率排名。`P` 不裁剪至 `[0,1]`；分母过小或非正须报告。

## 6. 参数表、模型边界与验收

| 类别 | 固定候选参数 | 含义与来源 |
|---|---|---|
| 运营与患者 | `I=270+15c_p+Unif(-50,50)`；`N⁻=3+Binom(5,σ(-.1+.4c_p))`；`N⁺=Binom(2,σ(-.4+.5c_e))` | 保留 v7 的库存和两类患者数量机制 |
| 连续画像 | `u~Beta(2,2)`；`r~Beta(2,3)`；`s~Bern(σ(-.15+.6c_e-.35c_p))` | 连续异质性，保留业务类型离散 |
| 等待优先 | `+0.5(1-b)` 加入 `ζ`；`+0.4(1-b)` 加入 `h` | 此前患者适度优先，今日患者不获此加分 |
| 患者竞争 | `ζ=2r+u+.5s+.5(1-b)`；`6T`；`g=σ((270-I)/12)` | v7 优先级挤出及库存调节 |
| 动态患者匹配 | `m=.5+.3c_p`；带宽 `.18`；`1.5d+0.3u`；`5ωr` | 恢复 v3 风格的日级动态画像匹配 |
| 稀缺与共同噪声 | `λ=.25+.65g`；`0.35U` | 低库存谨慎，高风险较少受抑制，同日相关性 |
| 七档选择 | `V(0)=0`；`V(y>0)=-.7+.5yh-.105y²` | 启动门槛、边际收益递减；保留医生选择机制 |

**最低检查（必须先于正式模型比较）**：

- 每日患者人数与需求支持满足边界；患者行动概率归一、极端剂量未普遍饱和、医生挤出方向正确、患者顺序置换不改变需求分布。
- 对 `A⁺` 边际化而不是泄漏实际今日患者；同一个 `U` 下先卷积再积分；QMC 与 GH 精度检查必须记录。
- 先检查总体 `C(SAA)-C*(L3)`、患者级共享的决策价值及两类机制对真实条件分位数的作用。连续版本 `L2` Oracle 不能靠现成 45 状态枚举；其 Bayes 差距需要独立且正确的条件推断/数值积分。未经计算不得声称 L2→L3 已达到某比例。
- **不承诺** DAT 一定优于 DeepQR、Sum Pool 或 No-SA；机制存在不等于有限样本架构排名。正式运行前冻结参数、数值积分预算与超参数，保留历史离散 v7 的反向结果作为开发记录，不将其伪装为本版结果。

## 7. 轻量伪代码（用于实现核对）

```python
for day in iid_days:
    cp, ce = uniform(-1, 1), uniform(-1, 1)
    I = 270 + 15*cp + uniform(-50, 50)
    Nold = 3 + binomial(5, sigmoid(-0.1 + 0.4*cp))
    Nnew = binomial(2, sigmoid(-0.4 + 0.5*ce))
    old = [draw_patient(cp, ce, today=0) for _ in range(Nold)]
    new = [draw_patient(cp, ce, today=1) for _ in range(Nnew)]
    X = make_X_from_old_only(cp, ce, I, old)
    U = normal(0, 1)
    all_patients = old + new
    zeta = priority(all_patients)       # includes +0.5 for old patients
    T = crowd_out(zeta, all_patients, I)
    omega = context_softmax(all_patients, cp, ce, bandwidth=0.18)
    h = clinician_scores(all_patients, T, omega, I, U)
    Y = [categorical(softmax(choice_utilities(h_i))) for h_i in h]
    J = uniform(0, 1)
    D = sum(Y) + J
    # 只向上游模型暴露 old、X；患者标签仅 Flat 训练时使用。
    yield {
        "A_old": old, "X": X, "I": I, "D": D,
        "Y_old_train_only": Y[:Nold],
        "A_new_private": new, "U_private": U,
    }
```

数值实现须记录生成器版本、随机种子、患者顺序处理、Oracle 精度配置。上面伪代码仅说明生成顺序；`draw_patient` 等函数必须严格按 §§2–4 的公式实现。
