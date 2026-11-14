# 02｜AXID-DC v8：两级血液供应链的医生离散选择 DGP

> **版本**：`axid_dc_v8_hb_context`；与已运行的 v3、v7、v7-Final **不同 DGP，不能沿用旧结果或旧 Oracle**。本文件是本轮完整生成合同；参数为受控合成设定，不是对真实临床行为的估计。配套性能协议：`04_AXID_DC_v8_Performance_Protocol.md`。
>
> **简洁性原则**：无额外隐含病例组合状态 `W`、无业务类型 `s`、无独立 `D_new`、无硬库存约束。医生机制只有 **优先级 → 挤出效应 → 上下文化风险 → 日级聚合 → 七档发血选择**。

## 1. 两个主体、两类患者与信息边界

- **血液中心**在订货时只看到过去登记患者集合 `A⁻`、日级运营状态 `c=(c_p,c_e)`、共享库存 `I`，用事前信息 `Z=(A⁻,X)` 决定补货 `q`，希望 `I+q-D` 接近目标库存 `G`。
- **医院输血科**发血时看到过去患者 `A⁻` 与今日新增患者 `A⁺` 的完整集合 `A=A⁻∪A⁺`，以及 `c,I`，逐人选择拟发血量 `Y_i`，形成总需求 `D`。
- 今日患者**全部在中心订货后出现**；血液中心不知道 `A⁺`、医生随机选择或 `D`，医院不知道当次中心实际补货 `q`。`D` **不依赖** `q`；`I` 是双方共同观察的区域库存／安全储备信号，不作为医生发血量的硬上限。
- 不模拟跨日队列；不同决策日 i.i.d.。`b_i=0` 表示过去患者、`b_i=1` 表示今日患者，仅为**医院生成机制内部的到达标签**；全部可见患者的 `b_i=0`，不传给 DAT。

记 `σ(x)=(1+e^{-x})^{-1}`，`[x]₊=max(x,0)`，同日共享一个未观测正态冲击 `U`。

## 2. 阶段 A：日级状态和两批患者

**A1｜日级运营与库存**

\[
\boxed{c_p,c_e\overset{\rm iid}{\sim}{\rm Unif}(-1,1),\quad
\varepsilon_I\sim{\rm Unif}(-50,50),\quad
I=270+15c_p+\varepsilon_I\in[205,335].}
\]

`c_p`：计划业务压力；`c_e`：急诊压力；`I`：事前库存信号。以上噪声独立。

**A2｜人数**

\[
\boxed{\begin{aligned}
N^-&=3+{\rm Binom}\bigl(5,\sigma(-0.1+0.4c_p)\bigr),\\
N^+&={\rm Binom}\bigl(2,\sigma(-0.4+0.5c_e)\bigr).
\end{aligned}}
\]

故 `3≤N⁻≤8`、`0≤N⁺≤2`、全日 `N≤10`；`A⁻` 全部可见，`A⁺` 全部不可见。

**A3｜患者画像**。两批患者遵循**相同**、相互独立的临床特征生成机制：

\[
\boxed{a_i=(u_i,r_i,\mathrm{Hb}_i),\quad
u_i\sim{\rm Beta}(2,2),\quad
r_i\sim{\rm Beta}(2,3),\quad
\mathrm{Hb}_i\sim{\rm Beta}(2,2).}
\]

`u_i`：预计用血强度；`r_i`：综合临床风险；`Hb_i`：标准化血红蛋白水平，数值越高表示越充足。定义 **Hb 不足程度**

\[\boxed{\overline{\mathrm{Hb}}_i=1-\mathrm{Hb}_i.}\]

上横线只表示**不足程度**，**不是**患者平均 Hb。三个临床特征在本受控 DGP 中相互独立；新/旧患者仅到达时间标签 `b_i` 不同，医生发血规则相同。`u_i` 不是 `Y_i` 的硬上限。

## 3. 阶段 B：医生从患者优先级形成全日判断

**以下 `ζ,T,\widetilde r,ω,R` 一律使用医院实际看到的完整集合 `A=A⁻∪A⁺` 计算，不能作为血液中心模型的直接输入。**

**B1｜患者优先级（Patient Priority）**

\[
\boxed{\begin{aligned}
\zeta_i={}&1.5r_i+0.75u_i
+0.5\overline{\mathrm{Hb}}_i
+0.75\overline{\mathrm{Hb}}_i^{\,2}\\
&+0.75r_i\overline{\mathrm{Hb}}_i
+0.5(1-b_i).
\end{aligned}}
\]

临床风险、预计用血、Hb 不足程度（含加速项与风险互补项）、过去患者等待优先权共同决定优先级。其对 Hb 不足的边际作用为 `0.5+1.5·Hb不足+0.75r_i>0`；所有系数均为合成行为参数。

**B2｜患者挤出效应（Patient Crowding-out）**

\[
\boxed{T_i=-\frac1{10}\sum_{j\in A,\,j\ne i}
u_j[\zeta_j-\zeta_i]_+\le0.}
\]

只有优先级更高的患者对当前患者产生**直接挤出**；对方预计用血越大、优先级差越大，挤出越明显。`10` 为固定最大患者数的尺度因子，不是硬容量约束。`T_i` 不直接依赖日级信息，以对应 DAT 先进行 Patient Self-Attention。

**B3｜上下文化风险（Contextualized Risk）**

\[
\boxed{\widetilde r_i=r_i+0.4\overline{\mathrm{Hb}}_i+0.6T_i.}
\]

`\widetilde r_i` 是经过患者竞争修正的**评分**，并非风险概率，不强制落在 `[0,1]`。

**B4｜日级函数（Day Context）**

\[
\boxed{\begin{aligned}
g_1(c_p)&=0.5+0.3c_p,\\
g_2(c_e)&=\frac{c_e+1}{2},\\
g(I)&=\sigma\!\left(\frac{270-I}{12}\right).
\end{aligned}}
\]

`g₁` 是计划业务状态对应的重点预计用血画像；`g₂` 是归一化急诊压力；`g(I)` 是库存紧张程度。

**B5｜日级注意权重与全日综合风险（Day-Conditioned Aggregation）**

\[
\boxed{\begin{aligned}
e_i={}&-\frac{(u_i-g_1(c_p))^2}{2(0.22)^2}
+1.5g_2(c_e)\widetilde r_i+0.4g(I)\zeta_i,\\
\omega_i&=\frac{\exp(e_i)}{\sum_{j\in A}\exp(e_j)},\\
R&=\sum_{i\in A}\omega_i\widetilde r_i.
\end{aligned}}
\]

`ω_i` 是**关注权重而非发血概率**。计划压力调整重点用血画像；急诊压力更关注上下文化风险；库存紧张时更关注高优先级患者。`R` 是**全日综合病例风险评分**，先上下文化、再由日级状态聚合，与 DAT 的 SA→CA 顺序一致。此处 `R` **不表示** Flat 的剩余需求。

## 4. 阶段 C：医生七档离散发血选择

**C1｜基础边际效用指数**

\[
\boxed{h_i=-1.9+0.35\zeta_i+3g(I)T_i+2.4R+0.175U,\qquad U\sim N(0,1).}
\]

`h_i` 是正发血量二次效用的**基础边际收益指数**，不等于所有剂量下的实际边际效用。`3g(I)T_i≤0` 表示库存紧张时患者竞争加剧；`2.4R` 为影响全部患者的日级共同病例压力。由于 `R` 也可能随患者组成变化，新增高优先级患者虽增加直接挤出，不保证对各人**净**效用的影响严格为负。

**C2｜随机效用离散选择**

\[
\boxed{\begin{gathered}
\mathcal Y=\{0,1,2,3,4,5,6\},\\
V_i(y)=\begin{cases}
0,&y=0,\\
-0.7+yh_i-0.105y^2,&y=1,\ldots,6,
\end{cases}\\
\Pr(Y_i=y\mid A,c,I,U)=\frac{e^{V_i(y)}}{\sum_{v=0}^{6}e^{V_i(v)}}.
\end{gathered}}
\]

为每个患者—行动加入独立单位尺度 Gumbel 扰动，选择最大随机效用得到上述 Logit。`−0.7` 为正发血启动门槛，负二次项使边际净效用递减；`\partial V_i/\partial y=h_i-0.21y`（仅为正剂量的连续延拓）。给定 `A,c,I,U` 时各患者个体选择扰动独立；**对血液中心的事前信息而言患者需求仍可相关**。

**C3｜总需求**

\[
\boxed{D_0=\sum_{i\in A}Y_i\in\{0,\ldots,60\},\quad
J\sim{\rm Unif}(0,1),\quad D=D_0+J\in[0,61]\ \text{a.s.}}
\]

`J` 独立于其他量，只使需求分位数连续；不单设新患者残差生成器，也不施加医院发血总量的硬库存约束。实际抽样时每日只采一个 `U`，所有患者共享。

## 5. 阶段 D：血液中心输入、Oracle 与补货

**D1｜事前输入接口**

\[
\boxed{\begin{aligned}
A^-&=\{(u_i,r_i,\mathrm{Hb}_i):i\text{ 为过去患者}\},\\
X&=\left(\frac{c_p+1}{2},\frac{c_e+1}{2},\frac{I-205}{130},
\frac{N^-}{8},\frac{\sum_{A^-}u_i}{8},
\frac{\sum_{A^-}r_i}{8},\frac{\sum_{A^-}\overline{\mathrm{Hb}}_i}{8}\right)\in[0,1]^7,\\
Z&=(A^-,X).
\end{aligned}}
\]

`Hb` 和 `Hb 不足` 可相互计算；中心只看到过去患者，不能读取今日患者 `A⁺`、其人数实现值、医生内部 `ζ,T,\widetilde r,ω,R,h,U` 或测试日真实 `D`。

**D2｜血液中心事前 Full Oracle**（不是事后 Perfect Foresight）

\[
\boxed{f^*(Z)=Q_{0.9}(D\mid A^-,X).}
\]

令 `p_k(Z)=Pr(D_0=k|Z)`，对今日患者及共享冲击积分：

\[
\boxed{p_k(Z)=\mathbb E_{N^+,A^+,U\mid Z}
\!\left[\left(\mathop{*}_{i\in A^-\cup A^+}
  p_{Y_i\mid A^-\cup A^+,c,I,U}\right)_k\right],\quad k=0,\ldots,60.}
\]

**正确且高效的数值顺序**：

1. 用 `c_e` 计算 `N⁺=0,1,2` 的 **Binomial 精确权重**；对 `N⁺=n` 的连续特征 `(u,r,Hb)^{n}` 使用随机化 Sobol/QMC（**最多 6 维**；不是旧版的 4 维或 45 状态枚举）。Beta 逆 CDF 将 `[0,1]` QMC 坐标映射到特征；`n=0` 无需 QMC。
2. 对每个候选今日患者集合和每个共同 `U` 的 Gauss–Hermite 节点，**重新计算完整患者的** `ζ,T,\widetilde r,ω,R,h`，构造所有人的 7 类 PMF，**先联合卷积**；再对 `U` 积分、对隐藏今日患者及人数加权。**禁止分别边际化共享 `U` 后再卷积患者 PMF**。
3. 得到 `D₀` PMF，结合独立 `J~Unif(0,1)`：若 `\sum_{j<k}p_j<\tau\le\sum_{j\le k}p_j`，则

   \[\boxed{Q_\tau(D\mid Z)=k+\frac{\tau-\sum_{j<k}p_j(Z)}{p_k(Z)}.}\]

4. `τ=0.9` 固定；Oracle 每个测试日**只求一个分位数**，五个 `G` 全部复用。QMC 点数依据独立小样本精度验收冻结；可批量计算、分块卷积，但不能因预测模型表现改 Oracle 精度。

**D3｜运营决策与指标**

\[
\boxed{\begin{aligned}
q_{f,G}&=[f(Z)+G-I]_+,\quad I_{\rm end}=I+q_{f,G}-D,\\
C_G(f)&=\mathbb E[0.9(G-I_{\rm end})_++0.1(I_{\rm end}-G)_+],\\
\mathcal P_G(f)&=\frac{C_G(\mathrm{SAA})-C_G(f)}{C_G(\mathrm{SAA})-C_G(f^*)},\\
SL_G(f)&=\Pr(I_{\rm end}\ge G).
\end{aligned}}
\]

主指标为 **Oracle-normalized Prescriptiveness**，辅助指标为 MeanCost 与目标服务率。固定 `G={260,280,300,320,340}`、`τ=0.9`；五 `G` 综合 P 必须**先汇总成本差再取比值**。分母非正/近零须标识，不裁剪 P。

## 6. 生成器数据结构与最小校验

```python
# 每个决策日 i.i.d.；以下为接口级伪代码，不是完整可执行程序
def generate_day(rng):
    cp, ce = rng.uniform(-1, 1), rng.uniform(-1, 1)
    I = 270 + 15*cp + rng.uniform(-50, 50)
    n_old = 3 + rng.binomial(5, sigmoid(-0.1 + 0.4*cp))
    n_new = rng.binomial(2, sigmoid(-0.4 + 0.5*ce))
    old = sample_patients(n_old, rng, b=0)   # (u,r,Hb)
    new = sample_patients(n_new, rng, b=1)   # 同一三维画像抽样律
    all_patients = old + new
    zeta = priority(all_patients)            # b 只在医院内部使用
    T = crowding_out(all_patients, zeta)     # 不含 I
    r_tilde = contextualized_risk(all_patients, T)
    omega, R = day_attention(all_patients, r_tilde, zeta, cp, ce, I)
    U = rng.normal()
    pmf_each = logit_action_pmfs(all_patients, zeta, T, R, I, U)
    Y_all = [rng.choice(range(7), p=pi) for pi in pmf_each]
    D = sum(Y_all) + rng.uniform(0, 1)
    X = build_X_from_old_only(old, cp, ce, I)
    return {
        'predictor_input': {'A_minus': old, 'X': X, 'I': I},
        'train_labels': {'D': D, 'Y_minus': Y_all[:n_old]},
        'private': {'A_plus': new, 'Y_plus': Y_all[n_old:], 'U': U},
    }
```

`Flat` 的训练剩余需求标签必须命名为 `D_rem = D − Σ_{i∈A⁻}Y_i`，**不得与全日风险 `R` 混名**。

**必要验收**：七类行动 PMF 和 Oracle 条件总 PMF 非负/归一；患者重排不改变分布或分位数；今日患者不可泄漏；比较少量输入上的 QMC 加密与 16/25/32 GH 分位数；检查零/最大剂量概率是否大量饱和。**本版尚未运行正式 Oracle 或 P1/P2/P3；不声称 DAT 必胜，也不将旧版本结果迁移至本版。**
