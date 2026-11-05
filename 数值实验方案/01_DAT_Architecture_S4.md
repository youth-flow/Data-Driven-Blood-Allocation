# 01｜DAT 模型架构与代码实现规范（S4 主协议）

> **状态**：2026-11-05 整合版；供 Synthetic 代码工程直接实现。  
> **适用范围**：理论实验 T1–T3、性能实验 P1–P3。正式 Synthetic 需求机制 **只使用 S4**；T1-C0 是独立的精确 ERM 校准环境，不算作第二个性能场景。  
> **主分位数**：`tau = 0.9`。其他分位数仅可作为单独标识的辅助实验。  
> **超参数规则**：所有方法的训练与结构配置必须在正式运行前**人为指定、冻结并归档**；本文不提供学习率、批量大小、Epoch、树数等具体数值，也不允许网格/随机/贝叶斯超参数搜索。  
> **资料边界**：依据已确定的最新版 DAT 架构、原 AXID DGP，以及本轮确定的实验口径。医院真实数据研究独立于本协议。

---

## 1. 预测任务、集合输入与运营决策

一天是一个决策样本：

\[
Z_t=(A_t,X_t),\qquad A_t=\{a_{t1},\ldots,a_{tN_t}\},\quad 1\le N_t\le M.
\]

- `A_t`：决策前已知的**无序、多重**患者订单集合；不同患者允许相同特征，不能去重。
- `a_{ti}=(u_{ti},v_{ti},r_{ti})∈[0,1]^3`：归一化申请、备血和临床风险。
- `X_t∈[0,1]^8`：完整八维日级输入，包含期初库存和 `N_t/M`。
- `D_t∈[0,B]`：决策日**总需求**标签；模型不是逐患者需求预测器。

DAT 输出条件分位数：

\[
\widehat Q_{\tau,t}=f_{\theta,\tau}(A_t,X_t)\in[0,B].
\]

运营策略作为独立模块实现：

\[
\boxed{q_{\tau,G,t}=[\widehat Q_{\tau,t}+G-I_t]_+.}
\]

预测器**不把** `G` 作为输入；更换 `G` 时不重新训练。主任务使用 `tau=0.9`；辅助 `tau` 使用独立训练的直接分位数模型，不与主模型混合。

## 2. 唯一允许的主 DAT 前向计算

记模型潜维 `d=d_model`，患者编码为共享两仿射层 ReLU：

\[
E^{(0)}_i=\phi_{\rm pat}(a_i)
=W_{p2}\operatorname{ReLU}(W_{p1}a_i+b_{p1})+b_{p2}.
\]

### 2.1 一层患者 Self-Attention

\[
\boxed{E=E^{(0)}+\operatorname{MHA}_{sa}(\operatorname{LN}_{sa}(E^{(0)}),\operatorname{LN}_{sa}(E^{(0)})).}
\]

这里自注意力的 Q/K/V 都来自同一个 `LN_sa(E0)`；存在残差加法，但**没有**患者侧 FFN、额外归一化或患者位置编码。

### 2.2 日级仿射编码

\[
\boxed{u=\phi_{\rm day}(X)=W_dX+b_d.}
\]

`Day Encoder` 主模型是 `Linear(8,d)`。它不承担融合后的决策非线性；后者由 `Day FFN` 承担。`X` **已经有八维并包含人数**，禁止在模型前向内部再追加一次 `N/M`。

### 2.3 一层日级 Query → 患者 Memory 的 Cross-Attention

\[
\boxed{v=u+\operatorname{MHA}_{ca}(\operatorname{LN}_q(u),\operatorname{LN}_m(E)).}
\]

- Q：**一个**日级 Token，`[batch,1,d]`。
- K/V：所有有效患者 Token，`[batch,n_pad,d]`。
- `LN_sa`、`LN_q`、`LN_m` 是三个**互不共享参数**的 LayerNorm。
- 其输出去掉长度为 1 的 Query 轴后，与原始日级向量 `u` 残差相加。

### 2.4 日级 FFN 与直接读出

\[
\begin{aligned}
\operatorname{FFN}_{day}(v)&=W_{f2}\operatorname{ReLU}(W_{f1}v+b_{f1})+b_{f2},\\
u^+&=v+\operatorname{FFN}_{day}(v),\\
\boxed{f_{\theta,\tau}(Z)&=\operatorname{clip}_{[0,B]}(w^\top u^++b_o).}
\end{aligned}
\]

必须坚持：

1. 日级 FFN **前没有** LayerNorm。
2. 最终读出 **前没有** Final LayerNorm。
3. 输出就是标量线性层加 `[0,B]` 的硬裁剪；不悄悄替换为 Sigmoid、Softplus 或另一架构。
4. Self-Attention 与 Cross-Attention 都只使用一层；隐藏宽度与注意力头数由运行前冻结的配置决定。
5. 训练/评价期间不引入随机 Dropout；固定的 LayerNorm 稳定项严格为正。

### 2.5 结构示意

```text
patient_features [B,N_pad,3]                  day_features [B,8]
        │                                             │
 shared Patient MLP                             Day Linear
        │                                             │
    E0 [B,N,d]                                      u [B,d]
        │                                             │
 LN_sa → Self-MHA → residual Add                      │
        │                                             │
    E [B,N,d]                                         │
        └────── LN_m → K,V ───────┐    LN_q(u) → Q ───┘
                                  │
                              Cross-MHA
                                  │
                               Add u → v
                                  │
                          Day FFN → Add → u+
                                  │
                         Linear → Clip[0,B]
                                  │
                      predicted conditional quantile
```

## 3. 数据接口及维度合同

| 名称 | 张量形状 | 必需含义 |
|---|---|---|
| `patient_features` | `[batch,n_pad,3]` | 有效患者 `(u,v,r)`，Padding 位置无效 |
| `patient_mask` | `[batch,n_pad]` | `True=valid`, `False=padding` |
| `day_features` | `[batch,8]` | 完整日级输入，不得自动拼人数 |
| `demand_target` | `[batch]` | 训练标签 `D`，单位与需求上界相同 |
| `inventory_raw` | `[batch]` | 原单位库存，仅在补充策略/评价时使用 |
| `sample_id, replicate_id` | metadata | 审计用，不作为预测特征 |

`day_features` 的 **0-based** 固定次序：

| 列索引 | 特征 | 精确定义 |
|---:|---|---|
| 0 | 计划强度 | `(c1+1)/2` |
| 1 | 急诊压力 | `(c2+1)/2` |
| 2 | 库存归一化 | `(I-225)/180` |
| 3 | 人数比例 | `N/M` |
| 4 | 申请汇总 | `sum(u_i)/M` |
| 5 | 备血汇总 | `sum(v_i)/M` |
| 6 | 风险均值 | `mean(r_i)` |
| 7 | 风险二阶原点矩 | `mean(r_i**2)` |

完整信息等级下，可断言 `X[:,3] == patient_mask.sum(1)/M`，并可从 `I=225+180*X[:,2]` 恢复库存；**P3 限制信息等级不适用第一条断言**，禁止据此反推出被隐藏的患者人数。

## 4. Attention Mask 与置换不变性

PyTorch `nn.MultiheadAttention(batch_first=True)` 使用 `key_padding_mask=True` 表示要屏蔽的位置，因此需要反转有效 Mask：

```python
valid = patient_mask.bool()
assert valid.any(dim=1).all()
ignore = ~valid

e0 = patient_encoder(patient_features)
sa_input = ln_sa(e0)
sa_out, _ = self_attn(sa_input, sa_input, sa_input,
                      key_padding_mask=ignore, need_weights=False)
e = e0 + sa_out

u = day_encoder(day_features)
q = ln_q(u).unsqueeze(1)
memory = ln_m(e)
ca_out, _ = cross_attn(q, memory, memory,
                       key_padding_mask=ignore, need_weights=False)
v = u + ca_out.squeeze(1)
u_plus = v + day_ffn(v)
y_hat = output_head(u_plus).squeeze(-1).clamp(0.0, B)
```

- 有效患者可以任意重排；输出应在浮点容差内不变。
- 同时重排 `patient_features` 和 `patient_mask`；不能只重排其一。
- 无效 Padding 行可以产生自注意力 Query 输出，但**绝对不能**成为后续 Cross-Attention 的有效 Key/Value。
- `N>=1` 为硬约束，禁止将一个全无效的患者集合喂给主 DAT。

## 5. 理论尺度与参数量

理论中的维数记号：

\[
J=\binom{d_a+M}{M}-1,\quad d_m=\min\{J,2Md_a+1\},\quad d_{\rm eff}=d_m+d_x.
\]

用于显式构造性逼近率的一个**充分条件**是：

\[
\boxed{d_{\rm model}\ge \max\{d_m+2,d_{\rm eff}+1\},\qquad
\text{Cross-Attention 总 Value 宽度}\ge d_m.}
\]

| 用途 | `M` | `d_a` | `d_x` | `d_m` | `d_eff` | 潜维 |
|---|---:|---:|---:|---:|---:|---:|
| T1–T3 | 2 | 3 | 7 | 9 | 16 | 20 |
| P1–P3 | 8 | 3 | 7 | 49 | 56 | 64 |

工程还须满足 `d_model % self_heads == 0`、`d_model % cross_heads == 0`（使用等宽多头时）。两个头数可以不同，均需事先固定。

**参数预算 `P` 是全部稠密标量参数位置数**，包括权重、偏置、LayerNorm、MHA 和输出头，即使构造中某些位置被设为零也计入。不得把 `P` 与欧氏球半径 `R_P` 混淆。

对理论潜维 20、Patient MLP 与 Day FFN 同宽为 `H` 的已有标准实现，参数数量为：

\[
\boxed{P(H)=3721+65H.}
\]

它用于 T2、T3 的容量轴和运行时参数计数；**不要**从模型名称或隐藏宽度直接猜测参数量，应在程序中读取 `sum(p.numel() for p in model.parameters())` 并断言和该公式一致。性能 DAT 的 MLP 宽度、人为固定的训练配置等，在正式运行配置文件中记录，此处不枚举具体值。

## 6. 训练规则与理论适用范围

### 6.1 主训练：Pinball 损失

\[
\rho_\tau(e)=\tau[e]_++(1-\tau)[-e]_+,\qquad
\mathcal L_\tau(\theta)=\frac1T\sum_t\rho_\tau(D_t-f_{\theta,\tau}(Z_t)).
\]

- 主任务 `tau=0.9`；若执行辅助分位数，直接分位数模型应按新的 `tau` **独立训练**。
- T1-DAT、T3 和性能 DAT 正常训练仅使用实际日需求 `D`；不得读取 Oracle 分位数作监督。
- **仅 T2** 的专用构造性逼近诊断使用 Oracle 分位数标签并以平方误差拟合，所得权重不复用到其他组。
- 优化器、学习率、批大小、训练轮数、森林/树模型配置等，统一事先人工确定并写入运行配置；**没有候选搜索、验证选优、测试选优或最佳测试 Seed**。

### 6.2 理论实验参数球

理论固定函数类使用预定闭欧氏参数球 `Theta_P={theta:||theta||_2<=R_P}`。对需要范数约束的 T1–T3，优化更新后应对**完整参数向量**执行欧氏球投影，而不是按层单独投影：

\[
\Pi_{R_P}(\theta)=\theta\min\{1,R_P/\|\theta\|_2\}.
\]

在 T2/T3 增长类中，参数半径是事先定义的 `P` 的多项式函数；参见 `03_Theory_Experiments_S4.md`。实际优化器即使投影成功，也**不等同于**找到了精确 ERM；定理的准确适用条件应在论文中分开叙述。

## 7. P3 五级信息授权：必须防泄漏

| 等级 | 授权日级信息 | 授权患者信息 | 实现方式 |
|---|---|---|---|
| L0 | `c1,c2,I` | 无 | 真正固定的 Dummy Token 与固定 Mask |
| L1 | L0 + `N/M`、申请/备血汇总 | 无 | 仍使用固定 Dummy Token |
| L2 | L1 + 风险均值/二阶矩 | 无 | 仍使用固定 Dummy Token |
| L3 | L2 | 逐患者 `(u_i,v_i)` | 真实 Token/Mask，但 `r_i=0` 屏蔽 |
| L4 | 全部八维 `X` | 完整 `(u_i,v_i,r_i)` | 正常 DAT 输入 |

- 未授权日级坐标统一置为不含个体信息的常数；不修改日级张量宽度。
- L0–L2 的 Token 数和 Mask 不得透露真实 N；不能在模型内部通过 Mask 自动添加 `N/M`。
- L3 保留已经授权的汇总风险量，但禁止使用逐患者风险及通过其他通道泄漏其对应关系。
- 五级使用同一个 DAT 模块布局、相同参数预算与训练规则，从头独立训练，不迁移权重。

## 8. 运营评价接口

性能实验最终指标严格只有两个：

\[
\boxed{\mathrm{MeanCost}_{\tau,G}=\frac1{n_{\rm test}}\sum_t
\left\{\tau[G-(I_t+q_t-D_t)]_+ +(1-\tau)[I_t+q_t-D_t-G]_+\right\}}
\]

\[
\boxed{\mathrm{SL}_G=\frac1{n_{\rm test}}\sum_t\mathbf1\{I_t+q_t-D_t\ge G\}.}
\]

- `MeanCost` 是**单个测试日平均成本**，**不是 3,000 天总成本**。
- `SL_G` 是**期末库存达到目标 G 的频率**，**不是**需求完全满足率或分位数覆盖率。
- 理论实验仍可以记录用于检验定理的 Oracle L1、超额 Pinball 风险和运营遗憾；这些不进入 P1–P3 的最终性能评价列。

```python
def replenish(qhat, inventory, target_G):
    return (qhat + target_G - inventory).clamp_min(0)

def evaluate_operational(q, inventory, demand, target_G, tau):
    end = inventory + q - demand
    daily = tau * (target_G-end).clamp_min(0) + (1-tau) * (end-target_G).clamp_min(0)
    return {"MeanCost": daily.mean(), "SL_G": (end >= target_G).float().mean()}
```

## 9. 工程验收

1. **结构**：一层 SA、一层 CA、三个 LN、一个 Day FFN，无患者 FFN/Final LN/患者位置编码。
2. **形状和范围**：输出 `[batch]`、位于 `[0,B]`；完整 `X` 恰好八维。
3. **置换/填充不变性**：有效患者任意重排、增加无效 Padding 均不改变输出（浮点容差内）。
4. **Mask**：任一有效日必须有患者；Padding 不影响任一聚合。
5. **信息权限**：P3 的 L0–L2 不泄漏人数，L3 不泄漏逐患者风险。
6. **参数统计**：理论潜维与容量类的 `P(H)` 逐档一致；全局欧氏投影后范数合格。
7. **预测与决策分离**：改变 `G` 不触发重新训练，也不改变模型输出的分位数。
8. **S4 Only**：除 T1-C0 校准外，所有合成样本都来自 S4；模型输入不含场景或未来结果标识。
9. **单主分位数**：正式主结果 `tau=0.9`；辅助 `tau` 如有单独登记。
10. **无自动调参**：训练配置在运行前冻结，不依据测试/Oracle 结果改变。

---

**本文件约束模型结构；概率生成见 `02_Synthetic_DGP_S4.md`；理论数值实验见 `03_Theory_Experiments_S4.md`；九模型、消融及信息实验见 `04_Performance_Experiments_S4.md`。**
