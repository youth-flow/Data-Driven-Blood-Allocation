# DAT 架构与代码实现规范

> **文档状态**：Synthetic 实验实施规范（2026-11-04 整理版）  
> **依据**：`DAT理论保证(3).pdf` 第 1–2、5–7 节；原始 `synthetic_protocol.md` 第 4、7–8 节；本轮确认的实验边界。  
> **适用范围**：合成数据实验 T1–T3、P1–P3；**不涉及医院真实数据实验**。  
> **固定原则**：不修改单周期目标库存成本函数，不修改最新版 DAT 的前向架构。本文中的工程接口为建议的明确实现合同；仓库原有模块名如有差异，以功能和测试结果为准。

## 1. 模型解决的问题

每天的一个样本是**决策前可见的患者多重集合**和**日级运营状态**：

\[
Z_t=(A_t,X_t),\qquad A_t=\{a_{t,1},\ldots,a_{t,N_t}\},\quad 1\le N_t\le M.
\]

需求标签为当日的医院层面总需求 \(D_t\in[0,B]\)，不是单名患者的用血量。DAT 输出一个标量 \(\tau\)-条件分位数预测：

\[
\widehat Q_{\tau,t}=f_{\theta,\tau}(A_t,X_t)\in[0,B].
\]

期初库存 \(I_t\) 是 \(X_t\) 的一个已知坐标；库存目标 \(G\) 是**决策和评价阶段**的外部参数，不是预测器额外的输入。补充量为

\[
q_{\theta,\tau,G}(Z_t)=\big[f_{\theta,\tau}(Z_t)+G-I_t\big]_+.
\]

不同 \(\tau\) 独立训练同一种 DAT 架构；改变 \(G\) 只重新计算补充决策，**不重新训练**。模型不直接输出 \(q\)，也不是多医院联动分配模型。

## 2. 最新 DAT 的精确前向定义

记模型潜维为 \(d=d_{\mathrm{mod}}\)，患者特征维数 \(d_a=3\)，完整日级特征维数 \(d_X=d_x+1=8\)。

### 2.1 患者共享编码器：两仿射层 ReLU

对每个有效患者 \(a_i\in\mathbb R^{d_a}\)：

\[
E_i^{(0)}=\phi_{\mathrm{pat}}(a_i)
=W_{p,2}\operatorname{ReLU}(W_{p,1}a_i+b_{p,1})+b_{p,2}.
\]

- 结构：`Linear(d_a,H_pat) → ReLU → Linear(H_pat,d)`。
- 所有患者 **共享同一组** \(\phi_{\mathrm{pat}}\) 参数。
- 不加入患者顺序位置编码、不进行患者去重；相同特征的两个患者仍是两个 Token。
- 患者集合允许变长，Padding 只用于 Batch 计算，不能参与有效聚合。

### 2.2 单层患者 Self-Attention 残差

\[
\boxed{
E=E^{(0)}+\operatorname{MHA}_{sa}\!\left(
\operatorname{LN}_{sa}(E^{(0)}),
\operatorname{LN}_{sa}(E^{(0)})
\right).
}
\]

- 先逐患者做 **同一个** `LN_sa`，再作为 Self-Attention 的 Query、Key、Value。
- 一层 MHA，一次残差 Add。
- **没有患者侧的第二个 FFN，也没有 Self-Attention 后额外 LayerNorm**。
- Key Padding Mask 必须排除无效患者；Attention 可输出权重供结构检查，但权重不是因果解释。

### 2.3 日级 Encoder：一层仿射映射

\[
\boxed{u=\phi_{\mathrm{day}}(X)=W_dX+b_d\in\mathbb R^d.}
\]

- 默认实现：`Linear(8,d)`。
- 这是最新版理论允许的**仿射 Day Encoder**；理论也允许固定隐藏宽度的两仿射层 ReLU 实现，但当前 Synthetic 主配置使用 Linear。
- 不要把 `Day Encoder` 与后面带非线性的 `Day FFN` 混为一谈。
- 完整日级特征 **已经包含** \(N/M\)；模型 **不得再自动拼接一遍**人数比例。

### 2.4 单层 Day-to-Patient Cross-Attention 残差

\[
\boxed{
v=u+\operatorname{MHA}_{ca}\!\left(
\operatorname{LN}_{q}(u),
\operatorname{LN}_{m}(E)
\right).
}
\]

- Query 为单个日级 Token \(\operatorname{LN}_{q}(u)\)，shape `[batch,1,d]`。
- Key、Value 均为归一化的患者 Memory \(\operatorname{LN}_{m}(E)\)，shape `[batch,n_max,d]`。
- Cross-Attention 输出 `[batch,1,d]`，去掉长度为 1 的 Query 轴后与日级 \(u\) 相加。
- `LN_q` 和 `LN_m` 是**两个独立**的 LayerNorm；与 `LN_sa` 合计恰好 **3 个 LayerNorm**。
- Cross-Attention 也必须正确使用患者 Key Padding Mask。

### 2.5 日级 FFN 残差、线性头、输出裁剪

\[
\boxed{
\begin{aligned}
u^+&=v+\operatorname{FFN}_{day}(v),\\
\operatorname{FFN}_{day}(v)&=W_{f,2}\operatorname{ReLU}(W_{f,1}v+b_{f,1})+b_{f,2},\\
f_{\theta,\tau}(Z)&=\operatorname{clip}_{[0,B]}(w^\top u^++b_o).
\end{aligned}}
\]

- FFN 为 `Linear(d,H_ffn) → ReLU → Linear(H_ffn,d)`。
- **FFN 前没有 LayerNorm**；其输出只与原始 \(v\) 残差相加。
- 输出层为 `Linear(d,1) → clamp(0,B)`；**没有 Final LayerNorm**。
- 训练、评价均使用定义中的 Clip；若大量样本输出在 0 或 \(B\) 处饱和，需检查梯度和初始化，但不能暗中改成 Sigmoid/Softplus 版本冒充原模型。
- `dropout=0`，保证前向过程确定性。

### 2.6 注意力头数

两层 Attention 的头数可分别设定：\(H_{sa}\) 和 \(H_{ca}\) **不要求相等**。原 PDF 允许任意固定头数和每头维数；当前工程默认两者均为 4。

使用 PyTorch 常规等宽多头实现时要求：

\[
d\bmod H_{sa}=0,\qquad d\bmod H_{ca}=0.
\]

原定理对 Cross-Attention 的总 Value 宽度还要求至少 \(d_m\)。当前常规实现总宽度为 \(d\)，两个规模均满足该条件。

## 3. 输入、Padding、Mask 的强制合同

### 3.1 完整信息的 Batch 张量

| 变量 | Shape | 说明 |
|---|---|---|
| `patient_features` | `[batch,n_pad,3]` | 患者顺序任意，填充位置可填 0 |
| `patient_mask` | `[batch,n_pad]` | `True` 表示真实患者；`False` 表示 Padding |
| `day_features` | `[batch,8]` | 已固定归一化的完整日级输入 \(X\) |
| `demand_target` | `[batch]` | 实际日级总需求 \(D\in[0,B]\) |
| `inventory_raw` | `[batch]` | 原单位期初库存，只用于决策及评价；其信息已在 \(X_3\) 中 |
| `scenario` / `sample_id` | Metadata | 不当作预测输入 |

支持每个 Batch 内动态 `n_pad=max(N_t)`；也可统一填充至 \(M\)。有效患者数必须满足 \(1\le N\le M\)。

### 3.2 日级八维特征的固定排列

按**一开始的协议**，使用以下 0-based 索引；不得重新排序、插入额外特征或重复人数信息。

| 索引 | 内容 | 计算 |
|---:|---|---|
| 0 | 计划业务强度 | `(c1 + 1)/2` |
| 1 | 急诊压力 | `(c2 + 1)/2` |
| 2 | 期初库存 | `(I - 225)/180` |
| 3 | 患者人数比例 | `N/M` |
| 4 | 申请量汇总 | `sum(u_i)/M` |
| 5 | 备血量汇总 | `sum(v_i)/M` |
| 6 | 临床风险均值 | `mean(r_i)` |
| 7 | 风险二阶原点矩 | `mean(r_i**2)` |

必须满足：

\[
\texttt{day\_features[:,3]}=\frac{\texttt{patient\_mask.sum(1)}}M
\]

**此断言仅适用于完整信息输入**。P3 的低信息等级刻意隐藏人数时，不能由 `patient_mask` 在模型内部重建并拼回人数，否则会产生信息泄漏。

库存恢复公式为 \(I=225+180X_3\)。原单位 \(I\) 可单独保存在样本中，但不能作为第 9 维特征重复输入。

### 3.3 患者顺序与 Batch Mask

在 PyTorch `nn.MultiheadAttention(batch_first=True)` 中使用：

```python
# patient_mask: True=valid；key_padding_mask: True=ignore
key_padding_mask = ~patient_mask.bool()

# Self-Attention
z = ln_sa(e0)
sa_out, _ = self_attn(z, z, z, key_padding_mask=key_padding_mask,
                      need_weights=False)
e = e0 + sa_out

# Day-to-patient Cross-Attention
u = day_encoder(day_features)
q = ln_q(u).unsqueeze(1)
mem = ln_m(e)
ca_out, _ = cross_attn(q, mem, mem, key_padding_mask=key_padding_mask,
                       need_weights=False)
v = u + ca_out.squeeze(1)

u_plus = v + day_ffn(v)
y_hat = output_head(u_plus).squeeze(-1).clamp(0.0, B)
```

上述代码是前向过程的**功能骨架**，不是声称已有仓库使用相同类名。自注意力中 Padding Query 的输出可以存在，但它不得作为 Cross-Attention 的有效 Key/Value；不要在特征编码阶段把 Padding 当成真实患者。

### 3.4 P3 信息受限输入：不可泄漏

- L0–L2：不得把真实患者个数通过实际 Token 数、有效 Mask 数、额外 `N` 特征传给网络。可采用一个**固定、与实际 N 无关的 Dummy Token**和固定 Mask；8 维 Day 输入中未授权的位置统一门控屏蔽。
- L3：允许真实患者 Token 数和 \((u_i,v_i)\)，但单名患者的 \(r_i\) 必须隐藏；保留 L2 已授权的风险均值与二阶矩。
- L4：完整 \((u_i,v_i,r_i)\) 和完整 8 维 Day 输入。
- 各等级保持模型潜维、模块和可训练参数预算一致，模型不得在内部无条件自动拼接 Mask 推导出的 \(N/M\)。

## 4. 参数配置、参数数目与理论维度

### 4.1 两个预设规模

\[
J=\binom{d_a+M}{M}-1,\qquad
 d_m=\min\{J,2Md_a+1\},\qquad
 d_{\mathrm{eff}}=d_m+d_x.
\]

定量逼近定理的一项**充分条件**为

\[
\boxed{d\geq\max\{d_m+2,d_{\mathrm{eff}}+1\}.}
\]

| 场景 | \(M\) | \(d_m\) | \(d_{\mathrm{eff}}\) | 定理充分下界 | 采用 \(d\) | Self/Cross 头数 |
|---|---:|---:|---:|---:|---:|---:|
| 理论实验 | 2 | 9 | 16 | 17 | 20 | 4 / 4 |
| 性能实验 | 8 | 49 | 56 | 57 | 64 | 4 / 4 |

此维度条件是**显式逼近率构造的充分条件，并非 DAT 前向可运行的必要条件**。

### 4.2 固定的基础超参数

| 项 | 理论主配置 | 性能主配置 |
|---|---:|---:|
| `d_a` | 3 | 3 |
| `day_input_dim` | 8 | 8 |
| `d_model` | 20 | 64 |
| `patient_hidden` | 32 | 128 |
| `day_ffn_hidden` | 32 | 256 |
| `self_heads` | 4 | 4 |
| `cross_heads` | 4 | 4 |
| `layernorm_eps` | `1e-5` | `1e-5` |
| `dropout` | 0 | 0 |
| `weight_decay` | 0 | 0 |
| `demand_cap_B` | 18.5 | 36.5 |
| 总稠密参数数 | **5,801** | **76,161** |

以上参数数目**按 PyTorch 常规 MHA 包含投影偏置**的实现统计；计数必须包含全部权重、偏置、三个 LayerNorm、输出头，以及构造中为零的稠密参数位置。

对于理论规模 \(d=20\)，两个增长 MLP 的隐藏宽度共同为 \(H\) 时：

\[
\boxed{p(H)=3721+65H.}
\]

| \(H\) | 16 | 32 | 64 | 128 | 256 | 512 |
|---:|---:|---:|---:|---:|---:|---:|
| \(p(H)\) | 4,761 | 5,801 | 7,881 | 12,041 | 20,361 | 37,001 |

如果实际代码统计不符，先检查是否额外加入了 FFN、LayerNorm、Projection、Embedding 或重复人数特征，不要直接修改协议中的数字。

## 5. 目标、训练与范数约束

### 5.1 Pinball 训练损失

\[
\tau=\frac b{b+h},\qquad
\rho_\tau(e)=\tau[e]_++(1-\tau)[-e]_+.
\]

\[
\boxed{\mathcal L_\tau(\theta)=\frac1T\sum_{t=1}^T\rho_\tau(D_t-f_{\theta,\tau}(Z_t)).}
\]

正常 DAT 性能训练和理论 T1/T3 都使用真实日级需求 \(D_t\) 计算原始 Pinball 损失。**仅 T2 逼近诊断**使用机制 Oracle 分位数标签进行平方误差拟合；不得将 T2 的训练权重迁移至其他实验。

### 5.2 参数球与逐步投影

理论实验的基准参数球半径为 \(R=100\)。定义完整参数向量（所有参数拼接）：

\[
\theta=(\operatorname{vec}W_1,\ldots,b_o),\qquad \|\theta\|_2\le R.
\]

每个优化器更新后统一做欧氏球投影：

\[
\Pi_R(\theta)=\begin{cases}\theta,&\|\theta\|_2\le R,\\
R\theta/\|\theta\|_2,&\|\theta\|_2>R.
\end{cases}
\]

不能只投影最后一个 Layer、只用 Weight Decay，或将每个参数张量分别投影到半径为 \(R\) 的球——这些都不等同于对完整参数向量的投影。

逼近容量与增长筛类使用事先确定的包络：

\[
\boxed{R_p=100\left(\frac p{5801}\right)^2.}
\]

此式用于理论规模增长类；不是性能规模所有模型的通用 radius 公式。记录每次投影后的真实全局范数，并断言不超过相应 \(R_p\)（允许数值容差）。

### 5.3 实际优化 ≠ 精确 ERM

理论第 5–7 节的高概率保证适用于**精确经验风险最小化解**。在代码中采用 AdamW、梯度裁剪、有限 epoch 和多次初始化，仅是固定数值优化规则；未经认证不能称为全局 ERM。对性能实验可以使用验证集选择超参数，但不要把性能试验的实际优化结果描述为理论已经认证的 ERM。

## 6. 运营决策接口与评价边界

设 \(\hat y=f_{\theta,\tau}(Z)\)，预测器本身**不接收** \(G\)。独立函数实现：

```python
def replenish(predicted_quantile, inventory_raw, G):
    return (predicted_quantile + G - inventory_raw).clamp_min(0.0)

def target_cost(q, demand, inventory_raw, G, tau):
    end_inventory = inventory_raw + q - demand
    return tau * (G - end_inventory).clamp_min(0.0) + (1-tau) * (end_inventory-G).clamp_min(0.0)

def target_service_flag(q, demand, inventory_raw, G):
    return (inventory_raw + q - demand >= G)
```

这里合成实验使用 \(b=\tau,h=1-\tau\)，因此 \(b+h=1\)。**性能三组只向最终结果报告平均成本 `AC` 与期末库存达到 \(G\) 的服务率 `SL_G`**；Pinball 是训练/超参选择用损失，不是性能论文的第三个评价指标。

## 7. 模块拆分建议

```text
synthetic/
  models/dat.py               # DAT 结构、mask、可选 Attention 权重
  data/dataset.py             # Dataset、collate、P3 信息门控
  losses/pinball.py           # 标准 pinball 损失
  operations/policy.py        # q=[f+G-I]_+
  operations/metrics.py       # AC 与目标 SL_G
  training/projection.py      # 全局欧氏参数球投影
  scripts/check_protocol.py   # 参数计数、支持集、对称性、泄漏检查
```

文件路径为建议的职责划分，不表示已核验你仓库中所有路径存在。实际统一实验入口见四号文档。

## 8. 必须通过的自动化测试

| 编号 | 测试 | 合格标准 |
|---|---|---|
| A01 | 输出形状及范围 | `[batch]` 且全部在 `[0,B]` |
| A02 | 患者置换不变性 | 同时置换特征和 Mask，预测不变（浮点容差内） |
| A03 | Padding 不变性 | 仅增加无效 Padding，预测不变 |
| A04 | 患者人数验证 | 完整信息 `X[:,3] == n_valid/M`；P3 不得触发该信息自动注入 |
| A05 | 参数计数 | 理论 5,801、性能 76,161 |
| A06 | 架构一致性 | 1 SA、1 CA、3 LN、1 Day FFN；无患者 FFN、FFN 前 LN、Final LN、位置编码 |
| A07 | 全局参数投影 | 每次更新后总 \(\ell_2\) 范数不超过 \(R\) |
| A08 | 独立头数 | 自注意力和交叉注意力头数分别可配置 |
| A09 | \(G\) 的外部性 | 改变 \(G\) 不改变 DAT 分位数预测，只改变 \(q\) |
| A10 | P3 信息泄漏 | L0–L2 的 Token 数/Mask 不含真实 \(N\)；L3 不含逐患者 \(r_i\) |
| A11 | 分位数独立训练 | 不同 \(\tau\) 权重独立；无跨分位强制排序 |
| A12 | Oracle 隔离 | 正常 DAT 训练不读取 oracle 分位数标签，T2 例外 |

## 9. 理论解释与不得夸大的结论

- **命题 1**：网络对患者排列不变，可通过 A02 验证数值实现。
- **命题 9**：抽样前固定的紧参数类、参数 Lipschitz 性与**精确 ERM**产生有限样本估计误差上界。
- **定理 10**：在人数有界、Hölder 目标及固定潜维充分条件下，存在宽度增长的范数受控 DAT 逼近器：

  \[
  \inf_{f\in\mathcal F_P}\|f-f^\star\|_{\infty,K}\le CP^{-\gamma},\qquad
  \gamma=\frac\alpha{Md_{\mathrm{eff}}}.
  \]

- **定理 17、推论 19/24**：运营遗憾、增长筛类速率及一致性，仍需相应统计和优化假设；实际有限步 AdamW 不自动被证明具有这些性质。
- 存在性逼近构造可以把 SA 置零、CA 置为均匀权重。这证明**类有表达能力**，并不能证明动态注意力在真实数据或模拟中一定优于 Mean Pool；需要 P2 消融检验。

---

**验收结论**：只要 A01–A12 全部通过，并且训练/评价接口符合上述合同，即可将该模型视为本 Synthetic 协议的**最新版 DAT 架构实现**。
