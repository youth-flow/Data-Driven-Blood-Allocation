# AXID-DC v10｜单一 G 的快速正式性能协议

> **冻结口径**：本文件对应 `scenario=AXID-DC-v10`。DGP 的
> `schema_version=generator_version=10`；性能协议因历史 v8 已使用
> `protocol_version=10`，故本版性能合同使用 `protocol_version=11`。这三个编号描述的对象不同。

## 1. 唯一目标与最终指标

医院 MDCEV 与血液中心评价共同使用

\[
G=256,\qquad \tau=0.9.
\]

对任一方法 \(m\)：

\[
q_{m,t}=[\widehat f_m(Z_t)+G-I_t]_+,\qquad
E_{m,t}=I_t+q_{m,t}-D_t,
\]

\[
c_{m,t}=0.9(G-E_{m,t})_++0.1(E_{m,t}-G)_+.
\]

正式结果只报告四项：

\[
\operatorname{MeanCost}_m=T^{-1}\sum_t c_{m,t},\qquad
SL_{256}(m)=T^{-1}\sum_t\mathbf 1\{E_{m,t}\ge256\},
\]

\[
P_{PF}(m)=\frac{C_{SAA}-C_m}{C_{SAA}-C_{PF}},\qquad
P_{Oracle}(m)=\frac{C_{SAA}-C_m}{C_{SAA}-C_{Oracle}}.
\]

两个 \(P\) 不裁剪到 \([0,1]\)；分母非正或绝对值不大于合同容差时标记不可解释。
Pinball 只作为训练损失和运行诊断，不是最终性能指标。不存在多 \(G\) 聚合或
`G=340` 特例。

## 2. 样本与训练预算

| 项目 | 冻结值 |
|---|---:|
| 训练日 | 524,288 |
| 测试日 | 10,000 |
| Validation | 0 |
| Batch | 128 |
| Epoch | 1 |
| optimizer steps | 4,096 |
| 随机初始化 | 1 |

因此每个神经配置恰好看到一次训练池：\(128\times4096=524,288\)。无早停、无超参搜索、
不使用测试集选择 checkpoint；最终权重为第 4,096 步权重。优化器仍为 AdamW，300 步
线性 warm-up 后 cosine 衰减，正式精度为 FP32。

## 3. P3、P2、P1

- P3：`L0/L1/L2/L3` 四个信息等级；L0–L2 使用固定 Dummy/Mask 防止人数泄漏。
- P2：`Full/No-SA/No-Day-Query/Mean-Pool/Sum-Pool` 五个结构；不加入 parameter-matched 变体。
- P1：`DAT/SAA/wSAA/Flat/Set-ETO/LQR/GBQR/DeepQR` 八种方法。
- `P3-L3 = P2-Full = P1-DAT` 必须引用同一个 checkpoint 和同一组逐日预测。
- 执行顺序：数据 → P3 → P2 → H1 人工门槛 → P1 → PF 与条件 Oracle。

## 4. 条件 Oracle 与 Bootstrap

第二个 \(P\) 要求

\[
f^*(Z)=Q_{0.9}(D\mid Z),\qquad Z=(A^-,c_p,c_e,I),
\]

采用每测试日 4,096 条条件 Monte Carlo 路径。预先固定 500 个测试日，再用每状态
16,384 路径检查数值稳定性。Oracle 逐日预测只计算一次，所有方法与 Bootstrap 共用。

正式区间使用 2,000 次同日配对 Bootstrap：每次用同一组测试日索引同时重抽所有方法、
SAA、PF 与 Oracle，并重新计算成本、服务率、成本差和两个比值。它不重训模型、不重跑
DGP、不重算 Oracle；区间只条件于已经拟合的固定模型。

## 5. 大数据执行合同

正式解释器为 `D:\anaconda\envs\attention\python.exe`，GPU 使用 CUDA、FP32。患者数据
采用连续数组加 day offsets 的 CSR/memmap 表示，按患者数分桶并在 batch 内动态 padding。
DataLoader 使用 4 个持久 worker、pinned memory 与异步 GPU 传输。单张 GPU 上各神经模型
顺序训练；不得同时启动多个模型争抢显存。

完整机器可读合同见
`synthetic/configs/confirmed_protocol_v10.yaml`；任何正式运行必须先通过
`axid_synthetic.contract_v10.validate_contract`。

## 6. 分阶段入口与本机预算

正式运行使用：

```powershell
D:\anaconda\envs\attention\python.exe scripts\run_v10_experiment.py --stage data
D:\anaconda\envs\attention\python.exe scripts\run_v10_experiment.py --stage p3
D:\anaconda\envs\attention\python.exe scripts\run_v10_experiment.py --stage p2
# 审阅 H1 后才运行：
D:\anaconda\envs\attention\python.exe scripts\run_v10_experiment.py --stage p1 --approve-p1
D:\anaconda\envs\attention\python.exe scripts\run_v10_experiment.py --stage metrics --oracle-workers 6
```

RTX 5060/8GB、Ryzen 5 9600X 上的中等规模实测为：131,072 个训练日、1,024 次
optimizer step及 2,048 个测试日前向共约 23.1 秒；正式 DGP 生成线性外推约 12 分钟。
据此，正式 P3 和 P2 各约 4–7 分钟，P1 七个新增基准约 3–6 分钟；普通指标、两个比值和
2,000 次 Bootstrap 小于 10 秒，条件 Oracle 及稳定性检查约 9–15 分钟。首次全流程常见
约 35–50 分钟，保守计入 Windows 调度、I/O 和进程启动按 40–60 分钟预留。每个 job
实际耗时写入 `fit_duration_seconds` 和 `duration_seconds`，正式结果以 manifest 为准。

DGP 的分布、映射及参数识别边界见 `02_AXID_DC_v10_DGP.md`。
