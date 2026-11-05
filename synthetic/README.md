# S4 完全人工 AXID 实验

生成器 v4 与 S4-only 实验入口。全部参数是人工设定，不读取真实原表、派生数据或已有预测模型。
完整生成公式见[02_Synthetic_DGP_S4.md](../数值实验方案/02_Synthetic_DGP_S4.md)，理论组见[03](../数值实验方案/03_Theory_Experiments_S4.md)，性能组见[04](../数值实验方案/04_Performance_Experiments_S4.md)。

## 数据合同

- A：非空、无序、最多 M=8 条的可见患者集合；每条为归一化申请、备血和已知风险。
- X：固定八维 `(c1,c2,I,N/M,sum u/M,sum v/M,risk mean,risk second moment)`，库存位于索引 2。
- I：`315 + 20*c1 + Uniform(-70,70)`；不滚动库存。
- D：患者实际用血 + 新增需求 + `Uniform(0,0.5)` 连续余量，位于 `[0,36.5]`。
- 日级 IID；日期只是 ID；事后 U、Y、L、J 不进入预测输入。
- 性能数据：每次重复 **2,500 训练 + 3,000 测试**，无验证集；理论数据按训练日数在内存生成。

## 入口

```powershell
python synthetic/scripts/check_protocol.py
python scripts/run_synthetic_experiments.py --family theory --smoke --name s4_theory_smoke
python scripts/run_synthetic_experiments.py --family performance --smoke --name s4_perf_smoke
python scripts/run_synthetic_experiments.py --family all --name s4_all
```

单独生成 S4 性能数据：

```powershell
python synthetic/scripts/generate.py --output synthetic/generated/s4 --n-train 2500 --n-test 3000
```

正式协议：10 个独立数据重复；随机方法 3 个固定 seed；无超参数搜索；主 `tau=0.9`；五个 G；最终只有 `MeanCost` 与 `SL_G`。

## 代码

| 组 | 入口参数 | 代码 |
| --- | --- | --- |
| T1 | theory / estimation | `theory.py`（C0 精确 ERM + S4 DAT） |
| T2 | theory / approximation | `theory.py`（S4 Oracle L1 vs P） |
| T3 | theory / consistency | `theory.py`（S4 固定/增长容量） |
| P1 | performance / comparison | `performance.py`（九方法） |
| P2 | performance / ablation | `performance.py`（七 DAT 变体） |
| P3 | performance / information | `performance.py`（L0–L4） |

DGP 在 `src/axid_synthetic/generator.py`，Oracle 在 `oracle.py`。所有生成样本、权重、结果均在 `outputs/` 或 `synthetic/generated/`，Git 忽略。
