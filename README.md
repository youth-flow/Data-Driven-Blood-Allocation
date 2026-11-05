# DAT S4 numerical study

配置定义参数，脚本提供入口，模型在各自目录实现；根目录统一发动实验。正式 Synthetic 协议为 **S4-only**，主分位数为 `tau=0.9`，最终性能指标只有 **MeanCost** 与 **SL_G**。

## 目录

```text
DAT/                         # 唯一主模型 DAT（含结构/信息干预）
ETO/Set-ETO/                 # DeepSets + 固定 HurdleGamma
ETO/Flat-ETO/                # 患者七类 PMF + 连续新增需求森林
SAA/SAA/                     # 经验分位数
SAA/wSAA/                    # 随机森林叶权重
SAA/Rolling-SAA/             # 训练集末段固定窗口
QR/LQR/                      # 线性分位数回归
QR/GBQR/                     # 梯度提升分位数回归
QR/DeepQR/                   # DeepSets 直接分位数
synthetic/
  configs/confirmed_protocol.yaml   # S4 协议（2500 train / 3000 test, 10x3）
  configs/reference.yaml            # 生成器参数
  src/axid_synthetic/               # DGP、Oracle、T1-T3、P1-P3、report
  generated/                        # 本机生成数据，Git 忽略
configs/                     # 公共 τ、target、模型配置位置
scripts/                     # 根入口
src/dat_study/               # 数据合同、评价、运行底座
tests/                       # 人工数据测试
outputs/                     # 本机结果，Git 忽略
```

## 协议要点

- DGP：仅 **S4**，`(λ_ctx, λ_peer, ρ)=(1,1.5,0.8)`；`M=8, B=36.5`；C0 仅用于 T1 精确 ERM 校准。
- 主 τ：`0.9`；辅助 τ 必须单独标记、单独拟合。
- 性能数据：每次独立重复生成 **2,500 训练日 + 3,000 测试日**，不再有验证集；所有方法共用同一训练/测试日。
- 统计：**10 个独立数据重复**；随机方法每份数据 **3 个固定训练 seed**；确定性方法只保留一个真实结果。
- **无超参数搜索**：所有结构/训练/算法配置在运行前冻结、随结果归档。
- 指标：`MeanCost = (1/3000)Σ c_t`，`SL_G = mean(I_end ≥ G)`；只有这两个数值列。
- P1：9 个固定方法；P2：5 个主结构变体 + 2 个参数匹配辅助变体；P3：L0–L4。
- T1：C0 200 重复精确 ERM + S4 完整 DAT 学习曲线；T2：S4 Oracle L1 随 P；T3：S4 固定/增长容量轨道。

## 安装

```powershell
python -m pip install -e ".[dev,plots]"
git config core.hooksPath .githooks
```

## 运行

```powershell
python synthetic/scripts/check_protocol.py
python scripts/run_synthetic_experiments.py --family theory --smoke --name s4_theory_smoke
python scripts/run_synthetic_experiments.py --family performance --smoke --name s4_perf_smoke
python scripts/run_synthetic_experiments.py --family theory --name theory_s4_main
python scripts/run_synthetic_experiments.py --family performance --name performance_s4_main
python scripts/run_synthetic_experiments.py --family all --name s4_all
```

`--resume` 校验计划、配置和已完成文件散列；配置改变必须换新运行名。`--smoke` 只检验流程，不能报告为正式结果。

## 输出

`outputs/<name>/` 保存冻结计划、S4 数据、逐任务配方/结果、长表、汇总、图和 `study_complete.json`。
主要文件：

- `performance_metrics.csv`、`performance_summary.csv`
- `paired_MeanCost_SL_differences.csv`、`paired_MeanCost_SL_summary.csv`
- `theory_metrics.csv`、`T1_C0_summary.csv`、`T1_DAT_summary.csv`、`T2_summary.csv`、`T3_summary.csv`
- `figures/`

生成数据、权重、结果全部 Git 忽略；只提交代码、配置、协议和人工测试。

## 协作与提交

当前工作分支为 `yyx`，跟踪 `origin/yyx`。完成一个可验证节点后，先运行测试、ruff 和隐私检查，再 add/commit/push。
不逐临时编辑提交，不提交失败或未完成节点；原始表、processed 数据、生成样本、权重和结果始终留在本机。

## 测试

```powershell
python -m pytest
python -m ruff check src tests DAT ETO SAA QR synthetic scripts
python -m ruff format --check src tests DAT ETO SAA QR synthetic scripts
python scripts/check_git_privacy.py --all-tracked
```

`data/` 只保留原有真实数据预处理代码，不属于本 S4 实验；S4 实验不读取任何真实表。
