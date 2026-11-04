# 完全人工 AXID 与六组实验

生成器 v3 和六组实验入口已实现。全部参数是人工实验设定，不读取真实原表、派生数据或已有预测模型。
完整生成公式、理论对应与统计边界见[数值实验方案](../数值实验方案/02_Synthetic_DGP.md)。
机器合同在 `configs/confirmed_protocol.yaml`，可执行 DGP 参数在 `configs/reference.yaml`。

## 数据合同

- A：非空、无序、最多 M 条的可见订单集合；每条为申请用血量、申请备血量、已知风险。
- X：两个已知环境、期初库存、人数、两种数量总和、风险均值与二阶矩；固定已知支持映射到 [0,1]，共8维。
- I：`315 + 20*c1 + Uniform(-70,70)`，与 target 独立定义；不滚动库存。
- D：可见订单实际贡献 + 当天新增需求 + `Uniform(0,0.5)` 连续余量。有界且不裁剪标签。
- 样本日 IID；日期只是标识，不是输入。订单结果和未来共同冲击不进入特征。
- 理论规模 M=2、B=18.5；性能规模 M=8、B=36.5。共同新增订单上界4。
- S1–S4依次增加情境交互、集合构成作用、未来共同冲击；C0为常数条件分位数的可解校准。

性能样本为2000训练、500验证、3000测试；冻结配方后在2500日重新拟合。3000是独立单期场景数。
相同 data seed 保证不同机制场景的 A/X/I 配对；扩大训练量保留原样本并固定验证和测试。理论组使用嵌套训练前缀，不合并验证样本。
导出合同为 `synthetic_axid_v2`（生成器版本3），旧人工 schema 不混用。

## 入口

根目录入口自动生成自己的数据，无需先单独生成：

```powershell
python synthetic/scripts/check_protocol.py
python scripts/run_synthetic_experiments.py --family all --smoke --name six_group_smoke --device cpu --threads 1
python scripts/run_synthetic_experiments.py --family theory --name theory_main
python scripts/run_synthetic_experiments.py --family performance --name performance_main
```

正式默认理论200个独立数据重复、性能10个，神经模型各3个初始化，计算量较大。检查计划用 `--dry-run`；先行耗时试运行可指定 `--repetitions 1`。缩小重复数的结果不得冒充完整协议结果。
单组运行用 `--group estimation|approximation|consistency|comparison|ablation|information`。
`--scenarios S2 S4`、`--methods DAT Set-ETO SAA` 可限制检查范围；methods只作用于性能组。
`--train-sizes` 只作用于理论组；`--eval-days` 改变评价样本量，且写入冻结计划。

同一命令加 `--resume` 可继续。计划、源码指纹、配置和已完成文件散列必须一致。
计划同时记录Python、数值库版本及MKL设置；Windows入口默认MKL顺序计算，避免与PyTorch重复加载OpenMP。
已完成任务跳过；中断任务从头重跑，原 attempt 保留；不支持 epoch 内恢复。代码或参数改变时换新实验名，不混合结果。

单独生成人工数据：

```powershell
python synthetic/scripts/generate.py --scenario S2 --output synthetic/generated/reference
```

## 六组代码

| 组 | 入口参数 | 代码 | 核心输出 |
| --- | --- | --- | --- |
| 估计误差 | theory / estimation | theory.py | 固定类风险尾部分布、合法界、C0精确偏置子类校准 |
| 逼近能力 | theory / approximation | theory.py | oracle L1/L2、抽样最大误差、可达到超额风险 |
| 联合一致性 | theory / consistency | theory.py | 固定/增长容量的超额风险、运营遗憾、复杂度趋势 |
| 基准比较 | performance / comparison | performance.py | 9种方法；LQR/GBQR各有basic/enhanced |
| 结构消融 | performance / ablation | performance.py | full和4项干预，另有2项有效容量匹配诊断 |
| 信息价值 | performance / information | performance.py | 固定 DAT 的 L0–L4，分别重训 |

这里只保留简化标量 DAT。所谓三步走、LoRA、订单预训练和多分位数共享输出不在流程中。
逼近诊断的 oracle 拟合模型不转移到性能实验。

## Oracle 与统计

`oracle.py` 对有限订单分布卷积；有共同冲击时进行一维自适应正态积分。分位数由连续单元内插值得到，条件 pinball 风险解析积分。
共同冲击积分记录误差估计和截断尾质量；估计不是严格数值证书。总体风险仍由独立输入平均估计，理论组记录 MCSE。
`f_star` 是完整信息 Bayes oracle，不是类内最优 DAT，也不知当天的未来实现值。
一般场景的风险曲线不能直接叫精确估计误差；精确 ERM 只在 C0 的常数偏置子类中认证。
低信息级别用完整信息 oracle 只作为共同参照，不声称是该级别专属 Bayes 最优值。

性能主指标为平均成本 `AC` 与目标服务率 `SL_G`，统一 `q=max(Q_tau+target-I,0)`；`P = 1 - AC/AC_SAA` 仅作内部诊断。共同测试日、共同 SAA 参照；不得以测试胜负修改机制。
同一数据重复内先平均模型种子，再报告跨独立数据重复的均值、t区间和配对差。
DAT的跨分位交叉照实报告，不通过排序改变主结果。

## 本机输出

`outputs/<name>/` 包含冻结计划、人工数据、逐任务配方/权重、条件分布缓存、长表、汇总及图。
主要文件为 `theory_metrics.csv`、`theory_summary.csv`、`performance_metrics.csv`、
`performance_summary.csv`、`paired_P_differences.csv`、`figures/` 和 `study_complete.json`。
PNG依赖可选安装 `pip install -e ".[plots]"`；无matplotlib时仍输出CSV并明确记录缺少图。

所有生成样本、权重、结果均Git忽略。Git只保存代码、配置、协议和人工测试；不接触真实数据。
`--smoke` 将神经训练减为1轮并进一步限制样本，用于接口核验，不能报告为正式性能。
