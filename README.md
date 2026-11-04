# DAT numerical study

配置定义参数，脚本提供入口，模型在各自目录实现；根目录统一发动实验。真实数据和完全合成数据分别管理。

## 协作分支与节点提交

当前工作分支为 `yyx`，跟踪远端 `origin/yyx`。后续开发在此分支进行，合并到 `main` 另行安排。
完成一个可验证的重要节点后，先运行对应检查与隐私审查，再自动执行 add、commit、push 到 `yyx`。
节点包括独立功能完成、接口调整完成或实验协议定稿；不逐个临时编辑提交，不强制推送，不提交失败或未完成的节点。
只提交源代码、配置、文档和人工测试。原始数据、处理数据、生成样本、权重和结果始终留在本机。

## 目录

```text
DAT/                         # 主模型、结构消融、信息消融
ETO/
  Set-ETO/                   # Normal / HurdleGamma / NB及连续平滑NB分布族
  Flat-ETO/
SAA/
  SAA/
  wSAA/
  Rolling-SAA/
QR/
  LQR/
  GBQR/
  DeepQR/                    # Deep Sets + 单调多分位数输出，不是普通日级 MLP
Historical-Receipt/          # 已观测历史到货量的直接成本参照
synthetic/
  configs/reference.yaml     # AXID 人工生成参数
  scripts/generate.py
  src/axid_synthetic/
  generated/                 # 本机生成，Git 忽略
configs/                     # 共同数据、τ、target、模型配置位置、实验级覆盖项
scripts/                     # 总实验入口与隐私检查
src/dat_study/               # 数据合同、评价、公共数值工具、调度
tests/                      # 只使用人工数据的接口和隔离测试
data/                       # 真实原表、完整预处理脚本、现行 processed；本次不改
outputs/                    # 本机结果、配方和权重，Git 忽略
```

每个模型有独立 `configs/main.yaml`、`scripts/run_experiment.py`、`src/<package>/`。不会导入其他模型包。
根目录 `src` 不保存模型专属网络；共享特征变换和选择工具允许复用。

## 安装

Python >= 3.10，在工程根目录执行：

```powershell
python -m pip install -e ".[dev,plots]"
git config core.hooksPath .githooks
```

无需安装本工程也可通过源码脚本运行；模型入口按自身位置寻找公共代码，从其他工作目录调用也有效。

## 当前工作：简化 DAT 与纯合成六组实验

按2026-11-03本机理论PDF实现唯一简化DAT：订单MLP、自注意力残差、日级仿射编码、
交叉注意力残差、日级FFN残差、截断标量线性头。没有订单FFN、额外输出LN或三步走。
五个τ各训练独立网络，主结果不排序修正。结构说明见[DAT/README.md](DAT/README.md)。

[四份数值实验方案](数值实验方案/01_DAT_Architecture.md)及生成器v3均已实现：
架构与接口见[01](数值实验方案/01_DAT_Architecture.md)，DGP与Oracle见[02](数值实验方案/02_Synthetic_DGP.md)，
理论组见[03](数值实验方案/03_Theory_Experiments.md)，性能组见[04](数值实验方案/04_Performance_Experiments.md)。
理论三组为估计误差、逼近能力、联合一致性；性能三组为基准比较、结构消融、信息价值。
机器合同在 `synthetic/configs/confirmed_protocol.yaml`。
DGP完全人工，有界、IID、非空集合、固定支持变换；不读取真实数据。

```powershell
python synthetic/scripts/check_protocol.py
python scripts/run_synthetic_experiments.py --family all --smoke --name six_group_smoke --device cpu --threads 1
python scripts/run_synthetic_experiments.py --family theory --name theory_main
python scripts/run_synthetic_experiments.py --family performance --name performance_main
```

- 主τ为0.5、0.6、0.7、0.8、0.9；target为260、280、300、320、340。
- 性能比较包含9种方法；Historical Receipt在合成实验中不伪造。
- 性能配置只用验证pinball选择，在2500日从头重训后评价3000个独立测试场景。
- 理论组分别研究固定类、容量增长、容量与样本联合增长；不把数值神经训练当成已认证精确ERM。
- `--smoke` 只检验流程。`--resume` 验证并跳过完成任务；中断任务从头重跑，保留原attempt。
- 正式默认理论200个独立数据重复、性能10个，各随机模型3个初始化，计算量较大。
- 计划和参数已冻结；改变代码或参数必须换运行名。更多选项和输出见[synthetic/README.md](synthetic/README.md)。

独立模型入口需要先生成合成数据：

```powershell
python synthetic/scripts/generate.py --scenario S2 --output synthetic/generated/reference
```

## 单独运行与 DAT 内部实验

```powershell
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --variant mean_pool --name dat_mean --smoke
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --information L2 --name dat_L2 --smoke
python QR/DeepQR/scripts/run_experiment.py --data-root synthetic/generated/reference --fold validation --name deepqr --smoke
python ETO/Set-ETO/scripts/run_experiment.py --config ETO/Set-ETO/configs/hurdle_gamma.yaml --data-root synthetic/generated/reference --fold validation --name hurdle --smoke
```

DAT 实验干预清单和信息分组见 [DAT/README.md](DAT/README.md)。DeepQR 是独立结构基准；DAT 内部的池化替换才是只改读出部分的消融。

## 真实数据接口保留

真实预处理全部位于 `data/数据预处理.py`，现行数据为 `data/processed`；没有复制、移动或重跑真实数据。
原有时间折 val2017 / val2018 / val2019 / final2020 保留。验证折不能直接作为未参与选择的测试结果。

```powershell
python scripts/run_experiments.py --fold val2019 --methods DAT SAA wSAA Flat-ETO Set-ETO --name comparison --smoke
python scripts/run_experiments.py --fold final2020 --recipe-dir outputs/comparison --name final2020
```

第二条必须指向正式开发运行，不能使用 smoke 配方。自定义派生折可通过 `--fold` 指定；测试角色一律要求此前冻结配方。
原表只允许在用户授权的本机预处理任务中读取；训练模型只读取明确指定的派生数据。

## 决策与统计

统一决策 `q = max(Q_tau + target - I, 0)`；成本为期末库存相对 target 偏差的不对称惩罚。
主指标 `P = 1 - mean_cost / mean_SAA_cost`，SAA 使用相应训练部分的经验分位数。
保存成本、P、pinball、覆盖率、平均配送量、零配送比例；成本分母为零时 P 明确记为未定义。
Historical Receipt 直接评价实际 q，不把它转换成分位数，pinball 和覆盖率为空。
LQR/GBQR 独立分位数使用预先固定的非负截断及排序纠正交叉；拟合、选参和测试使用相同规则。DAT保持独立标量结果，交叉比例直接报告。

## 隐私与测试

```powershell
python -m pytest
python -m ruff check src tests DAT ETO SAA QR Historical-Receipt synthetic scripts data/数据预处理.py
python -m ruff format --check src tests DAT ETO SAA QR Historical-Receipt synthetic scripts data/数据预处理.py
python scripts/check_git_privacy.py --all-tracked
```

测试只用人工数据，涵盖独立包加载、保存重载、测试标签隔离、信息屏蔽和历史配送直接评价。
原始表、派生数据、生成样本、权重、结果、业务资料全部留在本机；`data` 中只有预处理源码允许进入 Git。
Git 白名单和 hooks 同时检查暂存内容与推送的全部历史。合作者须设置 `core.hooksPath`；不得跳过保护或将真实明细贴入源码。
旧清理脚本仍禁用。清理中间结果时不得删除模型目录、原始表或现行数据。
