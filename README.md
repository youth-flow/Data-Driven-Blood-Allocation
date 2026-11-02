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
  Set-ETO/                   # Normal / NegativeBinomial / HurdleGamma 内部分布族
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
python -m pip install -e ".[dev]"
git config core.hooksPath .githooks
```

无需安装本工程也可通过源码脚本运行；模型入口按自身位置寻找公共代码，从其他工作目录调用也有效。

## 当前工作：纯合成实验

AXID 完整公式、参数及边界见 [synthetic/README.md](synthetic/README.md)。不读取真实表，不从真实模型生成标签。

```powershell
python synthetic/scripts/generate.py --scenario S2 --output synthetic/generated/reference
python scripts/run_synthetic_experiments.py --group all --name synthetic_dev --seeds 0 1 2 3 4
python scripts/run_synthetic_experiments.py --group all --fold test --name synthetic_test --seeds 0 1 2 3 4 --recipe-root outputs/synthetic_dev
```

- A：独立基准模型；LQR/GBQR 分别运行 basic 与 enhanced 输入。
- B：DAT 内部结构/损失消融。
- C：固定宽度 DAT 的 L0–L4 信息阶梯，每个等级重新训练。
- 合成共同配置在 `configs/synthetic.yaml`，τ 为 0.5、0.6、0.7、0.8、0.9、0.95；target 为 260、280、300、320、340。
- DAT 仅使用日级样本直接端到端训练所有参数；无订单级预训练、微调、冻结或低秩适配。
- Historical Receipt 没有真实合成对应物，纯合成总入口不运行它。
- `--smoke` 仅验证接口。`--resume` 可跳过校验通过的已完成子运行，不是轮内训练恢复。
- 默认 seed 0，正式多 seed 运行需显式给 `--seeds`。当前入口保存逐 seed 指标与预测，未自动生成论文用 ensemble/bootstrap 报表。

快速检查使用独立目录：

```powershell
python synthetic/scripts/generate.py --smoke --output synthetic/generated/check
python scripts/run_synthetic_experiments.py --group all --data-root synthetic/generated/check --name synthetic_check --smoke
```

## 单独运行与 DAT 内部实验

```powershell
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --variant mean_pool --name dat_mean --smoke
python DAT/scripts/run_experiment.py --config DAT/configs/synthetic.yaml --fold validation --information L2 --name dat_L2 --smoke
python QR/DeepQR/scripts/run_experiment.py --data-root synthetic/generated/reference --fold validation --name deepqr --smoke
python ETO/Set-ETO/scripts/run_experiment.py --config ETO/Set-ETO/configs/hurdle_gamma.yaml --data-root synthetic/generated/reference --fold validation --name hurdle --smoke
```

DAT 变体清单和信息分组见 [DAT/README.md](DAT/README.md)。DeepQR 是独立结构基准；DAT 内部的池化替换才是只改读出部分的消融。

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
LQR/GBQR 独立分位数使用预先固定的非负截断及排序纠正交叉；拟合、选参和测试使用相同规则。

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
