# DAT numerical study

五个模型互相独立，共用根目录的 `data`。运行关系是 **configs 定义参数 → scripts 作为入口 → src 提供实现**。

## 目录职责

| 目录 | 职责 |
| --- | --- |
| `configs/` | 总实验配置：数据路径、输出路径、随机种子、τ、target，以及五份模型配置的位置。模型超参数不放在这里。 |
| `scripts/` | 总实验入口 `run_experiments.py`，以及 Git 隐私检查工具。 |
| `src/dat_study/` | 公共训练数据读取、决策公式、P 评价、数值工具、运行记录和按需调度；不存放预处理实现、五个模型的网络或专属训练流程。 |
| `tests/` | 公共接口与跨模型回归测试。只用人工数据，不参与正式训练；删除后训练仍可运行，但会失去自动检查。 |
| `DAT/`、`SAA/`、`wSAA/`、`Flat-ETO/`、`Set-ETO/` | 各自的 `configs`、`scripts`、`src`。各自维护参数、模型、训练和推理，互不导入其他模型代码。 |
| `data/` | 两张本机原始表、包含完整实现的 `数据预处理.py`、现行派生数据 `processed/`。 |
| `outputs/` | 运行时生成的中间产物、模型权重和结果；不进入 Git。 |

```text
data/
  申请表明细.csv
  库存明细.csv
  数据预处理.py
  processed/
    val2017/
    val2018/
    val2019/
    final2020/
DAT/                  # 其余四个模型使用同样的组织方式
  configs/main.yaml
  scripts/run_experiment.py
  src/datransformer/
configs/experiment.yaml
scripts/run_experiments.py
src/dat_study/
tests/
```

`processed` 只是通用目录名，内部的数据版本标识与完整性校验保留。本次重命名没有重新生成或改变训练数据。
理论和文献保留在原目录，默认不进入代码仓库。

## 安装

Python >= 3.10。合作者各自建立环境，在工程根目录执行：

```sh
python -m pip install -e ".[dev]"
git config core.hooksPath .githooks
```

独立模型入口也支持直接从源码运行，只需要已安装依赖、根目录公共 `src`、共享 `data` 和该模型自己的目录。

## 从根目录统一运行

```sh
python scripts/run_experiments.py --fold val2019 --name comparison --smoke
```

默认调用五个模型。`--smoke` 只做小规模接口检查，正式实验时移除。也可明确选择：

```sh
python scripts/run_experiments.py --fold val2019 --methods DAT SAA wSAA Flat-ETO Set-ETO --name comparison_full
python scripts/run_experiments.py --fold val2019 --methods DAT-3stage --name dat_three_stage
```

总入口读取 `configs/experiment.yaml`，再读取各模型自己的 `configs/main.yaml`。
共同的数据路径、seed、τ、target 由根配置统一指定；其余参数使用各模型自己的配置。
只选择某个模型时不会加载其他模型代码或配置。

## 单独运行一个模型

```sh
python DAT/scripts/run_experiment.py --fold val2019 --name dat_only --smoke
python DAT/scripts/run_experiment.py --fold val2019 --name dat_three_stage_only --three-stage --smoke
python SAA/scripts/run_experiment.py --fold val2019 --name saa_only --smoke
python wSAA/scripts/run_experiment.py --fold val2019 --name wsaa_only --smoke
python Flat-ETO/scripts/run_experiment.py --fold val2019 --name flat_only --smoke
python Set-ETO/scripts/run_experiment.py --fold val2019 --name set_only --smoke
```

也可以进入模型文件夹后运行 `python scripts/run_experiment.py ...`。入口按自身位置定位配置和根目录数据，不依赖当前工作目录。
单独运行仅使用该模型自己的配置；修改它不会改变其他模型的配置。

## 数据与评价口径

默认训练数据为 `data/processed`，所有模型读取相同的四个时间折。原始表只由预处理读取，不打印明细、不上传、不改写。

```sh
python data/数据预处理.py --help
python data/数据预处理.py --output data/processed_new
```

不指定 `--output` 时输出到 `data/processed`。已有目录拒绝覆盖；当前数据已存在，不需要为了运行实验重新预处理。
全部预处理实现均位于 `data/数据预处理.py`，包括清洗、A/X/I/D 构造、训练期特征拟合、时间划分、隐私处理与导出。
脚本仅依赖标准库、NumPy 和 pandas，可单独运行；默认路径按脚本位置定位，不依赖当前工作目录。
`src` 中不再保留分散的预处理实现；根命令行的 `preprocess` 子命令只转调这一个脚本。

所有模型统一决策 `q = max(Q_tau + G - I, 0)`，主评价为相对于训练期 SAA 的 `P = 1 - mean_cost / mean_SAA_cost`。
τ 默认 0.5、0.6、0.7、0.8、0.9；target 默认 260、280、300、320、340。
模型选择只使用验证期。最后年份重训必须提供此前验证得到的冻结配方：

```sh
python scripts/run_experiments.py --fold final2020 --recipe-dir outputs/comparison_full --name final2020
```

每个运行写入独立目录；每个模型保存自己的实际配置、训练记录、预测器和逐情景指标，总入口输出 `comparison.csv`。
已有运行目录拒绝覆盖。旧实验结果已清除，需要重新产生验证配方后再做最终年份重训。

## 测试与隐私

```sh
python -m pytest
python -m ruff check src tests DAT SAA wSAA Flat-ETO Set-ETO scripts data/数据预处理.py
python -m ruff format --check src tests DAT SAA wSAA Flat-ETO Set-ETO scripts data/数据预处理.py
```

测试不需要真实数据；包括屏蔽其余模型后独立训练/重载、公共数据接口、P 指标、最终年份标签隔离和 Git 隐私边界。
原始表、派生数据、权重、结果及业务资料不得提交 Git；即使脱敏也不自动视为可公开资料。
Git 只管理代码、配置、文档和人工测试。原始表及 `data/processed` 全部忽略，`data` 中仅允许提交 `数据预处理.py`。
提交前检查整个暂存区；推送前检查待推送提交的全部历史，避免已经删除的数据通过旧提交进入远程。
源代码白名单同时排除权重、结果、缓存、业务资料、符号链接和子模块，并检测常见凭据格式。
这些检查不能替代人工检查；不要跳过 hook，也不要把真实记录粘贴到源码、文档或提交消息中。
合作者克隆后须执行前述 `git config core.hooksPath .githooks`，使提交和推送保护生效。

清理只针对实验中间数据、缓存、权重和结果，不能删除模型目录、源码、配置、原表或现行训练数据。
旧清理脚本已禁用并排除在代码仓库之外，不再使用其删除方案。
