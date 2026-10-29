# SAA

本模型独立维护配置、模型、训练和推理，不导入其他四个模型的代码。共用工程根目录的 `data/processed`，以及根目录公共数据与评价工具。

- `configs/main.yaml`：本模型的完整参数与运行配置，不继承其他模型或根实验配置。
- `scripts/run_experiment.py`：独立入口。
- `src/saa_experiment/`：模型实现、`training.py` 和 `inference.py`。

在工程根目录运行：

```sh
python SAA/scripts/run_experiment.py --fold val2019 --name saa_experiment_check --smoke
```

进入本模型目录后，也可运行 `python scripts/run_experiment.py ...`。正式训练移除 `--smoke`。
根目录的 `scripts/run_experiments.py` 可以统一调用本模型；总实验统一覆盖数据路径、输出路径、seed、τ、target，其余参数仍来自本文件夹的配置。
输出保存到根目录 `outputs/<运行名>/SAA`，原始表和派生数据均不进入 Git。
