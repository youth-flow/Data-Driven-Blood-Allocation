# QR

LQR、GBQR、DeepQR 各自独立实现。DeepQR 为逐条申请编码、求和池化、日级融合和单调分位数输出；不是普通日级 MLP。

统一入口：根目录 `scripts/run_synthetic_experiments.py`。各模型也有自己的 `scripts/run_experiment.py`。
