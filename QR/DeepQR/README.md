# DeepQR

本模型的 configs、scripts、src 独立维护，不导入任何其他模型包。共用根目录公共数据合同和评价。
结构沿用 Deep Sets：订单 MLP、求和池化、日级 MLP 与融合头；输出为 softplus 正增量的累积和，和 DAT 使用相同形式的单调分位数变换与 softplus-pinball 损失。它不导入 DAT 或 Set-ETO，且不声称与 DAT 参数量完全相同。

```powershell
python QR/DeepQR/scripts/run_experiment.py --data-root synthetic/generated/reference --fold validation --name deepqr_check --smoke
```

正式运行去掉 `--smoke`；测试时指定 `--fold test --recipe-dir outputs/<development-run>`。
根目录合成总入口统一设定六个 τ；独立运行以本目录自己的配置为准。
权重、逐日预测和汇总指标仅保存在本机忽略目录中。
