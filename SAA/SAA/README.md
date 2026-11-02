# SAA

本模型的 configs、scripts、src 独立维护，不导入任何其他模型包。共用根目录公共数据合同和评价。

```powershell
python SAA/SAA/scripts/run_experiment.py --data-root synthetic/generated/reference --fold validation --name saa_check --smoke
```

正式运行去掉 `--smoke`；测试时指定 `--fold test --recipe-dir outputs/<development-run>`。
根目录合成总入口统一设定六个 τ；独立运行以本目录自己的配置为准。
权重、逐日预测和汇总指标仅保存在本机忽略目录中。
