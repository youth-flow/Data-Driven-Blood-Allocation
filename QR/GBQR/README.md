# GBQR

本模型的 configs、scripts、src 独立维护，不导入任何其他模型包。共用根目录公共数据合同和评价。
支持 basic / enhanced 两套输入；增强输入只从已知申请计算，类别与标准化只用训练期拟合。各 τ 独立拟合后按固定规则进行非负截断和排序；超参数按验证集所有 τ/target 的平均 P 选择。

```powershell
python QR/GBQR/scripts/run_experiment.py --data-root synthetic/generated/reference --fold validation --name gbqr_check --smoke
```

正式运行去掉 `--smoke`；测试时指定 `--fold test --recipe-dir outputs/<development-run>`。
根目录合成总入口统一设定六个 τ；独立运行以本目录自己的配置为准。
权重、逐日预测和汇总指标仅保存在本机忽略目录中。
