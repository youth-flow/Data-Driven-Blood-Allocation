# Set-ETO

固定结构：

- 患者共享 MLP → **有效患者 Mean Pool**（`sum/mask.sum`）；
- 日级特征 MLP；
- 固定 **HurdleGamma** 分布头；
- 直接输出 `tau=0.9` 的条件分位数；
- 不做 Normal/NB 家族选择，不做验证集选优。

独立入口：

```powershell
python ETO/Set-ETO/scripts/run_experiment.py --config ETO/Set-ETO/configs/main.yaml --fold validation --name seteto_check --smoke
```

主 S4 实验从根目录统一入口启动。
