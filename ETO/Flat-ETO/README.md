# Flat-ETO

固定结构：

- 每名患者使用 `[u_i,v_i,r_i,psi(A,X)]`，共 22 维；
- 七类患者用血 PMF（0, 0.5, …, 3）；
- 19 维 `psi` 上的随机森林拟合连续新增需求 R；
- 患者 PMF 卷积与新增需求加权经验 CDF 合成总量分布，取 `tau=0.9`；
- 训练额外读取授权的患者 `Y_i`，不读取测试标签。

独立入口：

```powershell
python ETO/Flat-ETO/scripts/run_experiment.py --config ETO/Flat-ETO/configs/main.yaml --fold validation --name flat_check --smoke
```

主 S4 实验从根目录统一入口启动。
