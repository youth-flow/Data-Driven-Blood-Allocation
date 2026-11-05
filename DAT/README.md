# DAT

唯一主模型实现在 `src/datransformer/model.py`：共享患者编码器、单层患者 Self-Attention 残差、日级仿射编码、单层日级 Cross-Attention 残差、日级 FFN 残差、截断标量线性头。

- 三个 LayerNorm，分别位于 SA 前、CA 的 Query 与 Memory；
- 无患者 FFN、无 Final LayerNorm、无位置编码、无 Dropout；
- 每个 `tau` 独立训练一个标量网络；主协议 `tau=0.9`；
- 结构消融：`no_self_attention`、`mean_pool`、`sum_pool`、`no_day_query`；
- 信息级别：L0–L4；L0–L2 使用固定 Dummy Token 防人数泄漏，L3 屏蔽逐患者风险；
- 参数球投影在每次更新后执行。

独立入口：

```powershell
python DAT/scripts/run_experiment.py --config DAT/configs/main.yaml --fold validation --name dat_check --smoke
```

正式 S4 实验统一从根目录 `scripts/run_synthetic_experiments.py` 启动。
