# Historical Receipt

直接以派生数据的 `q_hist_receipt` 作为配送量，评价 `I + q_hist_receipt - D - target` 的不对称成本。
不训练需求预测器，不将历史配送量当成分位数，不改变历史配送量以适应不同 τ 或 target。
输出成本、相对 SAA 的 P、平均配送量和零配送比例；pinball 与需求覆盖率不适用，记为空值。

纯合成数据没有真实历史配送记录，因此合成总入口不运行此参照；单独运行也会明确拒绝合成数据。
如将来增加模拟传统策略，必须单独命名，不能把模拟行为称为医院真实历史做法。

```powershell
python Historical-Receipt/scripts/run_experiment.py --fold val2019 --name historical_receipt
```

只读取明确提供的派生数据，不打开原始表。缺失或非法历史配送量会报错，不补零、不静默删日。
