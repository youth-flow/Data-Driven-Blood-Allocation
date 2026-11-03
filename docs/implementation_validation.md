# 简化 DAT 与六组实验实现核验（2026-11-03）

理论依据为本机新版PDF的式(6)–(7)、定理10。PDF散列登记在confirmed_protocol.yaml；PDF不提交。
本次仅使用完全人工输入，未读取或修改真实原始表、处理数据，也未修改真实预处理脚本。

## 已完成核验

- 全部102项人工数据测试通过，Ruff检查与格式检查通过。
- 新DAT手工前向方程、恰好三个LN、独立注意力头数、集合置换不变性与参数投影检查通过。
- 理论基准5,801参数，性能基准76,161参数（均为单个tau）。
- S1/S3/S4/C0条件oracle与独立模拟的一阶矩、覆盖率对照通过；oracle不读未来实现标签。
- S2/S4六组端到端smoke完成：70个任务、120行理论指标、1,150行性能指标、12张图。
- 结构容量诊断的有效参数差约0.083%；信息L0–L4分配参数数目一致。
- 同拟合样本的SAA参照P为零；条件运营遗憾非负且不超过对应pinball超额风险。
- 同命令加resume后，1,386个任务文件的集合及修改时间完全不变，确认没有重新训练。
- Windows入口默认MKL顺序计算，解决本机Anaconda/PyTorch在仅恢复汇总时的OpenMP重复加载问题；不使用忽略重复库错误的开关。

复现入口（新目录名；同一源码、配置和环境版本）：

```powershell
python -m pytest -q
python synthetic/scripts/check_protocol.py
python scripts/run_synthetic_experiments.py --family all --scenarios S2 S4 --smoke --name six_group_check --device cpu --threads 1
python scripts/run_synthetic_experiments.py --family all --scenarios S2 S4 --smoke --name six_group_check --device cpu --threads 1 --resume
```

以上是实现检查，神经模型仅训练1轮，不是正式性能结果。正式默认预算与误差解释见synthetic_protocol.md。
数值神经训练未获全局最优证书；只有C0常数偏置子类的样本分位数解标记为精确ERM。
逼近组是独立oracle函数拟合诊断，不给其他实验转移权重；抽样最大误差不冒充全域上确界。

## 本机清理限制

源代码入口已删除旧DAT路线及废弃接口。本机旧build目录删除被自动审批拒绝，返回“blocked by policy”，未给具体原因。
目录因此仍保留，已被Git忽略；脚本显式加载当前源代码，不使用其中的旧副本。没有绕过拒绝再次删除。

原始数据、处理数据、生成数据、结果、权重与PDF均不进入Git；仅提交代码、配置、说明和人工测试。
