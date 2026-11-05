# QR / LQR / GBQR / DeepQR

- **LQR**：19 维 `psi(A,X)` 上的线性分位数回归，固定 `alpha`。
- **GBQR**：19 维 `psi(A,X)` 上的固定梯度提升分位数回归。
- **DeepQR**：患者编码 + **Mean Pool** + 日级编码 → 标量分位数，直接 pinball 训练；不是普通日级 MLP。

`psi` 固定为 19 维：8 维 `X` + `u²,v²,uv,ur,vr,min/max(u),min/max(v),min/max(r)`。三者在主协议中均为固定配置，不做网格搜索。
