import numpy as np
import torch
from sklearn.linear_model import QuantileRegressor

from dat_study.runtime import prepare_fit, device_for, seed_everything
from dat_study.tabular import PsiFeatures, ordered_quantiles


def fit_predict(data, config, recipe=None):
    prepare_fit(data, config, recipe)
    features = PsiFeatures().fit(data)
    train, evaluation = data.daily_split("train"), data.daily_split("eval")
    x, xe = features.transform(data, train), features.transform(data, evaluation)
    alpha = float(config["lqr"]["alpha"])
    if int(data.schema.get("generator_version", 0)) == 10:
        seed_everything(config["seed"])
        device = device_for(config)
        tx = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        ty = torch.from_numpy(train.D.to_numpy(np.float32)).to(device)
        te = torch.from_numpy(np.asarray(xe, dtype=np.float32)).to(device)
        predictions, diagnostics, iterations = [], [], int(config["lqr"].get("max_iter_v10", 120))
        for tau in config["taus"]:
            model = torch.nn.Linear(tx.shape[1], 1).to(device)
            torch.nn.init.zeros_(model.weight)
            torch.nn.init.constant_(model.bias, float(np.quantile(train.D, tau, method="higher")))
            optimizer = torch.optim.LBFGS(
                model.parameters(), lr=0.8, max_iter=iterations, history_size=10, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                residual = ty - model(tx).squeeze(-1)
                loss = torch.maximum(tau * residual, (tau - 1.0) * residual).mean()
                loss = loss + alpha * model.weight.abs().sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            final_objective = float(closure().detach().cpu())
            gradient_inf = max(float(parameter.grad.detach().abs().max().cpu()) for parameter in model.parameters())
            state = optimizer.state[next(iter(model.parameters()))]
            diagnostics.append(
                {
                    "tau": float(tau),
                    "objective": final_objective,
                    "gradient_inf": gradient_inf,
                    "iterations": int(state.get("n_iter", 0)),
                    "function_evaluations": int(state.get("func_evals", 0)),
                }
            )
            with torch.no_grad():
                predictions.append(model(te).squeeze(-1).cpu().numpy())
        prediction = ordered_quantiles(np.column_stack(predictions))
        return (
            prediction,
            {"epochs": 0, "alpha": alpha, "solver": "torch_lbfgs", "max_iter": iterations, "diagnostics": diagnostics},
            {},
        )
    models = [QuantileRegressor(quantile=tau, alpha=alpha, solver="highs").fit(x, train.D) for tau in config["taus"]]
    prediction = ordered_quantiles(np.column_stack([model.predict(xe) for model in models]))
    return prediction, {"epochs": 0, "alpha": alpha}, {"features": features, "models": models}
