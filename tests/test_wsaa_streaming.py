"""Large-scale wSAA prediction must match the historical dense definition."""

import numpy as np

from wsaa_experiment.estimators import RandomForestWeights, predict_weighted_quantiles


def test_streaming_forest_quantiles_match_dense_weights():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(120, 4))
    y = rng.normal(size=120)
    query = rng.normal(size=(17, 4))
    estimator = RandomForestWeights(
        n_estimators=8,
        min_samples_leaf=5,
        max_features=1.0,
        seed=21,
        n_jobs=1,
        max_samples=0.8,
    ).fit(x, y)
    dense = predict_weighted_quantiles(estimator.weights(query), y, [0.5, 0.9])
    streaming = estimator.predict_quantiles(query, y, [0.5, 0.9], n_jobs=1)
    np.testing.assert_allclose(streaming, dense, rtol=0.0, atol=0.0)
