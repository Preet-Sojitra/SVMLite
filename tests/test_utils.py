"""Tests for utils.py."""

import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from svmlite.utils import StandardScalerLite


class TestStandardScalerLite:
    """Test StandardScaler implementation against sklearn"""

    def test_fit_tranform(self, random_features):
        scaler_lite = StandardScalerLite()
        lite_scaled_data = scaler_lite.fit_transform(random_features)

        sklearn_scaler = SklearnStandardScaler()
        sklearn_scaled_data = sklearn_scaler.fit_transform(random_features)

        # compare the two scaled datasets
        np.testing.assert_allclose(
            lite_scaled_data,
            sklearn_scaled_data,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Scaled data from StandardScalerLite does not match sklearn's StandardScaler",
        )

    def test_mean_and_std(self, random_features):
        scaler_lite = StandardScalerLite()
        scaler_lite.fit(random_features)

        sklearn_scaler = SklearnStandardScaler()
        sklearn_scaler.fit(random_features)

        # compare means
        np.testing.assert_allclose(
            scaler_lite.mean_,
            sklearn_scaler.mean_,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Mean from StandardScalerLite does not match sklearn's StandardScaler",
        )

        # compare stds
        np.testing.assert_allclose(
            scaler_lite.std_,
            sklearn_scaler.scale_,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Std from StandardScalerLite does not match sklearn's StandardScaler",
        )
