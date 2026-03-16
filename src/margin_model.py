"""XGBoost point-differential regressor with spline probability calibration.

Trains on signed game margin (Team1_score - Team2_score), then fits a
UnivariateSpline mapping predicted margin to empirical win probability.
Exposes predict_proba so it slots into the existing CV and stack infrastructure
without changes to those modules.

Architecture mirrors the 2025 Kaggle winner (mohammad odeh, Brier 0.10411).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    from scipy.interpolate import UnivariateSpline

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBMarginSplineModel:
    def __init__(self, spline_clip: float = 25.0, num_rounds: int = 704, **xgb_params):
        if not XGB_AVAILABLE:
            raise ImportError(
                "XGBMarginSplineModel requires xgboost and scipy. Install both to enable xgb_margin."
            )

        default_xgb_params = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "eta": 0.0093,
            "subsample": 0.6,
            "colsample_bynode": 0.8,
            "num_parallel_tree": 2,
            "min_child_weight": 4,
            "max_depth": 4,
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "max_bin": 38,
        }
        default_xgb_params.update(xgb_params)

        self.spline_clip = float(spline_clip)
        self.num_rounds = int(num_rounds)
        self.xgb_params = default_xgb_params

    def fit(self, x: pd.DataFrame, y: pd.Series):
        y_margin = pd.Series(y).astype(float).to_numpy()

        self.imputer_ = SimpleImputer(strategy="median")
        x_train = self.imputer_.fit_transform(x)

        dtrain = xgb.DMatrix(x_train, label=y_margin)
        self.xgb_model_ = xgb.train(self.xgb_params, dtrain, num_boost_round=self.num_rounds)

        pred_margin = self.xgb_model_.predict(dtrain)
        pred_clip = np.clip(pred_margin, -self.spline_clip, self.spline_clip)
        win_target = (y_margin > 0.0).astype(float)

        order = np.argsort(pred_clip)
        x_sorted = pred_clip[order]
        y_sorted = win_target[order]

        self.spline_ = UnivariateSpline(x_sorted, y_sorted, k=5, ext=3)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        x_pred = self.imputer_.transform(x)
        dtest = xgb.DMatrix(x_pred)
        pred_margin = self.xgb_model_.predict(dtest)
        pred_clip = np.clip(pred_margin, -self.spline_clip, self.spline_clip)

        prob = np.asarray(self.spline_(pred_clip), dtype=float)
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(x)[:, 1]
