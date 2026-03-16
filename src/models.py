"""Model factories and prediction helpers for NCAA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CalibratedIsotonicWrapper:
    """Fits base classifier then isotonic calibration on train predictions."""

    base_model: object

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CalibratedIsotonicWrapper":
        self.base_model.fit(x, y)
        p_train = self._predict_raw(x)
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso_.fit(p_train, y)
        return self

    def _predict_raw(self, x: pd.DataFrame) -> np.ndarray:
        if hasattr(self.base_model, "predict_proba"):
            return np.asarray(self.base_model.predict_proba(x)[:, 1], dtype=float)
        return np.asarray(self.base_model.predict(x), dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = self._predict_raw(x)
        cal = np.clip(self.iso_.predict(raw), 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - cal, cal])


def build_logreg_model() -> object:
    """Baseline logistic regression model."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=5000, random_state=42)),
        ]
    )


def build_hgb_isotonic_model() -> object:
    """Baseline HistGradientBoosting with isotonic calibration."""
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_depth=5,
                    max_iter=700,
                    min_samples_leaf=25,
                    random_state=42,
                ),
            ),
        ]
    )
    return CalibratedIsotonicWrapper(base_model=base)


def build_base_model_factories(*, include_extras: bool = False) -> Dict[str, Callable[[], object]]:
    """Phase-4 base model factories.

    By default keeps a lightweight set for faster rolling CV on HPC.
    Extra models can be enabled when desired.
    """
    factories: Dict[str, Callable[[], object]] = {
        "logreg": build_logreg_model,
        "hgb": build_hgb_isotonic_model,
    }

    if not include_extras:
        return factories

    factories["mlp"] = lambda: Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=500,
                    alpha=1e-4,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=15,
                ),
            ),
        ]
    )

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        factories["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier  # type: ignore

        factories["catboost"] = lambda: CatBoostClassifier(
            depth=6,
            learning_rate=0.03,
            iterations=500,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier  # type: ignore

        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass

    try:
        from margin_model import XGBMarginSplineModel, XGB_AVAILABLE

        if XGB_AVAILABLE:
            from scipy.interpolate import UnivariateSpline  # noqa: F401

            factories["xgb_margin_m"] = lambda: XGBMarginSplineModel(gender="m")
            factories["xgb_margin_w"] = lambda: XGBMarginSplineModel(gender="w")
    except Exception:
        pass

    return factories
