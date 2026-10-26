"""Feature engineering, training and evaluation for departure delay models."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (f1_score, mean_absolute_error, roc_auc_score,
                             root_mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL = ["adep", "ades", "aircraft_type", "wake_cat"]
NUMERIC = [
    "sched_hour", "day_of_week", "month", "planned_flight_time_min",
    "demand_capacity_ratio", "wind_kt", "visibility_m", "convective",
    "prev_leg_delay_min", "hour_sin", "hour_cos", "is_peak", "low_vis",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "dep_delay_min"
DELAY_THRESHOLD_MIN = 15  # standard punctuality threshold


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["sched_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["sched_hour"] / 24)
    out["is_peak"] = out["sched_hour"].isin([7, 8, 17, 18]).astype(int)
    out["low_vis"] = (out["visibility_m"] < 1500).astype(int)
    return out


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split: train on the past, test on the future (no leakage)."""
    cutoff = df["date"].quantile(1 - test_frac)
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    n_cat = len(CATEGORICAL)
    estimator.set_params(categorical_features=list(range(n_cat)))
    return Pipeline([("pre", pre), ("model", estimator)])


def build_regressor(seed: int = 0) -> Pipeline:
    return _pipeline(HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        loss="absolute_error", early_stopping=True, random_state=seed))


def build_classifier(seed: int = 0) -> Pipeline:
    return _pipeline(HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        early_stopping=True, random_state=seed))


def build_quantile(alpha: float, seed: int = 0) -> Pipeline:
    """Quantile regressor, used for P10/P90 prediction intervals."""
    return _pipeline(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        loss="quantile", quantile=alpha, early_stopping=True, random_state=seed))


class HistoricalMeanBaseline:
    """Predicts the mean delay seen in training for the same airport and hour."""

    def fit(self, df: pd.DataFrame):
        self.table_ = df.groupby(["adep", "sched_hour"])[TARGET].mean()
        self.global_ = df[TARGET].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[["adep", "sched_hour"]])
        return self.table_.reindex(idx).fillna(self.global_).to_numpy()


@dataclass
class Report:
    metrics: dict = field(default_factory=dict)
    importance: pd.Series | None = None


def train_and_evaluate(df: pd.DataFrame, seed: int = 0) -> tuple[Report, dict]:
    df = add_features(df)
    train, test = time_split(df)
    y_tr, y_te = train[TARGET], test[TARGET]
    yb_tr = (y_tr > DELAY_THRESHOLD_MIN).astype(int)
    yb_te = (y_te > DELAY_THRESHOLD_MIN).astype(int)

    base = HistoricalMeanBaseline().fit(train)
    reg = build_regressor(seed).fit(train[FEATURES], y_tr)
    clf = build_classifier(seed).fit(train[FEATURES], yb_tr)

    q10 = build_quantile(0.1, seed).fit(train[FEATURES], y_tr)
    q90 = build_quantile(0.9, seed).fit(train[FEATURES], y_tr)

    p_base = base.predict(test)
    p_reg = reg.predict(test[FEATURES])
    proba = clf.predict_proba(test[FEATURES])[:, 1]
    lo, hi = q10.predict(test[FEATURES]), q90.predict(test[FEATURES])

    rep = Report()
    rep.metrics = {
        "n_train": len(train), "n_test": len(test),
        "baseline_mae": mean_absolute_error(y_te, p_base),
        "model_mae": mean_absolute_error(y_te, p_reg),
        "model_rmse": root_mean_squared_error(y_te, p_reg),
        "baseline_auc": roc_auc_score(yb_te, p_base),
        "model_auc": roc_auc_score(yb_te, proba),
        "model_f1": f1_score(yb_te, proba > 0.5),
        "delayed_share_test": float(yb_te.mean()),
        # Share of flights whose actual delay falls inside [P10, P90] (target 0.8)
        "interval_coverage": float(((y_te >= lo) & (y_te <= hi)).mean()),
        "interval_width_min": float((hi - lo).mean()),
    }
    rep.metrics["mae_improvement_pct"] = 100 * (
        1 - rep.metrics["model_mae"] / rep.metrics["baseline_mae"])

    sample = test.sample(min(5000, len(test)), random_state=seed)
    pi = permutation_importance(reg, sample[FEATURES], sample[TARGET],
                                scoring="neg_mean_absolute_error",
                                n_repeats=5, random_state=seed)
    rep.importance = pd.Series(pi.importances_mean, index=FEATURES).sort_values(
        ascending=False)

    artefacts = {"regressor": reg, "classifier": clf, "q10": q10, "q90": q90,
                 "train": train, "test": test, "pred": p_reg, "proba": proba,
                 "lo": lo, "hi": hi}
    return rep, artefacts
