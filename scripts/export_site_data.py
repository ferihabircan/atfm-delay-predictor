"""Train the models and export everything the project website needs to
docs/data.json (metrics, charts and a precomputed what-if scenario grid).

    PYTHONPATH=src python scripts/export_site_data.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from delay_predictor.data import AIRPORTS, generate_flights
from delay_predictor.model import FEATURES, add_features, train_and_evaluate

OUT = Path(__file__).resolve().parents[1] / "docs" / "data.json"
DCR_LEVELS = [0.6, 0.9, 1.1, 1.3, 1.6]


def scenario_grid(art: dict) -> list[dict]:
    """Model predictions for airport x hour x weather x demand/capacity,
    holding the other features at typical training values."""
    train = art["train"]
    typical = {
        "aircraft_type": "A320", "wake_cat": "M", "day_of_week": 2, "month": 7,
        "planned_flight_time_min": 130.0,
        "wind_kt": float(train["wind_kt"].median()),
        "visibility_m": float(train["visibility_m"].median()),
    }
    reac = train.groupby("sched_hour")["prev_leg_delay_min"].mean()
    rows = []
    for ap, hour, conv, dcr in itertools.product(AIRPORTS, range(24), (0, 1), DCR_LEVELS):
        ades = "EHAM" if ap != "EHAM" else "LFPG"
        rows.append(dict(typical, adep=ap, ades=ades, sched_hour=hour, convective=conv,
                         demand_capacity_ratio=dcr,
                         prev_leg_delay_min=float(reac.get(hour, 0.0))))
    grid = add_features(pd.DataFrame(rows))
    X = grid[FEATURES]
    grid["p50"] = art["regressor"].predict(X)
    grid["p10"] = art["q10"].predict(X)
    grid["p90"] = art["q90"].predict(X)
    grid["prob15"] = art["classifier"].predict_proba(X)[:, 1]
    cols = ["adep", "sched_hour", "convective", "demand_capacity_ratio",
            "p10", "p50", "p90", "prob15"]
    return grid[cols].round(3).to_dict(orient="records")


def main() -> None:
    df = generate_flights()
    rep, art = train_and_evaluate(df)
    test = art["test"].assign(pred=art["pred"], lo=art["lo"], hi=art["hi"],
                              proba=art["proba"])

    by_hour = test.groupby("sched_hour").agg(
        observed=("dep_delay_min", "mean"), predicted=("pred", "mean"),
        p10=("lo", "mean"), p90=("hi", "mean"), flights=("pred", "size")).reset_index()

    bins = np.linspace(0, 1, 11)
    test["bin"] = pd.cut(test["proba"], bins, include_lowest=True)
    calib = test.groupby("bin", observed=True).agg(
        predicted=("proba", "mean"),
        observed=("dep_delay_min", lambda s: float((s > 15).mean())),
        n=("proba", "size")).reset_index(drop=True)

    by_airport = test.groupby("adep").agg(
        observed=("dep_delay_min", "mean"), predicted=("pred", "mean"),
        delayed_share=("dep_delay_min", lambda s: float((s > 15).mean()))).reset_index()

    payload = {
        "metrics": {k: round(float(v), 4) for k, v in rep.metrics.items()},
        "importance": rep.importance.round(4).to_dict(),
        "by_hour": by_hour.round(3).to_dict(orient="records"),
        "calibration": calib.round(4).to_dict(orient="records"),
        "by_airport": by_airport.round(3).to_dict(orient="records"),
        "airports": list(AIRPORTS),
        "dcr_levels": DCR_LEVELS,
        "scenarios": scenario_grid(art),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print({k: payload["metrics"][k] for k in
           ("model_mae", "baseline_mae", "model_auc", "interval_coverage",
            "interval_width_min")})


if __name__ == "__main__":
    main()
