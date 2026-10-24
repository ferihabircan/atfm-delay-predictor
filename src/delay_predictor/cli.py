"""Command line entry point: python -m delay_predictor.cli [--data flights.csv]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import generate_flights, load_flights  # noqa: E402
from .model import train_and_evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="CSV/Parquet in the documented schema. "
                                   "If omitted, synthetic data is generated.")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    df = load_flights(args.data) if args.data else generate_flights(n_days=args.days)
    print(f"Loaded {len(df):,} flights")
    rep, art = train_and_evaluate(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rep.metrics, indent=2))
    joblib.dump(art["regressor"], out / "regressor.joblib")

    fig, ax = plt.subplots(figsize=(7, 5))
    rep.importance.head(10)[::-1].plot.barh(ax=ax, color="#1f5aa6")
    ax.set_title("Permutation importance (MAE increase, minutes)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=130)

    test = art["test"].assign(pred=art["pred"])
    by_hour = test.groupby("sched_hour")[["dep_delay_min", "pred"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    by_hour.plot(ax=ax, marker="o")
    ax.set_xlabel("Scheduled hour")
    ax.set_ylabel("Mean departure delay (min)")
    ax.legend(["Observed", "Predicted"])
    ax.set_title("Observed vs predicted delay by hour (test period)")
    fig.tight_layout()
    fig.savefig(out / "delay_by_hour.png", dpi=130)

    for k, v in rep.metrics.items():
        print(f"{k:>22}: {v:.3f}" if isinstance(v, float) else f"{k:>22}: {v}")


if __name__ == "__main__":
    main()
