"""Flight-level dataset generation and loading.

The generator produces a synthetic but structurally realistic set of European
departures. It is meant for development and testing; the same schema can be
filled from real sources such as the EUROCONTROL Aviation Data Repository for
Research (https://www.eurocontrol.int/dashboard/rnd-data-archive) or OpenSky.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ICAO code -> (declared hourly departure capacity, base congestion factor)
AIRPORTS = {
    "EGLL": (45, 1.30),  # London Heathrow
    "LFPG": (52, 1.15),  # Paris CDG
    "EHAM": (55, 1.10),  # Amsterdam Schiphol
    "EDDF": (50, 1.15),  # Frankfurt
    "LEMD": (48, 0.95),  # Madrid
    "LTFM": (60, 1.00),  # Istanbul
    "EBBR": (36, 0.90),  # Brussels
    "LIRF": (40, 0.90),  # Rome Fiumicino
}

AIRCRAFT = {  # type -> wake category
    "A320": "M", "B738": "M", "A321": "M", "E190": "M",
    "A333": "H", "B77W": "H", "A359": "H", "AT76": "L",
}

# Share of daily traffic per local hour (morning and evening banks).
HOURLY_PROFILE = np.array([
    0.2, 0.1, 0.1, 0.1, 0.3, 1.0, 2.6, 3.0, 2.8, 2.4, 2.2, 2.1,
    2.0, 2.1, 2.3, 2.4, 2.6, 2.9, 3.0, 2.7, 2.2, 1.6, 0.9, 0.4,
])

SCHEMA = [
    "flight_id", "date", "adep", "ades", "aircraft_type", "wake_cat",
    "sched_hour", "day_of_week", "month", "planned_flight_time_min",
    "demand_capacity_ratio", "wind_kt", "visibility_m", "convective",
    "prev_leg_delay_min", "dep_delay_min",
]


def generate_flights(n_days: int = 120, flights_per_day: int = 1500,
                     seed: int = 42) -> pd.DataFrame:
    """Generate synthetic departures with plausible delay drivers."""
    rng = np.random.default_rng(seed)
    airports = list(AIRPORTS)
    types = list(AIRCRAFT)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    hour_p = HOURLY_PROFILE / HOURLY_PROFILE.sum()

    frames = []
    for d in dates:
        n = int(flights_per_day * (1.1 if d.dayofweek in (4, 6) else 1.0))
        dep_idx = rng.integers(0, len(airports), n)
        arr_idx = (dep_idx + rng.integers(1, len(airports), n)) % len(airports)
        adep = np.array(airports)[dep_idx]
        ades = np.array(airports)[arr_idx]
        hour = rng.choice(24, n, p=hour_p)

        # Day-level weather per airport, with a summer thunderstorm season.
        summer = d.month in (6, 7, 8)
        wx = {
            a: dict(
                wind=rng.gamma(2.0, 6.0),
                vis=float(np.clip(rng.normal(9000, 2500), 300, 10000)),
                conv=rng.random() < (0.18 if summer else 0.04),
            )
            for a in airports
        }

        df = pd.DataFrame({"adep": adep, "ades": ades, "sched_hour": hour})
        df["date"] = d
        df["aircraft_type"] = rng.choice(types, n,
                                         p=[.3, .25, .15, .08, .07, .06, .05, .04])
        df["wake_cat"] = df["aircraft_type"].map(AIRCRAFT)

        # Demand/capacity: departures in the same airport-hour vs capacity.
        counts = df.groupby(["adep", "sched_hour"])["adep"].transform("size")
        # Scale declared capacity to the sample size so peaks sit near 1.0-1.3.
        cap = df["adep"].map(lambda a: AIRPORTS[a][0]) * flights_per_day / 6000
        df["demand_capacity_ratio"] = (counts / cap).round(3)

        df["wind_kt"] = df["adep"].map(lambda a: wx[a]["wind"]).round(1)
        df["visibility_m"] = df["adep"].map(lambda a: wx[a]["vis"]).round(0)
        df["convective"] = df["adep"].map(lambda a: wx[a]["conv"]).astype(int)
        df["planned_flight_time_min"] = rng.normal(130, 45, n).clip(45, 260).round()
        # Reactionary delay builds up through the day.
        df["prev_leg_delay_min"] = (rng.exponential(2 + 0.35 * hour, n)
                                    * (hour > 6)).round(1)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dep_delay_min"] = _simulate_delay(df, rng)
    df.insert(0, "flight_id", [f"F{i:07d}" for i in range(len(df))])
    return df[SCHEMA]


def _simulate_delay(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    congestion = df["adep"].map(lambda a: AIRPORTS[a][1]).to_numpy()
    dcr = df["demand_capacity_ratio"].to_numpy()
    delay = (
        -1.0
        + 30.0 * np.maximum(dcr - 0.85, 0) ** 1.5 * congestion   # capacity (C)
        + 22.0 * df["convective"].to_numpy()                       # weather (W)
        + 0.25 * np.maximum(df["wind_kt"].to_numpy() - 20, 0)
        + 8.0 * (df["visibility_m"].to_numpy() < 1500)             # LVP
        + 0.45 * df["prev_leg_delay_min"].to_numpy()               # reactionary
        + 3.0 * (df["wake_cat"].to_numpy() == "H")
        + 2.5 * np.isin(df["day_of_week"].to_numpy(), [4, 6])
    )
    noise = rng.gamma(1.2, 4.0, len(df)) - 3.0
    early = rng.random(len(df)) < 0.12
    out = np.where(early, -rng.uniform(0, 8, len(df)), delay + noise)
    return np.round(out, 1)


def load_flights(path: str | Path) -> pd.DataFrame:
    """Load a CSV/Parquet file that follows SCHEMA and validate it."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    return df
