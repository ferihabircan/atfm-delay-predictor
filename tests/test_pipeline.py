import numpy as np
import pytest

from delay_predictor.data import SCHEMA, generate_flights, load_flights
from delay_predictor.model import add_features, time_split, train_and_evaluate


@pytest.fixture(scope="module")
def flights():
    return generate_flights(n_days=30, flights_per_day=400, seed=1)


def test_schema_and_reproducibility(flights):
    assert list(flights.columns) == SCHEMA
    again = generate_flights(n_days=30, flights_per_day=400, seed=1)
    assert flights.equals(again)
    assert (flights["adep"] != flights["ades"]).all()


def test_time_split_has_no_overlap(flights):
    train, test = time_split(flights)
    assert train["date"].max() < test["date"].min()


def test_cyclic_hour_features(flights):
    f = add_features(flights)
    assert np.allclose(f["hour_sin"] ** 2 + f["hour_cos"] ** 2, 1)


def test_model_beats_baseline(flights):
    rep, _ = train_and_evaluate(flights)
    assert rep.metrics["model_mae"] < rep.metrics["baseline_mae"]
    assert rep.metrics["model_auc"] > 0.75


def test_load_rejects_bad_schema(tmp_path, flights):
    p = tmp_path / "bad.csv"
    flights.drop(columns=["wind_kt"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="wind_kt"):
        load_flights(p)
