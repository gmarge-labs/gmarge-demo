"""Shared fixtures. The dataset is generated once per test session."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmarge import generate as gen  # noqa: E402


@dataclass
class Dataset:
    path: Path
    orders: pd.DataFrame
    ad_spend: pd.DataFrame
    ga4: pd.DataFrame
    truth: dict

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(gen.START_DATE, periods=gen.N_DAYS, freq="D")

    def week(self, n: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Inclusive (start, end) timestamps of a 1-indexed week."""
        days = gen._week_days(n)
        return self.dates[days[0]], self.dates[days[-1]]


def _load(path: Path) -> Dataset:
    return Dataset(
        path=path,
        orders=pd.read_parquet(path / "shopify_orders.parquet"),
        ad_spend=pd.read_parquet(path / "ad_spend.parquet"),
        ga4=pd.read_parquet(path / "ga4_sessions.parquet"),
        truth=json.loads((path / "truth.json").read_text()),
    )


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> Dataset:
    out = tmp_path_factory.mktemp("data")
    gen.generate(out)
    return _load(out)


@pytest.fixture(scope="session")
def other_dataset(tmp_path_factory) -> Dataset:
    """A second dataset, from a different seed.

    A detection threshold has to survive a change of seed, not just land on
    the one dataset it was set against.
    """
    out = tmp_path_factory.mktemp("data_alt")
    gen.generate(out, seed=gen.SEED + 1)
    return _load(out)


@pytest.fixture(scope="session")
def load():
    return _load
