"""The generator must be deterministic: same seed in, same bytes out."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from gmarge import generate as gen

FILES = ["shopify_orders.parquet", "ad_spend.parquet", "ga4_sessions.parquet", "truth.json"]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_gives_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    gen.generate(a)
    gen.generate(b)

    for name in FILES:
        assert _digest(a / name) == _digest(b / name), f"{name} is not reproducible"


def test_same_seed_gives_identical_frames(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    gen.generate(a)
    gen.generate(b)

    for name in FILES[:-1]:
        pd.testing.assert_frame_equal(pd.read_parquet(a / name), pd.read_parquet(b / name))


def test_different_seed_gives_different_data(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    gen.generate(a, seed=gen.SEED)
    gen.generate(b, seed=gen.SEED + 1)

    assert _digest(a / "shopify_orders.parquet") != _digest(b / "shopify_orders.parquet")

    # ...but the planted structure survives a change of seed.
    ta, tb = (json.loads((d / "truth.json").read_text()) for d in (a, b))
    assert ta["attribution_gap"]["ratio"] == tb["attribution_gap"]["ratio"]
    assert (
        ta["incrementality_ranking"]["closest_to_reported"]
        == tb["incrementality_ranking"]["closest_to_reported"]
    )


def test_generate_returns_the_written_truth(tmp_path):
    returned = gen.generate(tmp_path)
    assert returned == json.loads((tmp_path / "truth.json").read_text())
