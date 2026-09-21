# gmarge-demo

> **This is sample data for a fictional brand.**
> "Northfield Goods (sample brand)" does not exist. Every figure in `data/` is
> synthetic, generated from a fixed seed by `gmarge/generate.py`. It is not any
> real company's revenue, not any real advertiser's performance, and must never
> be presented as such.

A marketing-measurement demo built on a dataset with **known ground truth**.
The generator plants a specific set of facts in the data — an attribution gap,
a true incremental ROAS per channel, five geo holdouts and four data-quality
artifacts — and records all of them in `data/truth.json`. That makes it possible
to check an analysis against what is actually true, which you can never do with
real marketing data.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m gmarge.generate --out data   # regenerate the sample data
pytest                                 # verify every planted truth is present
```

Python 3.11+.

## The data

26 weeks of daily data (2025-01-06 to 2025-07-06) across 20 regions,
`Region 01` … `Region 20`.

| file | grain | columns |
|---|---|---|
| `data/shopify_orders.parquet` | one row per order | `order_id`, `date`, `region`, `revenue`, `discount`, `refund`, `net_revenue`, `items`, `customer_type` |
| `data/ad_spend.parquet` | date × region × ad set | `date`, `region`, `channel`, `campaign`, `ad_set`, `spend`, `impressions`, `clicks`, `frequency`, `platform_attributed_revenue` |
| `data/ga4_sessions.parquet` | date × region × source/medium × landing page | `date`, `region`, `source_medium`, `landing_page`, `sessions` |
| `data/truth.json` | — | every planted truth, in full |

Paid channels: Meta prospecting, Meta retargeting, Google branded search,
Google shopping, TikTok. Email and organic drive revenue but have no spend, so
they appear in GA4 sessions and never in `ad_spend`. `frequency` is null for the
two search channels, which don't report it.

`revenue` is gross, before `discount` and `refund`; `net_revenue` is what's left.

## What is planted in the data

### 1. The attribution gap

Platform-attributed revenue summed across channels is **1.40×** what the store
actually took: $18.34M claimed against $13.10M of Shopify revenue. Every channel
takes credit for the same conversions.

### 2. Reported ROAS is not incremental ROAS

| channel | spend | reported ROAS | true incremental ROAS | true as % of reported |
|---|---:|---:|---:|---:|
| Meta prospecting | 973,267 | 3.53 | 2.29 | 65% |
| Google shopping | 885,427 | 5.01 | 2.06 | 41% |
| TikTok | 580,401 | 4.19 | 1.61 | 38% |
| Meta retargeting | 474,184 | 7.10 | 1.10 | 16% |
| Google branded search | 583,729 | 8.01 | 0.80 | 10% |

Branded search and retargeting look like the best channels on reported ROAS and
are the worst on truth — they mostly take credit for demand that already
existed. Branded search reports 8.01 and truly returns 0.80: less gross revenue
than it costs. Meta prospecting reports the *lowest* ROAS of the five and is the
closest to its true value, so ranking the channels on what the platforms say
gets the order almost exactly backwards.

### 3. Five geo holdouts

Each paid channel was paused in 5 test regions for 4 weeks, at a different time,
with 5 size-matched control regions left running.

| channel | weeks | dates | spend paused | true revenue lost | lift on test regions | implied true iROAS |
|---|---|---|---:|---:|---:|---:|
| Meta prospecting | 5–8 | 2025-02-03 → 2025-03-02 | 46,896 | 109,131 | 18.0% | 2.33 |
| Meta retargeting | 9–12 | 2025-03-03 → 2025-03-30 | 14,641 | 16,338 | 4.1% | 1.12 |
| Google branded search | 13–16 | 2025-03-31 → 2025-04-27 | 28,868 | 23,014 | 3.6% | 0.80 |
| Google shopping | 17–20 | 2025-04-28 → 2025-05-25 | 29,393 | 59,980 | 14.4% | 2.04 |
| TikTok | 21–24 | 2025-05-26 → 2025-06-22 | 31,194 | 49,382 | 7.4% | 1.58 |

Every holdout moves its test regions' revenue by 3–20%, which is the range a
real geo test lands in. A channel's effect size there depends on the channel
mix and the ROAS levels, not on the overall budget: revenue is pinned to total
attributed revenue by the 1.40× ratio, so scaling every budget scales revenue
with it.

Regions are built as 10 matched pairs of near-identical size, listed in
`truth.json` under `other_features.matched_region_pairs`. Consecutive holdouts
use disjoint halves of those pairs, so the four weeks before any test are a
usable baseline for it. A matched-market difference-in-differences on Shopify
revenue recovers each true lift to within a few percent — `tests/` does exactly
that.

### 4. Four things wrong with the data

| week | what | where |
|---|---|---|
| 20 | One Meta prospecting ad set, `Core \| Broad 25-44`, doubles its frequency (2.08×) and its ROAS collapses from 3.47 to 1.55 — 2025-05-19 → 05-25 | `ad_spend` |
| 12 | The Meta pixel double-counts purchases for 3 days, 2025-03-25 → 03-27. Attributed revenue only: spend, clicks and Shopify revenue are untouched | `ad_spend` |
| 8 | Two days missing entirely, 2025-02-26 and 02-27. No rows at all; the days exist in Shopify and `ad_spend` | `ga4_sessions` |
| 26 | The last two days, 2025-07-05 and 07-06, are still filling in — about 82% and 45% of final spend. Shopify is complete | `ad_spend`, `ga4_sessions` |

There is also a 4-day promo in week 15 (2025-04-14 → 04-17): higher demand and a
much higher share of discounted orders.

## How it is built

Revenue is modelled from the bottom up. Each region-day is unpaid baseline
demand plus the true incremental revenue each channel's spend actually caused,
times a demand shock. Platform-attributed revenue is generated separately, from
each channel's *reported* ROAS — that gap is the point. Baseline demand is then
scaled so the totals land exactly on the planted 1.40× ratio, and the region-day
revenue is exploded into orders that sum back to it to the cent.

Noise is split into a day-level shock shared by every region and a smaller
region-day wobble. That is both realistic — budget changes and press hits are
account-wide, not per-region — and what keeps matched-market comparisons
workable, since a shared shock cancels between test and control.

Generation is deterministic. Every draw comes from a named `numpy` bit-stream
derived from the seed, so the same seed produces byte-identical files no matter
what else changes. `tests/test_reproducibility.py` asserts it.

## Metrics

`gmarge/metrics.py` computes everything the app and the offline read-out
generator need, in pandas. One call returns the lot:

```python
from gmarge.metrics import all_metrics

m = all_metrics("data")
m["headline"]["over_claim_ratio"]        # 1.40
m["weekly_reconciliation"]               # Shopify vs attributed, by week
m["reported_roas_by_channel_week"]       # what the platforms claim
m["holdouts"]                            # what the channels are actually worth
```

The holdout estimator is a matched-market difference-in-differences: each test
region is predicted from its paired control region using their ratio over an
equally long pre-period, and the shortfall is the lift. The spend the paused
regions *would* have had is predicted the same way, which gives incremental
ROAS. The 90% interval comes from a bootstrap over the five matched pairs — the
independent units of a geo test — so it reflects how few markets a test like
this really has. On this dataset it recovers every planted lift to within 6%,
and every planted truth falls inside its interval.

The module never calls an AI API, and never reads an answer out of
`truth.json`: the only thing it takes from there is the test plan — which
regions were paused and when — which is what an analyst would have from the
media plan. `holdout_plan()` strips the rest, and a test asserts it.

## Tests

```bash
pytest
```

`tests/test_planted_truths.py` looks for each planted fact the way an analyst
would — measuring it from the tables, not reading it back out of the generator.

## Ground rules

`CLAUDE.md` holds the rules for working in this repo: clean-room build, no live
AI calls from the app, Python computes every number, secrets only in a
git-ignored `.env`, minimal pinned dependencies.
