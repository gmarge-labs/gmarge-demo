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
actually took: $10.96M claimed against $7.83M of Shopify revenue. Every channel
takes credit for the same conversions.

### 2. Reported ROAS is not incremental ROAS

| channel | spend | reported ROAS | true incremental ROAS | true as % of reported |
|---|---:|---:|---:|---:|
| Meta prospecting | 1,751,880 | 2.44 | 1.79 | 73% |
| Google shopping | 826,398 | 2.70 | 1.31 | 48% |
| TikTok | 541,708 | 2.27 | 1.00 | 44% |
| Meta retargeting | 493,942 | 3.65 | 0.60 | 16% |
| Google branded search | 350,238 | 4.05 | 0.45 | 11% |

Branded search and retargeting look like the best channels on reported ROAS and
are the worst on truth — they mostly take credit for demand that already
existed. Meta prospecting reports the lowest ROAS of the top three and is the
closest to its true value.

### 3. Five geo holdouts

Each paid channel was paused in 5 test regions for 4 weeks, at a different time,
with 5 size-matched control regions left running.

| channel | weeks | dates | spend paused | true revenue lost | implied true iROAS |
|---|---|---|---:|---:|---:|
| Meta prospecting | 5–8 | 2025-02-03 → 2025-03-02 | 84,413 | 153,733 | 1.82 |
| Meta retargeting | 9–12 | 2025-03-03 → 2025-03-30 | 15,251 | 9,283 | 0.61 |
| Google branded search | 13–16 | 2025-03-31 → 2025-04-27 | 17,321 | 7,767 | 0.45 |
| Google shopping | 17–20 | 2025-04-28 → 2025-05-25 | 27,434 | 35,500 | 1.29 |
| TikTok | 21–24 | 2025-05-26 → 2025-06-22 | 29,114 | 28,806 | 0.99 |

Regions are built as 10 matched pairs of near-identical size, listed in
`truth.json` under `other_features.matched_region_pairs`. Consecutive holdouts
use disjoint halves of those pairs, so the four weeks before any test are a
usable baseline for it. A matched-market difference-in-differences on Shopify
revenue recovers each true lift to within a few percent — `tests/` does exactly
that.

### 4. Four things wrong with the data

| week | what | where |
|---|---|---|
| 20 | One Meta prospecting ad set, `Core \| Broad 25-44`, doubles its frequency (2.08×) and its ROAS collapses from 2.40 to 1.07 — 2025-05-19 → 05-25 | `ad_spend` |
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
