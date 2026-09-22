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

## Data quality

`gmarge/quality.py` looks for the four things that are wrong with these tables.
Each check returns findings carrying a severity, the dates, the table, a
plain-English description and the numbers behind it:

```python
from gmarge.quality import run_checks

for finding in run_checks("data"):
    print(finding.severity, finding.check, finding.description)
```

On this dataset it returns four findings and nothing else: the two GA4 days
missing in week 8, the Meta pixel double count across 2025-03-25 → 03-27, and
the reporting lag on the last two days of `ad_spend` and of `ga4_sessions`. It
measures the lag at 82% and 45% of a settled day for `ad_spend`, against the
planted 82% and 45%.

Two of the checks are mostly about what they *don't* say. The table holds 163
pairs of orders that are identical line for line by coincidence — same day,
region, price and basket — so the duplicate check reports a repeated
`order_id` outright, but reports identical *contents* only on a date where the
rate of them is far above the table's own background rate, which is what a
batch loaded twice looks like. And a channel going dark in five regions for a
geo holdout moves its share of store revenue without anything being wrong, so
the double-count check measures a platform against what its own spend predicts
rather than against the store, and only fires when Shopify's revenue and order
count stood still.

## Anomalies

`gmarge/anomalies.py` scores every paid channel and week against that
channel's own trailing 8 weeks, using the median and the median absolute
deviation. Each flag names the campaign and ad set behind most of the move and
its share of it:

```python
from gmarge.anomalies import detect_anomalies

for flag in detect_anomalies("data"):
    print(flag.week, flag.channel, flag.metric, flag.ad_set, flag.share_of_move)
```

It scores *rates* — reported ROAS, CPC, CPM, CTR, frequency — not spend or
revenue levels, because levels move for planned reasons: budgets get raised, a
holdout takes a channel dark in a third of the account, a promo week lifts
everything. A rate survives all three, and survives the last two days still
filling in, because a lag scales a rate's numerator and denominator together.

A flag needs both a robust score past 3.5 and a move of at least 10%. Of the
414 comparisons it makes over 26 weeks, the 399 that hold no planted fault all
sit within 3.2% of their trailing median while the faults move 19% to 41%, so
the two conditions have roughly three times the headroom in each direction.
On this dataset the scan returns four flags and no false alarms:

| week | channel | metric | move | ad set behind it | share |
|---|---|---|---:|---|---:|
| 12 | Meta prospecting | reported ROAS | +41.2% | `Core \| Broad 25-44` | 32% |
| 12 | Meta retargeting | reported ROAS | +41.5% | `RET \| 7d Add-to-Cart` | 61% |
| 20 | Meta prospecting | frequency | +39.2% | `Core \| Broad 25-44` | 95% |
| 20 | Meta prospecting | reported ROAS | −18.8% | `Core \| Broad 25-44` | 100% |

Week 12 is the pixel firing twice for the whole platform, so the shares track
each ad set's size and no single ad set is blamed. Week 20 is one ad set
burning out, and the attribution says so.

Neither module calls an AI API, and neither reads an answer out of
`truth.json`.

## Read-outs

`gmarge/analyst.py` writes the weekly read-outs that the app displays. It is an
offline script, and it is the only thing in the repo that calls a model —
through `gmarge/llm.py`, the only module that imports `anthropic`:

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY; .env is git-ignored
python -m gmarge.analyst --weeks 8            # writes readouts/week-NN.json
python -m gmarge.analyst --weeks 8 --dry-run  # no API call at all
```

The model does not see the data. For each week, `build_facts()` computes every
number the read-out is allowed to use — the week's totals, the move against the
prior week, each channel's share of that move, the trailing 8-week band that
says whether the week is outside its normal range, the week's anomalies and any
quality finding that overlaps it — and hands them over as JSON. The model is
asked for at most 120 words of prose: what changed, what caused most of it,
whether it is outside the normal range, and what is missing if the week's data
is incomplete.

Incompleteness gets its own counts, because a week that is partly missing is
the one place a model is tempted to do arithmetic: how much of week 26 reported
is a subtraction, and a subtraction is exactly what it is not allowed to make.
So `completeness` carries the days in the week, the days with complete data,
the days affected and the affected dates themselves, and the prompt says to
describe incompleteness with those counts and never to count or subtract days.

Then the reply is checked. Every figure in it is extracted and traced back to a
value in the facts: a figure may be written to fewer decimal places, a ratio may
be written as a percentage, and a name from the facts may be quoted with the
digits it carries. Nothing else passes. A reply that fails is sent back once
with the offending figures named; if the second reply also fails, **nothing is
saved** and the week is reported as an error. A failure prints each offending
figure with the sentence it was written in — `6` on its own says nothing, `'6'
in: only 6 of 7 days reported` says the model was counting days — so a failure
can be diagnosed without running it again.

Each file holds the text, the facts it was written from, the model id and a
timestamp, so a reviewer can check the prose against the numbers it came from.
They are drafts until a human has read them — that review is the step between
generation and commit, and it is not optional.

`--dry-run` writes the same file with a read-out assembled in Python instead,
and `model: null`. The tests use only that path: no key, no network, no model.
It doubles as a check on the facts — if the template cannot be written from the
facts alone, neither can the model.

## Tests

```bash
pytest
```

`tests/test_planted_truths.py` looks for each planted fact the way an analyst
would — measuring it from the tables, not reading it back out of the generator.
`tests/test_quality.py` and `tests/test_anomalies.py` hold the detectors to the
same standard: every planted defect found, checked against the figures in
`truth.json`, with a budget of two false alarms across the 26 weeks that
neither module spends. Both are also run against a second seed, so a threshold
cannot pass by landing well on one dataset.

## Ground rules

`CLAUDE.md` holds the rules for working in this repo: clean-room build, no live
AI calls from the app, Python computes every number, secrets only in a
git-ignored `.env`, minimal pinned dependencies.
