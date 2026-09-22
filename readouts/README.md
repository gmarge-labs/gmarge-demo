# readouts/

Generated AI read-outs live here as committed files.

The rule (see `CLAUDE.md`): read-outs are produced **offline**, checked by a
human, and saved here. The Streamlit app only ever reads these files — it never
calls an AI API at runtime. Every number in a read-out is computed *and
formatted* in Python and passed to the model as a display string; the model's
only job is to quote it.

Written by `python -m gmarge.analyst --weeks 8`, one file per week:

| field | |
|---|---|
| `week`, `week_start`, `week_end` | which week the read-out covers |
| `brand`, `disclaimer` | the sample-brand disclaimer travels with the text |
| `text` | the read-out itself, at most 120 words |
| `facts` | every number the read-out was allowed to use, as `{value, display}` |
| `mode`, `model` | `model` with the model id, or `dry-run` with `model: null` |
| `generated_at` | UTC timestamp of the run |
| `word_count`, `word_limit` | |

The model is shown the `display` half of each fact and nothing else, and every
figure in `text` was matched back to one of those strings before the file was
written — a read-out that fails that check is not saved.

A week that fails the check is not written, and any read-out already here for
that week is deleted rather than left to look current. A missing `week-NN.json`
means that week failed; it never means the old one still applies.

Three things the check cannot do for you, so read for them:

* **Does it lead with what matters?** A flagged week should open with the flag,
  never with the totals.
* **Does it claim a week is normal?** Only `anomalies` can say that, and a week
  with no flag is a week nothing was flagged in — not a week that went well.
* **Does a holdout result belong in this week?** It may appear in the week the
  test concluded and once more the week after. Earlier than that, the answer
  did not exist yet.

These are drafts until a human has compared the prose with the facts beside it.
