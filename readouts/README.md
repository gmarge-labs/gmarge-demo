# readouts/

Generated AI read-outs live here as committed files.

The rule (see `CLAUDE.md`): read-outs are produced **offline**, checked by a
human, and saved here. The Streamlit app only ever reads these files — it never
calls an AI API at runtime. Every number in a read-out is computed in Python
and passed to the model as a fact; the model's only job is to phrase it.

Written by `python -m gmarge.analyst --weeks 8`, one file per week:

| field | |
|---|---|
| `week`, `week_start`, `week_end` | which week the read-out covers |
| `brand`, `disclaimer` | the sample-brand disclaimer travels with the text |
| `text` | the read-out itself, at most 120 words |
| `facts` | every number the read-out was allowed to use |
| `mode`, `model` | `model` with the model id, or `dry-run` with `model: null` |
| `generated_at` | UTC timestamp of the run |
| `word_count`, `word_limit` | |

Every figure in `text` has already been traced back to a value in `facts`
before the file was written — a read-out that fails that check is not saved.
The check is not a substitute for reading the thing: these are drafts until a
human has compared the prose with the facts beside it.
