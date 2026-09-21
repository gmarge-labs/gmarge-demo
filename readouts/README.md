# readouts/

Generated AI read-outs live here as committed files.

The rule (see `CLAUDE.md`): read-outs are produced **offline**, checked by a
human, and saved here. The Streamlit app only ever reads these files — it never
calls an AI API at runtime. Every number in a read-out is computed in Python
and passed to the model as a fact; the model's only job is to phrase it.
