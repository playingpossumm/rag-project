# Archive

Code kept for reference and not imported by anything. `2026-08-23-pre-refinement/`
is a snapshot of the interface before the interaction pass of that date and has
its own README.

`late_interaction.py` and `web_fallback.py` were moved here from `src/` on
2026-09-06. The first is ColBERT-style MaxSim scoring over per-token
embeddings, written as a reranking stage over the candidate pool; the second is
an opt-in web search for questions the corpus declines, with no provider
registered. Neither was ever wired into `ask()`, no test imported either, and
the README's module table listed both as if they were stages of the pipeline
until the same day. `late_interaction.py` lost to the cross-encoder, which was
already measured on every corpus and is what the server runs; `web_fallback.py`
contradicts the project's rule that a question about private documents never
leaves the machine, and its own docstring says so. Both still import on their
own and neither is maintained.
