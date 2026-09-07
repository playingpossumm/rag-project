# UI snapshot — 23 Aug 2026, before the interaction refinement

Taken at commit `21b3c56`, the last state before the pass that renamed the site
to "retrieval visualized", stripped the header down to one link, moved the
dataset choice into the hero, and put the example questions behind the input.

Kept because that pass changed the front page substantially and the earlier
version is worth being able to look at rather than reconstruct from a diff.

## What is here

- `index.html` — the page, self-contained apart from `/pipeline-map.js` and the
  bundled fonts.
- `pipeline-map.js` — the isometric drawing at 60 degrees, with each stage a
  block of sheets whose depth tracks how many candidates survive.

## Running it

It is a snapshot, not a second app. To look at it:

    git stash                     # or commit whatever is in progress
    git checkout 21b3c56 -- ui/
    python src/serve.py
    # then: git checkout HEAD -- ui/

Serving these files directly will not work on its own — they expect the API
that `src/serve.py` provides.

## The state of it

Everything in the tree at that commit passed: 22/22 metric checks, 15/15 loader
checks, 80/80 trace checks.
