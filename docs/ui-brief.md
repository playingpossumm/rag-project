# UI design brief

Answers from a design interview on 2026-08-19, recorded so the direction survives
a session restart. These are the user's stated preferences, not inferences.

## Direction

| Question | Answer |
|---|---|
| Audience | **All of them** — portfolio showpiece, personal debugging tool, demo for non-technical colleagues, and eventually a real tool |
| Aesthetic | **Editorial × laboratory instrument.** Verbatim: *"elegant and complex yet easy to understand and follow"* |
| Visual reference | **Stripe docs / Observable** — light, editorial, generous whitespace, diagrams integrated into prose |
| Typography | **Serif headings, sans body, mono data** |
| Density | **Layered** — summary first, detail on demand |
| Motion | **A mix of all three** — purposeful (meaning-carrying), choreographed (load sequence, reveals), and ambient (generative centrepiece) |
| Ambient placement | **Undecided — wants to see several options side by side before choosing** |
| Default theme | **Follow the system** |
| Landing state | Search box first. Also wants a way to **point it at their own folder** |
| Refusal display | **Prominent and explained** — the refusal is a differentiator and should read as a feature, not an error |
| Folder linking | **Deferred** — "later" |
| Feature set | Delegated: *"you decide what's best"* |

## Consequences worth remembering

**The three aesthetic inputs are in tension and that is the actual design problem.**
Editorial wants air, prose and serif; instrument wants density, monospace and
restraint; "easy to follow" wants less on screen than "complex" implies. The
resolution is layering — an editorial surface that opens into instrument-grade
detail — not splitting the difference on every element.

**"A mix of all three" for motion cannot be taken literally.** Ambient generative
motion and "almost none" are opposites. Read as: motion is welcome, but it must
justify itself. Build the ambient options as a comparison the user picks from,
since that is what they explicitly asked for.

**Feature choice was delegated.** Recommended priority, highest value first:

1. **Compare pipeline settings live** — toggle reranking, fusion and the diversity
   cap and watch the ranking move. This is the single strongest differentiator:
   it is the thing a closed product structurally cannot offer, and it turns the
   inspector into an experiment bench.
2. **Read the full passage** — click a result for the whole chunk and its
   neighbours. Cheap to build, immediately useful.
3. **Evaluation view** — the 84-case golden set and how each configuration scores.
   Makes the measurements visible rather than claimed.
4. **Save/share a trace** — lowest priority; useful only once someone else is looking.

## Constraint discovered during the interview

A browser cannot read a real filesystem path from a file picker. So "link my
folder" must be either a pasted path (works with synced Drive folders, no
copying) or drag-and-drop (copies files into `data/`). The user deferred the
decision; the constraint stands whenever it is picked up.

## Defects found by looking at the rendered page

Screenshotting the UI with Playwright caught four things that reading the markup
did not. Recorded because the lesson generalises: **render it and look at it.**

- Adjacent right-aligned table columns collided (`#60.5416`) — `padding-right:0`
  removed the gap between columns, not just at the table edge.
- Endpoint labels truncated mid-token (`attention_is_all_you_n`).
- The bump chart drew a fixed 12 rank rows; real queries only ever filled ~6, so
  every chart carried five empty rows.
- A document appearing twice in the final answer was labelled twice.

All four are fixed. The first was invisible in source and obvious on screen.
