# CLAUDE.md

## Writing style

Everything written in this project, including documentation, the README, code comments,
commit messages and any prose produced for it, follows the register set out in
[`docs/writing-style.md`](docs/writing-style.md). Read that file before writing prose here.
The fenced block inside it is the authority; the summary below exists so that the rules are
not missed when the file is not open.

Neutral, straightforward and formal, reading like a reference work rather than a blog post.
Orwell's six rules override preference. Sentences are full and elaborated rather than
stacked fragments, and the colon is used sparingly. A heading is a label and never a
question. Prose is the default, with tables reserved for genuinely two-dimensional content
and bullets for strictly parallel one-line items. Dates are absolute, numbers are digits and
symbols are standard. Nothing rates the work as important or exciting. When a mistake was
corrected, the mistake is recorded alongside the correction.

## Standing constraints

`.env` holds an Anthropic API key with no credit. It is never echoed, committed, or placed
in an artifact.

The git identity is repo-local and deliberate, and it is not changed.

`src/serve.py` binds 127.0.0.1 on purpose, because a corpus may be private. The interface
loads no CDN and makes no external request, and the fonts are bundled in `ui/fonts/`.

The chart palette is validated all-pairs for colourblind separation. A fourth hue requires
re-running the dataviz validator first.

A `\n` inside a bash heredoc becomes a real newline and breaks Python and JavaScript string
literals, so patch scripts are written with the Write tool rather than pasted into a
heredoc.
