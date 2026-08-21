/**
 * The authored half of the architecture map.
 *
 * The rule this file exists to keep: prose, groups and flows are authored;
 * counts, coverage and geometry are measured. Everything written by hand
 * here is something no scanner could produce -- what a subsystem is *for*, why
 * it was built the way it was, and which paths through it are worth walking.
 * Everything numeric comes from `measured.generated.ts`, which the sync script
 * rewrites from the tree, so the map cannot quietly disagree with the code.
 *
 * The prose is drawn from the modules' own docstrings, `README.md` and
 * `HANDOFF.md` rather than invented. Where this project measured something
 * that overturned the obvious plan, the building says so -- that history is
 * the most useful thing on the map and the first thing a summary would drop.
 *
 * Edges are real import or filesystem paths. If a line is drawn here, there is
 * a statement in the source that makes it happen; the import graph was
 * extracted with `ast` rather than read off the architecture diagram, because
 * the diagram is the thing being checked.
 */
import { deriveArchetype, deriveHeight, deriveSize, packLayout } from './core/layout'
import type { ArchEdge, ArchFlow, ArchNode, Group } from './core/types'
import { MEASURED, UNCLAIMED } from './measured.generated'

/* --------------------------------------------------------------- groups
   Ordered as the story runs, because the legend lists them in this order and
   the packer lays them out in it: the ways in, then how documents get here,
   then what the system refuses to redo, then how a question is answered, then
   how any of it is shown to be true. */
export const GROUPS: readonly Group[] = [
  { id: 'entry', label: 'Ways in' },
  { id: 'ingestion', label: 'Documents in' },
  { id: 'caching', label: 'Work it skips' },
  { id: 'retrieval', label: 'Answering' },
  { id: 'evaluation', label: 'Proving it works' },
]

type Authored = Omit<ArchNode, 'archetype' | 'params' | 'footprint' | 'height' | 'count' | 'loc'>

const AUTHORED: Authored[] = [
  /* ============================================================== entry */
  {
    id: 'api', code: 'AP', name: 'ask()', role: 'the single entry point', group: 'entry',
    whatItDoes:
      'The one function the whole system is reached through. Takes a question, returns ' +
      'passages with citations and a confidence verdict — as [[structured data]], never ' +
      'printed, so a caller can render it, store it, or diff two runs.',
    howItsBuilt:
      'Retrieval-only by default: returning the source text verbatim costs nothing and ' +
      'cannot hallucinate, so the system stays fully usable with no API key and no spend. ' +
      'Generation is an optional layer on top rather than a dependency.',
    files: ['src/api.py'],
    stack: ['sentence-transformers', 'FAISS'],
  },
  {
    id: 'serve', code: 'SV', name: 'HTTP server', role: 'the local server', group: 'entry',
    whatItDoes:
      'Serves two audiences from one process: POST /ask for another tool to embed, and ' +
      'the [[Retrieval Inspector]] at /, which runs the same pipeline but shows every ' +
      'intermediate ranking instead of only the five passages that survived.',
    howItsBuilt:
      'Standard library, not FastAPI. This is a thin adapter — parse JSON, call one ' +
      'function, serialise — and a web framework plus an ASGI server would be two more ' +
      'dependencies to pin and carry for no behaviour a caller can see. Binds to 127.0.0.1 ' +
      'deliberately: the corpus may hold private documents, so exposing it has to be an act.',
    files: ['src/serve.py'],
    stack: ['http.server', 'ThreadingHTTPServer'],
  },
  {
    id: 'cli', code: 'CL', name: 'Command line', role: 'the terminal shell', group: 'entry',
    whatItDoes:
      'Asks a question from a terminal. Readable output by default, JSON when piped ' +
      'somewhere that wants to parse it.',
    howItsBuilt:
      'The third and last shell over ask(). Library, HTTP and CLI all call the same ' +
      'function, so the three interfaces cannot drift apart in behaviour.',
    files: ['src/cli.py'],
  },
  {
    id: 'trace', code: 'TR', name: 'Pipeline trace', role: 'the flight recorder', group: 'entry',
    whatItDoes:
      'Re-runs retrieval keeping every intermediate ranking — which retriever found what, ' +
      'how fusion reconciled two disagreeing orders, what the reranker promoted or buried, ' +
      'which passages the diversity cap displaced. It is what makes [[“why did it return ' +
      'that?”]] an answerable question.',
    howItsBuilt:
      'A deliberate duplicate of the pipeline rather than debug flags threaded through ' +
      'retrieve(): a serving path that carries debug state is one that eventually ships ' +
      'it, and the two have opposite goals — one wants to be fast and forget, this one ' +
      'wants to be slow and remember. This file claimed for months that a test held the ' +
      'two together; the test did not exist, and the duplication was held together by ' +
      'care alone. It exists now, and it earns its keep: retrieve() takes a different ' +
      'path when reranking is off — shortlisting k rather than candidate_k, and skipping ' +
      'the diversity cap — and the trace had to be taught the same, or it would have drawn ' +
      'a pool and a cap the serving path never ran.',
    files: ['src/pipeline_trace.py', 'src/test_trace.py'],
  },

  /* ========================================================== ingestion */
  {
    id: 'ingest', code: 'IN', name: 'Index builder', role: 'the assembly line', group: 'ingestion',
    whatItDoes:
      'Walks the corpus folder and turns documents into a searchable index: parse, check, ' +
      'chunk, embed, write FAISS. Emits [[structured progress events]] as it goes, which is ' +
      'what lets the Inspector show a live re-index without a second code path.',
    howItsBuilt:
      'Chunks are measured in tokens, not words, because the encoder\'s ceiling is a ' +
      'token count. Sizing in words let chunks reach ~1,200 tokens and everything past 256 ' +
      'was silently dropped before the encoder saw it — stored and citable, but invisible to ' +
      'search. Now 210 tokens with an assertion that refuses to build an index if any chunk ' +
      'exceeds the ceiling.',
    files: ['src/ingest.py'],
    stack: ['FAISS', 'NumPy', 'sentence-transformers'],
  },
  {
    id: 'loaders', code: 'LD', name: 'Format desk', role: 'the parser dispatch', group: 'ingestion',
    whatItDoes:
      'Reads PDF, Word, PowerPoint and Excel, and gives each one a citation locator its own ' +
      'format can honour: a PDF cites a page, PowerPoint a slide, Excel a sheet and row ' +
      'range, Word a [[section]].',
    howItsBuilt:
      'Word is the awkward one and drove the design. A .docx has no fixed pages — ' +
      'pagination is computed by the renderer and shifts with fonts and margins — so any ' +
      'page number would be wrong on the reader\'s copy. Sections are the unit that ' +
      'actually exists in the file. Note: only the PDF path has ever run on real files.',
    files: ['src/loaders.py'],
    stack: ['pymupdf4llm', 'python-docx', 'python-pptx', 'openpyxl'],
  },
  {
    id: 'ocr', code: 'OC', name: 'OCR fallback', role: 'the last resort', group: 'ingestion',
    whatItDoes:
      'Rescues pages that have no extractable text layer — scanned exhibits, a contract ' +
      'whose signature pages were scanned back in.',
    howItsBuilt:
      'Applied [[per page, not per document]]. The common real case is mixed, and running ' +
      'OCR across a whole document to rescue three pages would be slow and would also ' +
      '*degrade* the good pages, since OCR output is worse than a real text layer. ' +
      'Extraction stays primary; OCR only fills the gaps it leaves. Never run on real files.',
    files: ['src/ocr.py'],
    stack: ['RapidOCR', 'PyMuPDF'],
  },
  {
    id: 'health', code: 'HL', name: 'Corpus health', role: 'the gatekeeper', group: 'ingestion',
    whatItDoes:
      'Decides, before anything is embedded, whether a document will index badly — and ' +
      'excludes it loudly rather than letting it in quietly.',
    howItsBuilt:
      'Exists because the failure it prevents is [[silent]]. A scanned PDF has no text ' +
      'layer, so the extractor returns an empty string rather than an error: the page is ' +
      'chunked into nothing, embedded into nothing, and simply never retrieved. Evaluation ' +
      'keeps producing confident numbers, computed over a corpus with holes in it.',
    files: ['src/corpus_health.py'],
  },
  {
    id: 'fetch', code: 'FC', name: 'Corpus fetcher', role: 'the sample corpus', group: 'ingestion',
    whatItDoes:
      'Downloads twenty ML and NLP papers from arXiv so the system has something real to be ' +
      'measured against.',
    howItsBuilt:
      'The papers are [[topically adjacent on purpose]]. A corpus drawn from different ' +
      'fields would make retrieval and abstention look good for the wrong reason — telling ' +
      'physics from poetry is easy. Papers that all discuss transformers and training ' +
      'procedures force the retriever to separate genuinely similar sources.',
    files: ['src/fetch_corpus.py'],
  },

  /* ============================================================ caching */
  {
    id: 'parsecache', code: 'PC', name: 'Parse cache', role: 'the parse memo', group: 'caching',
    whatItDoes:
      'Remembers the parsed units of every document, so re-indexing an unchanged folder ' +
      'does not re-read a single PDF.',
    howItsBuilt:
      'Keyed on a hash of the file\'s [[bytes]], not its path or modification time. A moved ' +
      'or renamed file is recognised as the same content, so reorganising a folder does not ' +
      'trigger a full re-parse. Built because removing the embedding cost exposed parsing ' +
      'as the new bottleneck — 300 of the remaining 432 seconds.',
    files: ['src/parse_cache.py'],
  },
  {
    id: 'embedcache', code: 'EC', name: 'Embedding cache', role: 'the vector memo', group: 'caching',
    whatItDoes:
      'Stores every chunk embedding against a hash of its text and model, so adding one ' +
      'document to twenty does not re-embed the other twenty.',
    howItsBuilt:
      'Embeddings are a [[pure function]] of (text, model), so they cache perfectly. Keying ' +
      'on both means an edited document re-embeds only the chunks that actually changed, ' +
      'and switching models invalidates cleanly instead of silently mixing vector spaces. ' +
      'Together with the parse cache this is what makes a no-op re-index 0.76 s against a ' +
      '432 s cold build.',
    files: ['src/embedding_cache.py'],
    stack: ['NumPy'],
  },

  /* ========================================================== retrieval */
  {
    id: 'retrieve', code: 'RT', name: 'Retrieve', role: 'the conductor', group: 'retrieval',
    whatItDoes:
      'Runs the pipeline in order: shortlist by dense and sparse search, fuse, rerank, cap ' +
      'per document, expand to context. The five passages that come out are the answer.',
    howItsBuilt:
      'Over-retrieves on purpose — a [[candidate pool of 20]] for a top-5 answer. The ' +
      'reranker only reorders what the first stage hands it, so a relevant chunk missing ' +
      'from the candidate set can never be recovered; that trades a little latency for ' +
      'recall headroom.',
    files: ['src/retrieve.py'],
    stack: ['FAISS IndexFlatIP', 'all-MiniLM-L6-v2'],
  },
  {
    id: 'hybrid', code: 'HY', name: 'BM25 & fusion', role: 'the second opinion', group: 'retrieval',
    whatItDoes:
      'Adds a keyword retriever alongside the vector one and reconciles the two rankings. ' +
      'Worth +7.6 points of hit rate over dense alone.',
    howItsBuilt:
      'Fuses on [[rank, never score]]. A cosine similarity and a BM25 score live on ' +
      'incomparable scales, so anything that adds them needs per-query normalisation — ' +
      'which is relative, so a query where every candidate is mediocre still yields a top ' +
      'score of 1.0. RRF uses position only: scale-free, with nothing to tune.',
    files: ['src/hybrid.py'],
    stack: ['rank_bm25'],
  },
  {
    id: 'rerank', code: 'RR', name: 'Cross-encoder', role: 'the close reader', group: 'retrieval',
    whatItDoes:
      'Re-scores the shortlist by reading the question and the passage [[together]] in one ' +
      'pass. Lifts MRR from 0.601 to 0.710 — and leaves hit rate flat, which is the point ' +
      'of measuring both.',
    howItsBuilt:
      'Kept at MiniLM-L6 after the obvious upgrade was measured and rejected: an 8× larger ' +
      'model (BGE, 278M) fixed the *same* 2 of 10 failing cases at 10 s/query. Capacity is ' +
      'not the bottleneck — the failures are contrastive clauses, and negation is a known ' +
      'transformer weakness that scale does not resolve.',
    files: ['src/rerank.py'],
    stack: ['cross-encoder/ms-marco-MiniLM-L-6-v2'],
  },
  {
    id: 'diversify', code: 'DV', name: 'Diversity cap', role: 'the spread', group: 'retrieval',
    whatItDoes:
      'Takes at most two passages from any one document, so an answer that lives in five ' +
      'papers comes back from five papers.',
    howItsBuilt:
      'A per-document cap rather than [[MMR]]: MMR diversifies on embedding distance, which ' +
      'conflates similar wording with same source and needs a lambda tuned per corpus. Here ' +
      'the unit of redundancy is known exactly — it is the document. It is a trade, not a ' +
      'free win: +3.1 source recall for −1.6 any-hit. An earlier, smaller golden set said ' +
      'it was free; that was wrong.',
    files: ['src/diversify.py'],
  },
  {
    id: 'parent', code: 'PR', name: 'Context expansion', role: 'the widener', group: 'retrieval',
    whatItDoes:
      'Ranks on small chunks but returns bigger context around each hit, so an answer is ' +
      'not cut off mid-argument.',
    howItsBuilt:
      'Chunk size sits under opposing pressures — small chunks rank better, large chunks ' +
      'read better — so the two are separated rather than compromised. [[Window ±1]] is the ' +
      'default: it reaches 0.833 context recall for half the tokens page expansion needs. ' +
      'That reversed an earlier conclusion that window was strictly dominated.',
    files: ['src/parent.py'],
  },
  {
    id: 'abstain', code: 'AB', name: 'Abstention gate', role: 'the refusal', group: 'retrieval',
    whatItDoes:
      'Decides whether the corpus can answer at all, and declines when it cannot — the ' +
      'behaviour a closed product structurally will not offer.',
    howItsBuilt:
      'Nearest-neighbour search has no notion of "nothing here": ask about a topic the ' +
      'corpus has never heard of and it still returns its five least-unrelated chunks, at ' +
      'scores indistinguishable from a real match. The cross-encoder scores differently, so ' +
      'the gate reads *its* score. The threshold is [[0.0]], and the distributions overlap ' +
      '— a deliberate trade, recalibrated three times, not a clean split.',
    files: ['src/abstain.py'],
  },
  {
    id: 'generate', code: 'GN', name: 'Generation', role: 'the unproven layer', group: 'retrieval',
    whatItDoes:
      'The optional step that would turn retrieved passages into a written answer with ' +
      'inline citations. Everything else works without it.',
    howItsBuilt:
      'Written but never executed — the account has no credit, so no call has ever ' +
      'completed. Two things follow: the code is evidence-free, and api.py reaches for it ' +
      'with [[from generate_answer import synthesize]], a module that does not exist ' +
      '(the module is generate.py, and it has no synthesize). A lazy import inside the ' +
      'optional branch is why nothing has ever raised.',
    files: ['src/generate.py'],
    stack: ['anthropic', 'claude-opus-5'],
  },

  /* ========================================================= evaluation */
  {
    id: 'evaluate', code: 'EV', name: 'Eval harness', role: 'the scoreboard', group: 'evaluation',
    whatItDoes:
      'Scores retrieval configurations against the labelled set and writes every number the ' +
      'README quotes. Reports hit rate, MRR, NDCG, source recall and context recall, ' +
      'because a change can move one and not the others.',
    howItsBuilt:
      'Relevance is judged at [[page level]], not chunk level: coarser than ideal, but it ' +
      'is the granularity a human can label reliably and it stays stable when chunking ' +
      'parameters change. Its own NDCG once printed 1.373 — impossible for a normalised ' +
      'metric — because several returned chunks shared one gold page. Caught only because ' +
      'the number violated a bound the metric is known to have. A companion script writes ' +
      'the same run out case by case rather than in aggregate, which is what makes ' +
      '[[“which cases did this break?”]] answerable — and which immediately showed that ' +
      'the configuration documented as the default is measured on the weighted fusion ' +
      'branch while the app serves RRF.',
    files: ['src/evaluate.py', 'src/per_case.py'],
  },
  {
    id: 'goldenset', code: 'GS', name: 'Golden set', role: 'the labels', group: 'evaluation',
    whatItDoes:
      'Builds and audits the 84 labelled cases everything is measured against, and checks ' +
      'that each answer string actually occurs in the corpus [[as parsed]] rather than as ' +
      'the PDF renders it.',
    howItsBuilt:
      'Labels are derived, not written. Each case declares a distinctive answer string ' +
      'and every location containing it *becomes* gold, so labels cannot drift from the ' +
      'corpus. That rule earned itself: when the corpus grew from 1 document to 20, twelve ' +
      'of sixteen adversarial cases had silently become answerable and nothing errored.',
    files: [
      'src/build_golden_set.py', 'src/audit_golden_set.py', 'src/check_answers.py',
      'src/verify_candidates.py', 'src/label_multisource.py',
    ],
  },
  {
    id: 'probes', code: 'PB', name: 'Investigations', role: 'the diagnostics', group: 'evaluation',
    whatItDoes:
      'One-off scripts written to settle a specific question with a measurement instead of ' +
      'an argument: where cross-document confusion actually happens, whether a bigger ' +
      'reranker fixes it, what reranking changes on a single query.',
    howItsBuilt:
      'Each exists because one anecdote is not a problem statement. diagnose_crossdoc ' +
      'classifies every failing case by *where* it fails, because the three modes need ' +
      'different fixes; it found real confusion is [[~15%]], not the 20% the raw count ' +
      'suggested, because derived labels are narrow and score a document that answers in ' +
      'different words as wrong.',
    files: ['src/diagnose_crossdoc.py', 'src/compare_rerankers.py', 'src/eval_probe.py'],
  },
]

/* ---------------------------------------------------------------- edges
   Every one of these is an import statement or a filesystem read/write that
   exists in the source. The import graph was extracted with ast; nothing
   here was copied from the diagram in HANDOFF.md, because that diagram is one
   of the things this map is meant to check. */
const EDGE_SPEC: Omit<ArchEdge, 'flowIds'>[] = [
  // ways in -> the work
  { id: 'cli-api', from: 'cli', to: 'api', kind: 'call', label: 'the question' },
  { id: 'serve-api', from: 'serve', to: 'api', kind: 'call', label: 'the question' },
  { id: 'serve-trace', from: 'serve', to: 'trace', kind: 'call', label: 'the question' },
  { id: 'serve-ingest', from: 'serve', to: 'ingest', kind: 'call', label: 'a re-index request' },
  { id: 'serve-retrieve', from: 'serve', to: 'retrieve', kind: 'support', label: 'index & model, loaded once' },

  // ask()
  { id: 'api-retrieve', from: 'api', to: 'retrieve', kind: 'call', label: 'the question' },
  { id: 'api-hybrid', from: 'api', to: 'hybrid', kind: 'support', label: 'the BM25 index' },
  { id: 'api-parent', from: 'api', to: 'parent', kind: 'call', label: 'five passages' },
  { id: 'api-abstain', from: 'api', to: 'abstain', kind: 'call', label: 'the top score' },
  { id: 'api-generate', from: 'api', to: 'generate', kind: 'retry', label: 'passages to write from' },

  // the pipeline proper
  { id: 'retrieve-hybrid', from: 'retrieve', to: 'hybrid', kind: 'call', label: 'two rankings to fuse' },
  { id: 'retrieve-rerank', from: 'retrieve', to: 'rerank', kind: 'call', label: '20 candidates' },
  { id: 'retrieve-diversify', from: 'retrieve', to: 'diversify', kind: 'call', label: 'the reranked order' },
  { id: 'retrieve-parent', from: 'retrieve', to: 'parent', kind: 'call', label: 'the surviving chunks' },

  // inspection
  { id: 'trace-retrieve', from: 'trace', to: 'retrieve', kind: 'call', label: 'the dense shortlist' },
  { id: 'trace-hybrid', from: 'trace', to: 'hybrid', kind: 'call', label: 'the sparse shortlist' },
  { id: 'trace-rerank', from: 'trace', to: 'rerank', kind: 'call', label: 'the fused pool' },
  { id: 'trace-diversify', from: 'trace', to: 'diversify', kind: 'support', label: 'the cap setting' },
  { id: 'trace-abstain', from: 'trace', to: 'abstain', kind: 'call', label: 'the top score' },

  // building the index
  { id: 'fetch-ingest', from: 'fetch', to: 'ingest', kind: 'data', label: '20 arXiv PDFs into data/' },
  { id: 'ingest-health', from: 'ingest', to: 'health', kind: 'call', label: 'each document, before embedding' },
  { id: 'ingest-loaders', from: 'ingest', to: 'loaders', kind: 'call', label: 'a file to parse' },
  { id: 'loaders-ocr', from: 'loaders', to: 'ocr', kind: 'retry', label: 'pages with no text layer' },
  { id: 'ingest-parsecache', from: 'ingest', to: 'parsecache', kind: 'data', label: 'parsed units, by file hash' },
  { id: 'ingest-embedcache', from: 'ingest', to: 'embedcache', kind: 'data', label: 'chunk vectors, by text hash' },
  { id: 'ingest-retrieve', from: 'ingest', to: 'retrieve', kind: 'data', label: 'index.faiss + metadata.json' },

  // measurement
  { id: 'goldenset-evaluate', from: 'goldenset', to: 'evaluate', kind: 'data', label: '84 labelled cases' },
  { id: 'goldenset-hybrid', from: 'goldenset', to: 'hybrid', kind: 'support', label: 'BM25 only, to propose labels' },
  { id: 'evaluate-retrieve', from: 'evaluate', to: 'retrieve', kind: 'call', label: 'one configuration' },
  { id: 'evaluate-hybrid', from: 'evaluate', to: 'hybrid', kind: 'call', label: 'the fusion under test' },
  { id: 'probes-evaluate', from: 'probes', to: 'evaluate', kind: 'call', label: 'the scoring functions' },
  { id: 'probes-rerank', from: 'probes', to: 'rerank', kind: 'call', label: 'a candidate reranker' },
]

/* ---------------------------------------------------------------- flows
   Five paths that are worth walking end to end. Each is a real sequence: you
   can follow every step in the source. */
export const FLOWS: readonly ArchFlow[] = [
  {
    id: 'ask',
    name: 'Ask a question',
    payload: 'the question',
    summary:
      'The main loop: one question in, five cited passages and a confidence verdict out.',
    route: [
      'cli-api', 'api-retrieve', 'retrieve-hybrid', 'retrieve-rerank',
      'retrieve-diversify', 'retrieve-parent', 'api-abstain',
    ],
  },
  {
    id: 'inspect',
    name: 'Inspect a query',
    payload: 'the rankings',
    summary:
      'The same pipeline again, keeping what serving throws away — the trace behind the ' +
      'Retrieval Inspector.',
    route: ['serve-trace', 'trace-retrieve', 'trace-hybrid', 'trace-rerank', 'trace-abstain'],
  },
  {
    id: 'index',
    name: 'Build the index',
    payload: 'document text',
    summary:
      'A cold build: fetch the corpus, check each document, parse it, chunk on tokens, ' +
      'embed, write FAISS. 432 seconds from nothing.',
    route: [
      'fetch-ingest', 'ingest-health', 'ingest-loaders', 'loaders-ocr',
      'ingest-parsecache', 'ingest-embedcache', 'ingest-retrieve',
    ],
  },
  {
    id: 'reindex',
    name: 'Re-index',
    payload: 'one changed file',
    summary:
      'The same path with both caches warm. Nothing changed costs 0.76 s; one new document ' +
      'among twenty costs 18.8 s.',
    route: ['serve-ingest', 'ingest-parsecache', 'ingest-embedcache', 'ingest-retrieve'],
  },
  {
    id: 'measure',
    name: 'Score a change',
    payload: '84 labelled cases',
    summary:
      'What every default in this repo had to survive: run a configuration against the ' +
      'golden set and write the numbers the README quotes.',
    route: ['goldenset-evaluate', 'evaluate-retrieve', 'evaluate-hybrid', 'probes-evaluate'],
  },
]

/* ------------------------------------------------------------- assembly
   Geometry is derived here rather than written above, so a module that grows
   gets a taller building on the next sync without anyone editing this file. */
const measureOf = (id: string) => MEASURED[id] ?? { count: 0, loc: 0 }

const sized = AUTHORED.map((node) => {
  const measure = measureOf(node.id)
  const { archetype, params } = deriveArchetype(measure)
  return { node, measure, archetype, params, size: deriveSize(archetype, params, measure) }
})

const FOOTPRINTS = packLayout(
  sized.map((s) => ({ item: s.node.id, group: s.node.group, size: s.size })),
  GROUPS.map((g) => g.id),
)

export const NODES: readonly ArchNode[] = sized.map((s) => ({
  ...s.node,
  archetype: s.archetype,
  params: s.params,
  height: deriveHeight(s.measure),
  footprint: FOOTPRINTS.get(s.node.id)!,
  count: s.measure.count,
  loc: s.measure.loc,
}))

/** Each edge carries the ids of the flows that travel it, so the rail can light one. */
export const EDGES: readonly ArchEdge[] = EDGE_SPEC.map((edge) => ({
  ...edge,
  flowIds: FLOWS.filter((f) => f.route.includes(edge.id)).map((f) => f.id),
}))

export const INTRO = {
  title: 'rag-project',
  lede:
    'A retrieval system built from parts, where every default had to earn its place ' +
    'against a number — and four conclusions that had already been written up as results ' +
    'turned out to be wrong.',
  whatItDoes:
    'Answers questions from your own documents and cites exactly where each passage came ' +
    'from — page, slide, or spreadsheet row — and [[declines]] when the corpus cannot ' +
    'answer, rather than returning the closest topical match.',
  howItsBuilt:
    'Twenty-seven Python modules, no framework. The shape worth noticing is that ' +
    'measurement is a subsystem, not a script: the golden set, the harness and the ' +
    'diagnostic probes are as much of this repo as the pipeline they judge, and the ' +
    'pipeline is duplicated once on purpose so it can be watched without being slowed down.',
}

export const ARCHITECTURE = {
  groups: GROUPS,
  nodes: NODES,
  edges: EDGES,
  flows: FLOWS,
  intro: INTRO,
  unmapped: UNCLAIMED,
  repo: 'rag-project',
}
