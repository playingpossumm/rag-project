#!/usr/bin/env node
/**
 * Bundles the architecture map into one self-contained page.
 *
 * `docs/architecture.html` is committed, because the whole value of this thing
 * is that someone can open it -- from the repo, from a checkout, from a link --
 * without installing Node. The build is reproducible from source at any time;
 * the artifact is convenience, not the source of truth.
 *
 * No CDN, no external stylesheet, no font request. That is the same rule
 * `ui/index.html` follows, and for the same reason: the corpus this project
 * indexes may be private, and a page about it should not phone anywhere.
 *
 * The design tokens below are the Inspector's own, declared here rather than
 * imported so the file stands alone. `components/theme.ts` maps every semantic
 * name in the map onto them, which is the only place the two are coupled.
 */
import { build } from 'esbuild'
import { readFileSync, writeFileSync } from 'node:fs'

const OUT = 'docs/architecture.html'

const result = await build({
  entryPoints: ['architecture/entry.tsx'],
  bundle: true,
  minify: true,
  format: 'iife',
  target: ['es2020'],
  jsx: 'automatic',
  loader: { '.css': 'text' },
  define: { 'process.env.NODE_ENV': '"production"' },
  write: false,
  logLevel: 'warning',
})

// With `write:false` and no outfile, esbuild names the single output
// `<stdout>` -- matching on a .js suffix silently yields an empty bundle.
const js = result.outputFiles[0]?.text ?? ''
if (js.length < 1000) throw new Error('architecture: bundle came back empty')
const css = readFileSync('architecture/components/keyframes.css', 'utf8')

/* The Inspector's tokens, both themes, following the system by default with an
   explicit override honoured either way -- identical in structure to the blocks
   in ui/index.html so the two pages cannot drift on what "dark" means. */
const TOKENS = `
:root{
  --ground:#F6F7F9; --panel:#FFFFFF; --sunk:#EDEFF3;
  --ink:#151A21; --muted:#5D6875; --faint:#8A94A1;
  --line:#DFE3E9; --line-soft:#EAEDF1;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --accent-wash:#DCEAFA;
  --serif:"Iowan Old Style",Charter,Cambria,"Palatino Linotype",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI Variable Text","Segoe UI",sans-serif;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1216; --panel:#151A20; --sunk:#1B2128;
  --ink:#E7EBEF; --muted:#95A0AD; --faint:#6C7885;
  --line:#242C35; --line-soft:#1E252D;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --accent-wash:#17293E;
}}
:root[data-theme="dark"]{
  --ground:#0E1216; --panel:#151A20; --sunk:#1B2128;
  --ink:#E7EBEF; --muted:#95A0AD; --faint:#6C7885;
  --line:#242C35; --line-soft:#1E252D;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --accent-wash:#17293E;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--panel);color:var(--ink);
  font-family:var(--sans);overflow:hidden}
select{font:inherit}
`

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Architecture — rag-project</title>
<style>${TOKENS}${css}</style>
</head>
<body>
<div id="root"></div>
<script>${js}</script>
</body>
</html>
`

writeFileSync(OUT, html)
const kb = (Buffer.byteLength(html) / 1024).toFixed(0)
console.log(`architecture — bundled ${OUT} (${kb} kB, self-contained)`)
