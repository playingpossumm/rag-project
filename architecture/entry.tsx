/**
 * Mount point for the bundled map.
 *
 * This repo is Python and has no React app to host a route, so the honest
 * output is a standalone page: everything -- React, the map, the graph, the
 * measurements -- is bundled into `docs/architecture.html`, which opens from
 * the filesystem with no server and is also served at `/~/architecture` when
 * `src/serve.py` is running. The measurement script still runs independently,
 * so the drift counter keeps working; only the mounting is unusual.
 */
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import ArchitectureMap from './components/ArchitectureMap'
// keyframes.css is not imported here: the bundler inlines it into the page
// shell's <style> instead, so the standalone file needs no stylesheet request.
import { ARCHITECTURE } from './graph'
import { HISTORY } from './history.generated'

const host = document.getElementById('root')
if (!host) throw new Error('architecture: #root is missing from the page shell')

createRoot(host).render(
  <StrictMode>
    <ArchitectureMap data={{ ...ARCHITECTURE, history: HISTORY }} />
  </StrictMode>,
)
