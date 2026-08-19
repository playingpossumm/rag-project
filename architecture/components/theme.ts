/**
 * The one file that knows about the host repo's design system.
 *
 * Every colour and type style the map uses passes through here, under a
 * semantic name. Point these at the repo's own tokens and the map re-themes
 * with the app -- including dark mode, because a CSS custom property changes
 * underneath a `var()` without anything re-rendering.
 *
 * Adapted to the Retrieval Inspector. The names on the right (`--panel`,
 * `--ink`, `--line`, `--series-1`) are the Inspector's own tokens, declared in
 * the page shell with the same light/dark blocks `ui/index.html` uses, so the
 * map follows the system theme exactly as the rest of the project does. The
 * accent is `--series-1`, the first slot of the validated categorical palette
 * -- reused here as a UI accent rather than as a data colour, which is the only
 * safe way to borrow from that palette: nothing on this map encodes a series.
 *
 * Fallbacks are kept so the bundle still renders if it is opened somewhere the
 * tokens are not declared.
 */

export const paint = {
  /** The page under everything, and the top face of every building. */
  surface: 'var(--panel, #ffffff)',
  /** Hairlines: the floor grid, cell borders, the quiet edges. */
  border: 'var(--line, #dfe3e9)',
  /** Building walls and inactive strokes. */
  structure: 'var(--faint, #8a94a1)',

  inkPrimary: 'var(--ink, #151a21)',
  inkSecondary: 'var(--muted, #5d6875)',
  inkTertiary: 'var(--faint, #8a94a1)',

  /** Selection, the active flow, the lit neighborhood. */
  accent: 'var(--series-1, #2a78d6)',
  /** The accent at wash strength, for filled plates and chips. */
  accentWash: 'var(--accent-wash, #dceafa)',
} as const

/**
 * Type. Three roles only: a title, running prose, and the mono label used for
 * codes, paths and chips. Anything a sentence goes in must not be the mono
 * one -- monospace is for names, not for reading.
 *
 * Serif headings, sans body, mono data: the pairing the UI brief asks for, and
 * the same three stacks the Inspector sets.
 */
export const type = {
  title: 'var(--serif, "Iowan Old Style", Charter, Cambria, "Palatino Linotype", Georgia, serif)',
  body: 'var(--sans, system-ui, -apple-system, "Segoe UI Variable Text", "Segoe UI", sans-serif)',
  mono: 'var(--mono, ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace)',
} as const

/** The house motion curve, and the two durations the map uses. */
export const motion = {
  ease: 'cubic-bezier(0.32, 0.72, 0, 1)',
  /** Enter and exit. */
  base: 200,
  /** Hover, which should feel immediate. */
  hover: 150,
} as const

/** Joins class names, dropping anything falsy. */
export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}
