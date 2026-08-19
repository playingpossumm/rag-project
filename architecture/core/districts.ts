import type { Group, ArchNode } from './types'
import type { Footprint, GridPt } from './iso'

/**
 * The neighborhoods: one plot of floor per group, named on the ground.
 *
 * Rects are *derived*, never authored — each is the bounding box of the
 * group's buildings plus padding. A hand-written rect would be a second claim
 * about where a group lives, and the moment a building moved the two would
 * disagree.
 *
 * The padding is deliberately asymmetric: the *front* edge gets extra room,
 * because that is where the plot plants its flag. The flag stands at the
 * front-left corner for a geometric reason rather than a stylistic one — a
 * building hides whatever lies on the floor behind it, so a marker at a
 * district's back edge is buried by that district's own towers. Planted at the
 * front, every building rises *away* from it.
 */

const PAD = 1.2
// 3.2 rather than 2.2: the flag is a banner extending right from the front-left
// corner, so a plot whose front row starts immediately at the pad had the label
// lying across its own buildings. Still well inside DISTRICT_GAP (6).
const PAD_FRONT = 3.2

export type District = {
  id: string
  label: string
  rect: Footprint
  /** Where the flagpole is planted: the plot's near-left corner. */
  flagAt: GridPt
  nodeIds: string[]
}

function derive(group: Group, nodes: readonly ArchNode[]): District | null {
  const members = nodes.filter((node) => node.group === group.id)
  if (members.length === 0) return null

  let gx0 = Infinity
  let gy0 = Infinity
  let gx1 = -Infinity
  let gy1 = -Infinity
  for (const { footprint: fp } of members) {
    gx0 = Math.min(gx0, fp.gx)
    gy0 = Math.min(gy0, fp.gy)
    gx1 = Math.max(gx1, fp.gx + fp.w)
    gy1 = Math.max(gy1, fp.gy + fp.d)
  }

  const rect: Footprint = {
    gx: gx0 - PAD,
    gy: gy0 - PAD,
    w: gx1 - gx0 + PAD * 2,
    d: gy1 - gy0 + PAD + PAD_FRONT,
  }

  return {
    id: group.id,
    label: group.label,
    rect,
    // Planted one cell diagonally forward of the plot's left corner. Advancing
    // gx and gy together moves the flag straight *down* the screen without
    // moving it sideways (x is a function of gx - gy), which is the only
    // direction that clears the front row: the banner runs rightward in screen
    // space, back across the plate, so widening the plot cannot get it out of
    // the way -- only dropping it below the front edge can.
    flagAt: { gx: rect.gx + 1.15, gy: rect.gy + rect.d + 0.45 },
    nodeIds: members.map((node) => node.id),
  }
}

/**
 * The neighborhoods a given set of buildings forms. Parameterised rather than
 * reading a module-level table, so a filtered view — one group, or one moment
 * in the repo's history — derives its own plots without a second code path.
 */
export function deriveDistricts(groups: readonly Group[], nodes: readonly ArchNode[]): District[] {
  return groups
    .map((group) => derive(group, nodes))
    .filter((district): district is District => district !== null)
}

export function districtIdOf(
  districts: readonly District[],
  nodeId: string | null | undefined,
): string | null {
  if (!nodeId) return null
  return districts.find((d) => d.nodeIds.includes(nodeId))?.id ?? null
}
