import type { Stats } from '../types'

const LABELS: Record<string, string> = {
  total: 'Total',
  new: 'New',
  notified: 'Notified',
  drafted: 'Drafted',
  built: 'Built',
  skipped: 'Skipped',
  archived: 'Archived',
  won: 'Won',
  lost: 'Lost',
}

const ORDER = ['total', 'new', 'notified', 'drafted', 'built', 'won', 'lost', 'skipped', 'archived']

export function StatsBar({ stats }: { stats: Stats | null }) {
  if (!stats) {
    return <div className="stats-bar stats-bar--loading">Loading stats…</div>
  }

  const keys = ORDER.filter((key) => key in stats)

  return (
    <div className="stats-bar">
      {keys.map((key) => (
        <div key={key} className="stats-bar__tile">
          <div className="stats-bar__value">{stats[key]}</div>
          <div className="stats-bar__label">{LABELS[key] ?? key}</div>
        </div>
      ))}
    </div>
  )
}
