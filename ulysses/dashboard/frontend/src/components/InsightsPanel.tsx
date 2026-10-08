import type { Analytics } from '../types'

function Bar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="bar-row">
      <div className="bar-row__label">{label}</div>
      <div className="bar-row__track">
        <div className="bar-row__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="bar-row__value">{pct}%</div>
    </div>
  )
}

export function InsightsPanel({ analytics }: { analytics: Analytics | null }) {
  if (!analytics) {
    return <div className="insights insights--loading">Loading insights…</div>
  }

  const { win_rate_by_category, win_rate_by_score_bucket, average_score_won_vs_lost } = analytics

  return (
    <div className="insights">
      <section className="insights__section">
        <h3>Win rate by category</h3>
        {Object.entries(win_rate_by_category).map(([key, value]) => (
          <Bar key={key} label={key} value={value} />
        ))}
      </section>

      <section className="insights__section">
        <h3>Win rate by score bucket</h3>
        {Object.entries(win_rate_by_score_bucket).map(([key, value]) => (
          <Bar key={key} label={key} value={value} />
        ))}
      </section>

      <section className="insights__section">
        <h3>Average score: won vs. lost</h3>
        <div className="insights__stat-row">
          <span className="score--green">Won: {average_score_won_vs_lost.won?.toFixed(1)}</span>
          <span className="score--red">Lost: {average_score_won_vs_lost.lost?.toFixed(1)}</span>
        </div>
      </section>

      <section className="insights__section">
        <h3>Connects spent per win</h3>
        <div className="insights__stat-row">
          <span>
            {analytics.average_connects_spent_per_win !== null
              ? analytics.average_connects_spent_per_win.toFixed(1)
              : 'Not enough recorded wins yet'}
          </span>
        </div>
      </section>

      <section className="insights__section">
        <h3>Scoring weight suggestions</h3>
        <ul>
          {analytics.scoring_weight_suggestions.map((suggestion) => (
            <li key={suggestion}>{suggestion}</li>
          ))}
        </ul>
      </section>
    </div>
  )
}
