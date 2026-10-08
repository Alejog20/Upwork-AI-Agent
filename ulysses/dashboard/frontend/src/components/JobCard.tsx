import { useState } from 'react'
import {
  archiveJob,
  buildJob,
  draftJob,
  prototypeZipUrl,
  recordOutcome,
  saveDraft,
  skipJob,
} from '../api'
import type { JobSummary } from '../types'
import { ChatModal } from './ChatModal'

const SCORE_COMPONENTS: Array<{ key: keyof JobSummary; label: string }> = [
  { key: 'freshness_score', label: 'Freshness' },
  { key: 'proposal_score', label: 'Proposals' },
  { key: 'client_score', label: 'Client' },
  { key: 'skill_score', label: 'Skills' },
  { key: 'budget_score', label: 'Budget' },
]

const RECOMMENDATION_LABEL: Record<string, string> = {
  apply_now: 'APPLY NOW',
  review: 'REVIEW',
  skip: 'SKIP',
}

const RECOMMENDATION_CLASS: Record<string, string> = {
  apply_now: 'pill--green',
  review: 'pill--yellow',
  skip: 'pill--red',
}

function scoreClass(score: number): string {
  if (score >= 75) return 'score--green'
  if (score >= 50) return 'score--yellow'
  return 'score--red'
}

function draftButtonLabel(busy: boolean, hasDraft: boolean): string {
  if (busy) return hasDraft ? 'Regenerating…' : 'Drafting…'
  return hasDraft ? '🔄 Regenerate' : '📝 Draft'
}

function formatPostedAgo(isoDate: string): string {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(isoDate).getTime()) / 60000))
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

export function JobCard({
  job,
  onChanged,
}: {
  job: JobSummary
  onChanged: () => void
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [draftText, setDraftText] = useState<string | null>(null)
  const [draftSaved, setDraftSaved] = useState(false)
  const [builtZip, setBuiltZip] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(false)

  async function run(action: string, work: () => Promise<void>): Promise<void> {
    setBusy(action)
    setError(null)
    try {
      await work()
      onChanged()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="job-card">
      <div className="job-card__header">
        <span className={`job-card__score ${scoreClass(job.score)}`}>
          🎯 {Math.round(job.score)}
        </span>
        {job.recommendation && (
          <span className={`pill ${RECOMMENDATION_CLASS[job.recommendation] ?? ''}`}>
            {RECOMMENDATION_LABEL[job.recommendation] ?? job.recommendation}
          </span>
        )}
        <span className="job-card__status">{job.status}</span>
        {job.days_since_seen !== undefined && (
          <span className="pill pill--yellow">{job.days_since_seen}d, no resolution</span>
        )}
      </div>

      <h3 className="job-card__title">{job.title}</h3>

      <div className="job-card__meta">
        <span>💰 {job.budget ?? 'Budget not listed'}</span>
        <span>⏱ {formatPostedAgo(job.posted_at)}</span>
        <span>
          {job.payment_verified ? '✅ Payment verified' : '⚠️ Payment not verified'}
        </span>
      </div>

      {job.skills_required && job.skills_required.length > 0 && (
        <div className="job-card__skills">
          {job.skills_required.map((skill) => (
            <span key={skill} className="chip">
              {skill}
            </span>
          ))}
        </div>
      )}

      {job.best_repo_match && (
        <div className="job-card__repo">📁 Best repo match: {job.best_repo_match}</div>
      )}

      {job.red_flags && job.red_flags.length > 0 && (
        <div className="job-card__flags">⚠️ {job.red_flags.join(', ')}</div>
      )}

      <div className="job-card__actions">
        {job.url.startsWith('http') ? (
          <a href={job.url} target="_blank" rel="noreferrer" className="button button--primary">
            Apply on Upwork ↗
          </a>
        ) : (
          <span className="button button--disabled" title="This job has no captured Upwork link">
            No Upwork link
          </span>
        )}
        <button
          disabled={busy !== null}
          onClick={() =>
            run('draft', async () => {
              const result = await draftJob(job.id)
              setDraftText(result.content)
              setDraftSaved(false)
            })
          }
        >
          {draftButtonLabel(busy === 'draft', draftText !== null)}
        </button>
        <button
          disabled={busy !== null}
          onClick={() =>
            run('build', async () => {
              const result = await buildJob(job.id)
              setBuiltZip(result.zip_filename)
            })
          }
        >
          {busy === 'build' ? 'Building…' : '🛠 Build'}
        </button>
        <button disabled={busy !== null} onClick={() => run('skip', () => skipJob(job.id).then(() => undefined))}>
          ⏭ Skip
        </button>
        <button
          disabled={busy !== null}
          onClick={() => run('archive', () => archiveJob(job.id).then(() => undefined))}
        >
          📁 Archive
        </button>
        <button
          disabled={busy !== null}
          onClick={() =>
            run('won', async () => {
              const raw = window.prompt('Connects spent on this one? (optional)')
              const connects = raw && raw.trim() !== '' ? Number(raw) : undefined
              await recordOutcome(job.id, { won: true, connects_spent: connects })
            })
          }
        >
          🏆 Won
        </button>
        <button
          disabled={busy !== null}
          onClick={() => run('lost', () => recordOutcome(job.id, { won: false }).then(() => undefined))}
        >
          ✗ Lost
        </button>
        <button onClick={() => setChatOpen(true)}>💬 Ask about this</button>
      </div>

      {error && <div className="job-card__error">{error}</div>}

      {job.freshness_score !== undefined && (
        <details className="job-card__breakdown">
          <summary>Score breakdown</summary>
          <div className="job-card__breakdown-grid">
            {SCORE_COMPONENTS.map(({ key, label }) => (
              <span key={key}>
                {label}: {Math.round(job[key] as number)}
              </span>
            ))}
          </div>
        </details>
      )}

      {draftText !== null && (
        <details className="job-card__result" open>
          <summary>Proposal draft</summary>
          <textarea
            className="job-card__draft-editor"
            value={draftText}
            onChange={(e) => {
              setDraftText(e.target.value)
              setDraftSaved(false)
            }}
            rows={10}
          />
          <div className="job-card__draft-actions">
            <button
              onClick={() => {
                navigator.clipboard.writeText(draftText)
              }}
            >
              📋 Copy
            </button>
            <button
              disabled={busy !== null}
              onClick={() =>
                run('save-draft', async () => {
                  await saveDraft(job.id, draftText)
                  setDraftSaved(true)
                })
              }
            >
              {busy === 'save-draft' ? 'Saving…' : draftSaved ? '✅ Saved' : '💾 Save edit'}
            </button>
          </div>
        </details>
      )}

      {builtZip && (
        <details className="job-card__result" open>
          <summary>Demo prototype built</summary>
          <a href={prototypeZipUrl(job.id)}>Download {builtZip}</a>
        </details>
      )}

      {chatOpen && (
        <ChatModal threadId={job.id} title={job.title} onClose={() => setChatOpen(false)} />
      )}
    </div>
  )
}
