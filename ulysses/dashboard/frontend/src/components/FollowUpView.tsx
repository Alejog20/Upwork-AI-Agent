import { useEffect, useState } from 'react'
import { listStaleJobs } from '../api'
import type { JobSummary } from '../types'
import { JobCard } from './JobCard'

export function FollowUpView({ refreshKey }: { refreshKey: number }) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  async function reload(): Promise<void> {
    try {
      setJobs(await listStaleJobs())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load follow-ups')
    }
  }

  useEffect(() => {
    reload()
  }, [refreshKey])

  if (error) {
    return <div className="job-feed__error">{error}</div>
  }

  if (jobs.length === 0) {
    return (
      <div className="job-feed__empty">
        Nothing's gone quiet -- every notified/drafted/built job has moved on or is still fresh.
      </div>
    )
  }

  return (
    <div className="job-feed">
      <p className="followup__hint">
        Applied, drafted, or built more than 5 days ago with no Won/Lost/Skip/Archive recorded --
        worth a follow-up or a decision either way.
      </p>
      {jobs.map((job) => (
        <JobCard key={job.id} job={job} onChanged={reload} />
      ))}
    </div>
  )
}
