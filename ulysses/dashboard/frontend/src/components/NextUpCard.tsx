import { useEffect, useState } from 'react'
import { listJobs } from '../api'
import type { JobSummary } from '../types'
import { JobCard } from './JobCard'

export function NextUpCard({ refreshKey }: { refreshKey: number }) {
  const [queue, setQueue] = useState<JobSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  async function reload(): Promise<void> {
    try {
      const jobs = await listJobs({ status: 'new' })
      jobs.sort((a, b) => b.score - a.score)
      setQueue(jobs)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load the queue')
    }
  }

  useEffect(() => {
    reload()
  }, [refreshKey])

  if (error) {
    return <div className="job-feed__error">{error}</div>
  }

  if (queue.length === 0) {
    return (
      <div className="job-feed__empty">
        Nothing new waiting on a decision. New jobs land here the moment Scout scores them.
      </div>
    )
  }

  const current = queue[0]

  return (
    <div className="next-up">
      <p className="next-up__hint">
        {queue.length} job{queue.length === 1 ? '' : 's'} waiting -- highest score first. Act on
        this one and the next appears automatically.
      </p>
      <div className="next-up__card">
        <JobCard key={current.id} job={current} onChanged={reload} />
      </div>
    </div>
  )
}
