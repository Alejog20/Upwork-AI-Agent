import { useEffect, useState } from 'react'
import { listJobs, type JobFilters } from '../api'
import type { JobSummary } from '../types'
import { FilterBar } from './FilterBar'
import { JobCard } from './JobCard'

export function JobFeed({ refreshKey }: { refreshKey: number }) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<JobFilters>({})

  async function reload(): Promise<void> {
    try {
      const result = await listJobs(filters)
      setJobs(result)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load jobs')
    }
  }

  useEffect(() => {
    reload()
  }, [refreshKey, filters])

  return (
    <div>
      <FilterBar filters={filters} onChange={setFilters} />
      {error ? (
        <div className="job-feed__error">{error}</div>
      ) : jobs.length === 0 ? (
        <div className="job-feed__empty">
          No jobs match these filters yet.
        </div>
      ) : (
        <div className="job-feed">
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} onChanged={reload} />
          ))}
        </div>
      )}
    </div>
  )
}
