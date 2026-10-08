export type JobStatus =
  | 'new'
  | 'notified'
  | 'drafted'
  | 'built'
  | 'skipped'
  | 'archived'
  | 'won'
  | 'lost'

export type Recommendation = 'apply_now' | 'review' | 'skip'

export type JobSource = 'email' | 'manual'

export interface JobSummary {
  id: string
  title: string
  url: string
  score: number
  category: string
  status: JobStatus
  source: JobSource
  posted_at: string
  seen_at: string
  budget?: string
  skills_required?: string[]
  client_hires?: number
  client_rating?: number | null
  payment_verified?: boolean
  proposals_count?: number | null
  red_flags?: string[]
  recommendation?: Recommendation
  best_repo_match?: string | null
  days_since_seen?: number
}

export interface JobDetail {
  job: Record<string, unknown>
  score: Record<string, unknown>
  proposal_drafts: string[]
  prototype_files: string[]
}

export interface Stats {
  total: number
  [status: string]: number
}

export interface Analytics {
  win_rate_by_category: Record<string, number>
  win_rate_by_score_bucket: Record<string, number>
  win_rate_by_red_flags: Record<string, number>
  average_score_won_vs_lost: Record<string, number>
  average_connects_spent_per_win: number | null
  scoring_weight_suggestions: string[]
}

export interface DashboardEvent {
  type: 'job_scored' | 'job_updated'
  job_id: string
}
