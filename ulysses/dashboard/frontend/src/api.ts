import type {
  Analytics,
  ChatMessage,
  DashboardEvent,
  JobDetail,
  JobSource,
  JobSummary,
  Stats,
} from './types'

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

export interface JobFilters {
  minScore?: number
  category?: string
  status?: string
  source?: JobSource
  maxProposals?: number
  hasRedFlags?: boolean
}

export async function listJobs(filters: JobFilters = {}): Promise<JobSummary[]> {
  const params = new URLSearchParams()
  if (filters.minScore !== undefined) params.set('min_score', String(filters.minScore))
  if (filters.category) params.set('category', filters.category)
  if (filters.status) params.set('status', filters.status)
  if (filters.source) params.set('source', filters.source)
  if (filters.maxProposals !== undefined) params.set('max_proposals', String(filters.maxProposals))
  if (filters.hasRedFlags !== undefined) params.set('has_red_flags', String(filters.hasRedFlags))
  const query = params.toString()
  const response = await fetch(`/api/jobs${query ? `?${query}` : ''}`)
  return parseOrThrow<JobSummary[]>(response)
}

export async function listStaleJobs(days = 5): Promise<JobSummary[]> {
  const response = await fetch(`/api/jobs/stale?days=${days}`)
  return parseOrThrow<JobSummary[]>(response)
}

export async function getJobDetail(jobId: string): Promise<JobDetail> {
  const response = await fetch(`/api/jobs/${jobId}`)
  return parseOrThrow<JobDetail>(response)
}

export async function getStats(): Promise<Stats> {
  const response = await fetch('/api/stats')
  return parseOrThrow<Stats>(response)
}

export async function getAnalytics(): Promise<Analytics> {
  const response = await fetch('/api/analytics')
  return parseOrThrow<Analytics>(response)
}

export async function draftJob(jobId: string): Promise<{ content: string }> {
  const response = await fetch(`/api/jobs/${jobId}/draft`, { method: 'POST' })
  return parseOrThrow(response)
}

export async function saveDraft(jobId: string, content: string): Promise<{ content: string }> {
  const response = await fetch(`/api/jobs/${jobId}/draft`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  return parseOrThrow(response)
}

export async function buildJob(jobId: string): Promise<{ zip_filename: string; readme_md: string }> {
  const response = await fetch(`/api/jobs/${jobId}/build`, { method: 'POST' })
  return parseOrThrow(response)
}

export async function skipJob(jobId: string): Promise<JobSummary> {
  const response = await fetch(`/api/jobs/${jobId}/skip`, { method: 'POST' })
  return parseOrThrow<JobSummary>(response)
}

export async function archiveJob(jobId: string): Promise<JobSummary> {
  const response = await fetch(`/api/jobs/${jobId}/archive`, { method: 'POST' })
  return parseOrThrow<JobSummary>(response)
}

export async function recordOutcome(
  jobId: string,
  outcome: {
    won: boolean
    contract_value_usd?: number | null
    connects_spent?: number | null
    note?: string | null
  },
): Promise<{
  won: boolean
  contract_value_usd: number | null
  connects_spent: number | null
  note: string | null
}> {
  const response = await fetch(`/api/jobs/${jobId}/outcome`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(outcome),
  })
  return parseOrThrow(response)
}

export function prototypeZipUrl(jobId: string): string {
  return `/api/jobs/${jobId}/prototype.zip`
}

export async function getChatMessages(threadId: string): Promise<ChatMessage[]> {
  const response = await fetch(`/api/chat/${threadId}/messages`)
  return parseOrThrow<ChatMessage[]>(response)
}

export async function clearChatThread(threadId: string): Promise<void> {
  const response = await fetch(`/api/chat/${threadId}/messages`, { method: 'DELETE' })
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`)
  }
}

export interface ChatStreamHandlers {
  onDelta: (text: string) => void
  onDone: () => void
  onError: (detail: string) => void
}

export interface ChatConnection {
  send: (message: string) => void
  disconnect: () => void
}

/**
 * Opens a streaming chat WebSocket for one thread and reconnects with
 * backoff if it drops -- same self-healing shape as `connectEvents`, but
 * also exposes `send` since the caller actively pushes messages over it
 * rather than only listening.
 */
export function connectChat(threadId: string, handlers: ChatStreamHandlers): ChatConnection {
  let socket: WebSocket | null = null
  let retryDelayMs = 1000
  let stopped = false

  const connect = (): void => {
    if (stopped) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/chat/${threadId}`)

    socket.onmessage = (message) => {
      const data = JSON.parse(message.data) as { type: string; text?: string; detail?: string }
      if (data.type === 'delta' && data.text) {
        handlers.onDelta(data.text)
      } else if (data.type === 'done') {
        handlers.onDone()
      } else if (data.type === 'error') {
        handlers.onError(data.detail ?? 'Something went wrong.')
      }
    }
    socket.onopen = () => {
      retryDelayMs = 1000
    }
    socket.onclose = () => {
      if (stopped) return
      setTimeout(connect, retryDelayMs)
      retryDelayMs = Math.min(retryDelayMs * 2, 15000)
    }
  }

  connect()

  return {
    send: (message: string) => {
      socket?.send(JSON.stringify({ message }))
    },
    disconnect: () => {
      stopped = true
      socket?.close()
    },
  }
}

/**
 * Opens the live-events WebSocket and reconnects with backoff if it drops --
 * the dashboard should keep self-healing across a brief backend restart
 * without the user needing to reload the page.
 */
export function connectEvents(
  onEvent: (event: DashboardEvent) => void,
  onConnectionChange?: (connected: boolean) => void,
): () => void {
  let socket: WebSocket | null = null
  let retryDelayMs = 1000
  let stopped = false

  const connect = (): void => {
    if (stopped) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/events`)

    socket.onmessage = (message) => {
      onEvent(JSON.parse(message.data) as DashboardEvent)
    }
    socket.onopen = () => {
      retryDelayMs = 1000
      onConnectionChange?.(true)
    }
    socket.onclose = () => {
      onConnectionChange?.(false)
      if (stopped) return
      setTimeout(connect, retryDelayMs)
      retryDelayMs = Math.min(retryDelayMs * 2, 15000)
    }
  }

  connect()

  return () => {
    stopped = true
    socket?.close()
  }
}
