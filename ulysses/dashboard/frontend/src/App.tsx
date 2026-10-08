import { useEffect, useState } from 'react'
import './App.css'
import { connectEvents, getAnalytics, getStats } from './api'
import { ChatView } from './components/ChatView'
import { FollowUpView } from './components/FollowUpView'
import { InsightsPanel } from './components/InsightsPanel'
import { JobFeed } from './components/JobFeed'
import { NextUpCard } from './components/NextUpCard'
import { StatsBar } from './components/StatsBar'
import { GENERAL_CHAT_THREAD_ID, type Analytics, type Stats } from './types'

type Tab = 'feed' | 'next-up' | 'followup' | 'chat' | 'insights'

export default function App() {
  const [tab, setTab] = useState<Tab>('feed')
  const [stats, setStats] = useState<Stats | null>(null)
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [connected, setConnected] = useState(false)
  const [pulsing, setPulsing] = useState(false)

  useEffect(() => {
    getStats().then(setStats).catch(() => undefined)
  }, [refreshKey])

  useEffect(() => {
    if (tab === 'insights') {
      getAnalytics().then(setAnalytics).catch(() => undefined)
    }
  }, [tab, refreshKey])

  useEffect(() => {
    const disconnect = connectEvents(
      () => {
        setPulsing(true)
        setRefreshKey((key) => key + 1)
        setTimeout(() => setPulsing(false), 1500)
      },
      setConnected,
    )
    return disconnect
  }, [])

  return (
    <div className="app">
      <header className="app__header">
        <img src="/ulysses-icon.png" alt="" className="app__icon" />
        <h1>Ulysses</h1>
        <span
          className={`live-dot ${connected ? 'live-dot--active' : ''} ${pulsing ? 'live-dot--pulse' : ''}`}
          title={connected ? 'Connected — listening for new jobs' : 'Disconnected'}
        />
        <nav className="app__tabs">
          <button className={tab === 'feed' ? 'tab tab--active' : 'tab'} onClick={() => setTab('feed')}>
            Job Feed
          </button>
          <button
            className={tab === 'next-up' ? 'tab tab--active' : 'tab'}
            onClick={() => setTab('next-up')}
          >
            Next Up
          </button>
          <button
            className={tab === 'followup' ? 'tab tab--active' : 'tab'}
            onClick={() => setTab('followup')}
          >
            Needs Follow-Up
          </button>
          <button
            className={tab === 'chat' ? 'tab tab--active' : 'tab'}
            onClick={() => setTab('chat')}
          >
            Chat
          </button>
          <button
            className={tab === 'insights' ? 'tab tab--active' : 'tab'}
            onClick={() => setTab('insights')}
          >
            Insights
          </button>
        </nav>
      </header>

      <StatsBar stats={stats} />

      <main className="app__main">
        {tab === 'feed' && <JobFeed refreshKey={refreshKey} />}
        {tab === 'next-up' && <NextUpCard refreshKey={refreshKey} />}
        {tab === 'followup' && <FollowUpView refreshKey={refreshKey} />}
        {tab === 'chat' && <ChatView threadId={GENERAL_CHAT_THREAD_ID} title="Ulysses Copilot" />}
        {tab === 'insights' && <InsightsPanel analytics={analytics} />}
      </main>
    </div>
  )
}
