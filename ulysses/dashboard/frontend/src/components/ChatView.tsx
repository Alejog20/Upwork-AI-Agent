import { useEffect, useRef, useState } from 'react'
import { clearChatThread, connectChat, getChatMessages, type ChatConnection } from '../api'
import type { ChatMessage } from '../types'

function formatTime(isoDate: string): string {
  return new Date(isoDate).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function ChatView({ threadId, title }: { threadId: string; title?: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [input, setInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const connectionRef = useRef<ChatConnection | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    setMessages([])
    setStreamingText('')
    setError(null)

    getChatMessages(threadId)
      .then((history) => {
        if (!cancelled) setMessages(history)
      })
      .catch(() => undefined)

    const connection = connectChat(threadId, {
      onDelta: (text) => setStreamingText((prev) => prev + text),
      onDone: () => {
        setStreamingText((prev) => {
          if (prev) {
            setMessages((msgs) => [
              ...msgs,
              { role: 'assistant', content: prev, created_at: new Date().toISOString() },
            ])
          }
          return ''
        })
        setIsStreaming(false)
      },
      onError: (detail) => {
        setError(detail)
        setIsStreaming(false)
        setStreamingText('')
      },
    })
    connectionRef.current = connection

    return () => {
      cancelled = true
      connection.disconnect()
    }
  }, [threadId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, streamingText])

  function send(): void {
    const text = input.trim()
    if (!text || isStreaming) return
    setMessages((msgs) => [
      ...msgs,
      { role: 'user', content: text, created_at: new Date().toISOString() },
    ])
    setInput('')
    setError(null)
    setIsStreaming(true)
    connectionRef.current?.send(text)
  }

  async function startNewConversation(): Promise<void> {
    await clearChatThread(threadId)
    setMessages([])
    setStreamingText('')
    setError(null)
  }

  return (
    <div className="chat-view">
      <div className="chat-view__header">
        <span className="chat-view__title">{title ?? 'Chat'}</span>
        <button className="chat-view__new" onClick={startNewConversation}>
          New conversation
        </button>
      </div>

      <div className="chat-view__messages" ref={scrollRef}>
        {messages.length === 0 && !streamingText && (
          <div className="chat-view__empty">
            Ask about your queue, strategy, or this job -- whatever's on your mind.
          </div>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`chat-bubble chat-bubble--${message.role}`}>
            <div className="chat-bubble__content">{message.content}</div>
            <div className="chat-bubble__time">{formatTime(message.created_at)}</div>
          </div>
        ))}
        {(streamingText || isStreaming) && (
          <div className="chat-bubble chat-bubble--assistant">
            <div className="chat-bubble__content">
              {streamingText}
              <span className="chat-cursor" />
            </div>
          </div>
        )}
      </div>

      {error && <div className="chat-view__error">{error}</div>}

      <div className="chat-view__input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          placeholder="Message Ulysses..."
          rows={2}
        />
        <button disabled={isStreaming || !input.trim()} onClick={send}>
          Send
        </button>
      </div>
    </div>
  )
}
