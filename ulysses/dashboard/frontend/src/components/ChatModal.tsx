import { ChatView } from './ChatView'

export function ChatModal({
  threadId,
  title,
  onClose,
}: {
  threadId: string
  title: string
  onClose: () => void
}) {
  return (
    <div className="chat-modal-backdrop" onClick={onClose}>
      <div className="chat-modal" onClick={(e) => e.stopPropagation()}>
        <button className="chat-modal__close" onClick={onClose}>
          ✕ Close
        </button>
        <ChatView threadId={threadId} title={title} />
      </div>
    </div>
  )
}
