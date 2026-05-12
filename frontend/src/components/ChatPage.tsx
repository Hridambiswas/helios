import { useState } from 'react'
import { motion } from 'framer-motion'
import { ChatView } from './ChatView'
import { ChatSidebar } from './ChatSidebar'
import { UploadPanel } from './UploadPanel'
import type { User } from '../hooks/useAuth'
import type { Conversation } from '../hooks/useConversations'

interface Props {
  conversations:    Conversation[]
  activeId:         string | null
  active:           Conversation | null | undefined
  onSelect:         (id: string) => void
  onNew:            () => string
  onDelete:         (id: string) => void
  onAddUserMessage: (id: string, content: string) => string
  onAddPlaceholder: (id: string) => string
  onUpdateMessage:  (id: string, msgId: string, updates: Partial<Conversation['messages'][0]>) => void
  onNeedConversation: () => string
  user:             User | null
  onAuthClick:      () => void
  onLogout:         () => void
  onBack:           () => void
  onToast:          (msg: string, type?: string) => void
}

export function ChatPage({
  conversations, activeId, active,
  onSelect, onNew, onDelete,
  onAddUserMessage, onAddPlaceholder, onUpdateMessage, onNeedConversation,
  user, onAuthClick, onLogout, onBack, onToast,
}: Props) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.innerWidth < 640)
  const [showUpload, setShowUpload] = useState(false)

  return (
    <motion.div
      initial={{ clipPath: 'circle(0% at 50% 55%)', opacity: 0 }}
      animate={{ clipPath: 'circle(150% at 50% 55%)', opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.75, ease: [0.16, 1, 0.3, 1] }}
      className="venom-chat"
      style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#000' }}
    >
      {/* Sidebar */}
      <ChatSidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={onSelect}
        onNew={onNew}
        onDelete={id => { onDelete(id); onToast('Conversation deleted', 'info') }}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(v => !v)}
        user={user}
        onAuthClick={onAuthClick}
        onLogout={() => { onLogout(); onToast('Signed out', 'info') }}
        onUploadClick={() => setShowUpload(v => !v)}
      />

      {/* Main column */}
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>

        {/* Top bar */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          height: 56, padding: '0 20px', flexShrink: 0,
          borderBottom: '1px solid rgba(255,255,255,0.04)',
          background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(12px)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
            <button
              onClick={onBack}
              style={{
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: 10, letterSpacing: '0.22em',
                textTransform: 'uppercase',
                color: 'rgba(255,255,255,0.26)',
                background: 'none', border: 'none', cursor: 'none',
                transition: 'color 0.2s',
              }}
              onMouseEnter={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.8)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.26)')}
            >
              ← Home
            </button>
            {active?.title && (
              <span style={{
                fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                color: 'rgba(255,255,255,0.15)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                maxWidth: 240,
              }}>
                {active.title}
              </span>
            )}
          </div>

          <div>
            {user ? (
              <span style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 10, color: 'rgba(255,255,255,0.20)' }}>
                {user.username}
              </span>
            ) : (
              <button
                onClick={onAuthClick}
                style={{
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                  letterSpacing: '0.18em', textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.42)', background: 'none', border: 'none', cursor: 'none',
                  transition: 'color 0.2s',
                }}
                onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.42)')}
              >
                Sign in
              </button>
            )}
          </div>
        </div>

        {/* Upload panel */}
        {showUpload && (
          <UploadPanel
            isLoggedIn={!!user}
            onAuthClick={onAuthClick}
            onClose={() => setShowUpload(false)}
          />
        )}

        {/* Chat area */}
        <div style={{ flex: 1, minHeight: 0 }}>
          <ChatView
            conversation={active ?? null}
            isLoggedIn={!!user}
            onAuthRequired={onAuthClick}
            onAddUserMessage={onAddUserMessage}
            onAddPlaceholder={onAddPlaceholder}
            onUpdateMessage={onUpdateMessage}
            onNeedConversation={onNeedConversation}
          />
        </div>
      </div>
    </motion.div>
  )
}
