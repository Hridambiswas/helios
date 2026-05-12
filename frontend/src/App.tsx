import { useState, useEffect, useRef } from 'react'
import { AnimatePresence } from 'framer-motion'
import { VenomOverlay } from './components/VenomOverlay'
import { PromptPage } from './components/PromptPage'
import { ChatPage } from './components/ChatPage'
import { AuthModal } from './components/AuthModal'
import { MobileBottomNav } from './components/MobileBottomNav'
import { useAuth } from './hooks/useAuth'
import { useToast } from './hooks/useToast'
import { useConversations } from './hooks/useConversations'

export default function App() {
  const { user, loading, login, register, logout } = useAuth()
  const { toasts, add: addToast, remove: removeToast } = useToast()
  const {
    conversations, active, activeId,
    selectConversation, newConversation, deleteConversation,
    addUserMessage, addAssistantPlaceholder, updateMessage,
  } = useConversations(!!user)

  const [overlayDone, setOverlayDone] = useState(false)
  const [chatMode, setChatMode]       = useState(false)
  const [showAuth, setShowAuth]       = useState(false)

  // App owns the overlay timer — no prop-callback dance
  useEffect(() => {
    const t = setTimeout(() => setOverlayDone(true), 3200)
    return () => clearTimeout(t)
  }, [])

  // Handle OAuth redirect
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const at = p.get('access_token'), rt = p.get('refresh_token')
    if (p.get('oauth_error')) {
      addToast('OAuth sign-in failed', 'error')
      window.history.replaceState({}, '', window.location.pathname)
      return
    }
    if (at && rt) {
      localStorage.setItem('access_token', at)
      localStorage.setItem('refresh_token', rt)
      window.dispatchEvent(new Event('helios:oauth-login'))
      window.history.replaceState({}, '', window.location.pathname)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const openChat = (q?: string) => {
    let id = activeId
    if (!id || !active || active.messages.length > 0) id = newConversation()
    if (q && id) setTimeout(() => addUserMessage(id!, q), 50)
    setChatMode(true)
    window.scrollTo(0, 0)
  }

  // While auth is resolving keep the overlay up (don't flash blank)
  if (loading) {
    return (
      <div style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: '#000',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 24,
      }}>
        <h1 style={{
          fontFamily: '"Inter Tight", "Montserrat", sans-serif',
          fontWeight: 900, fontSize: 'clamp(72px, 16vw, 200px)',
          letterSpacing: '-0.045em', color: '#fff', lineHeight: 1,
        }}>
          HELIOS
        </h1>
        <p style={{
          fontFamily: '"IBM Plex Mono", monospace', fontSize: 9,
          letterSpacing: '0.5em', textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.3)',
        }}>
          Loading
        </p>
      </div>
    )
  }

  return (
    <>
      <div className="scanline" />

      <AnimatePresence mode="wait">
        {!overlayDone ? (
          <VenomOverlay key="overlay" />
        ) : !chatMode ? (
          <PromptPage
            key="prompt"
            onSubmit={q => openChat(q)}
            user={user}
            onAuthClick={() => setShowAuth(true)}
          />
        ) : (
          <ChatPage
            key="chat"
            conversations={conversations}
            activeId={activeId}
            active={active}
            onSelect={selectConversation}
            onNew={newConversation}
            onDelete={deleteConversation}
            onAddUserMessage={addUserMessage}
            onAddPlaceholder={addAssistantPlaceholder}
            onUpdateMessage={updateMessage}
            onNeedConversation={newConversation}
            user={user}
            onAuthClick={() => setShowAuth(true)}
            onLogout={() => { logout(); addToast('Signed out', 'info') }}
            onBack={() => setChatMode(false)}
            onToast={(m, t) => addToast(m, (t as 'info' | 'success' | 'error') ?? 'info')}
          />
        )}
      </AnimatePresence>

      <MobileBottomNav
        chatMode={chatMode}
        onHome={() => setChatMode(false)}
        onChat={() => openChat()}
        onUpload={() => { if (!chatMode) setChatMode(true) }}
        user={user}
        onAuthClick={() => setShowAuth(true)}
        onLogout={() => { logout(); addToast('Signed out', 'info') }}
      />

      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onLogin={async (u, p) => { await login(u, p); addToast(`Welcome back, ${u}!`, 'success') }}
          onRegister={async (u, e, p) => { await register(u, e, p); addToast(`Welcome, ${u}!`, 'success') }}
        />
      )}

      {/* Toasts */}
      <div style={{
        position: 'fixed', top: 16, right: 16, zIndex: 9990,
        display: 'flex', flexDirection: 'column', gap: 8, pointerEvents: 'none',
      }}>
        {toasts.map(t => (
          <div key={t.id} style={{
            pointerEvents: 'auto', display: 'flex', alignItems: 'center', gap: 12,
            padding: '10px 16px',
            background: 'rgba(0,0,0,0.9)',
            border: '1px solid rgba(255,255,255,0.08)',
            backdropFilter: 'blur(16px)',
            fontFamily: '"IBM Plex Mono", monospace', fontSize: 11,
            color: t.type === 'error' ? 'rgba(255,120,120,0.9)' : 'rgba(255,255,255,0.65)',
            maxWidth: 320,
          }}>
            <span style={{ flex: 1 }}>{t.message}</span>
            <button onClick={() => removeToast(t.id)} style={{
              background: 'none', border: 'none',
              color: 'rgba(255,255,255,0.3)', cursor: 'pointer', fontSize: 14,
            }}>×</button>
          </div>
        ))}
      </div>
    </>
  )
}
