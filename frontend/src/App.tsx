import { useState, useEffect } from 'react'
import { AnimatePresence } from 'framer-motion'
import { CustomCursor } from './components/CustomCursor'
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

  // Timer fires once — empty deps, no dependency on any prop/state
  useEffect(() => {
    const t = setTimeout(() => setOverlayDone(true), 3200)
    return () => clearTimeout(t)
  }, [])

  // OAuth token from query string
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    if (p.get('oauth_error')) {
      addToast('OAuth sign-in failed', 'error')
      window.history.replaceState({}, '', window.location.pathname)
      return
    }
    const at = p.get('access_token'), rt = p.get('refresh_token')
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

  // Auth loading state — cursor still renders
  if (loading) {
    return (
      <>
        <CustomCursor />
        <div style={{
          position: 'fixed', inset: 0,
          background: '#000',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <h1 style={{
            fontFamily: '"Montserrat", sans-serif', fontWeight: 900,
            fontSize: 'clamp(72px, 16vw, 200px)', letterSpacing: '-0.045em', color: '#fff',
          }}>HELIOS</h1>
        </div>
      </>
    )
  }

  return (
    <>
      {/* Custom metaball cursor — always on top */}
      <CustomCursor />

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

      {/* Toast stack */}
      <div style={{
        position: 'fixed', top: 16, right: 16, zIndex: 9990,
        display: 'flex', flexDirection: 'column', gap: 8, pointerEvents: 'none',
      }}>
        {toasts.map(t => (
          <div key={t.id} style={{
            pointerEvents: 'auto', display: 'flex', alignItems: 'center', gap: 12,
            padding: '10px 16px', background: 'rgba(0,0,0,0.9)',
            border: '1px solid rgba(255,255,255,0.08)',
            fontFamily: '"IBM Plex Mono", monospace', fontSize: 11,
            color: t.type === 'error' ? '#f87171' : 'rgba(255,255,255,0.65)', maxWidth: 320,
          }}>
            <span style={{ flex: 1 }}>{t.message}</span>
            <button onClick={() => removeToast(t.id)}
              style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)', cursor: 'pointer', fontSize: 14 }}>×</button>
          </div>
        ))}
      </div>
    </>
  )
}
