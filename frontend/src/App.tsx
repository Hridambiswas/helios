import { useState, useEffect } from 'react'
import { ChatSidebar } from './components/ChatSidebar'
import { ChatView } from './components/ChatView'
import { UploadSection } from './components/UploadSection'
import { AuthModal } from './components/AuthModal'
import { SplashScreen } from './components/SplashScreen'
import { ParticleField } from './components/ParticleField'
import { useAuth } from './hooks/useAuth'
import { useToast } from './hooks/useToast'
import { useConversations } from './hooks/useConversations'

export default function App() {
  const { user, loading, login, register, logout } = useAuth()
  const { toasts, add: addToast, remove: removeToast } = useToast()
  const {
    conversations,
    active,
    activeId,
    selectConversation,
    newConversation,
    deleteConversation,
    addUserMessage,
    addAssistantPlaceholder,
    updateMessage,
  } = useConversations()

  const [showAuth, setShowAuth] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [splashDone, setSplashDone] = useState(() => sessionStorage.getItem('helios_splash') === '1')

  // Handle OAuth redirect
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const accessToken = params.get('access_token')
    const refreshToken = params.get('refresh_token')
    const oauthError = params.get('oauth_error')

    if (oauthError) {
      addToast('OAuth sign-in failed — please try again', 'error')
      window.history.replaceState({}, '', window.location.pathname)
      return
    }
    if (accessToken && refreshToken) {
      localStorage.setItem('access_token', accessToken)
      localStorage.setItem('refresh_token', refreshToken)
      window.dispatchEvent(new Event('helios:oauth-login'))
      window.history.replaceState({}, '', window.location.pathname)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Handle ?q= deep link — open a new conversation with that query
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const q = params.get('q')
    if (q) {
      const id = newConversation()
      addUserMessage(id, q.trim())
      const url = new URL(window.location.href)
      url.searchParams.delete('q')
      window.history.replaceState({}, '', url.toString())
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-ink">
        <div className="flex items-center gap-3">
          <div className="status-dot" />
          <span className="font-mono text-xs text-[#555] tracking-widest">INITIALIZING...</span>
        </div>
      </div>
    )
  }

  return (
    <>
      {!splashDone && (
        <SplashScreen onComplete={() => { sessionStorage.setItem('helios_splash', '1'); setSplashDone(true) }} />
      )}

      <ParticleField />
      <div className="scanline" />

      {/* Full-screen chat layout */}
      <div className="flex h-screen overflow-hidden bg-ink">

        {/* Sidebar */}
        <ChatSidebar
          conversations={conversations}
          activeId={activeId}
          onSelect={selectConversation}
          onNew={newConversation}
          onDelete={(id) => { deleteConversation(id); addToast('Conversation deleted', 'info') }}
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(v => !v)}
          user={user}
          onAuthClick={() => setShowAuth(true)}
          onLogout={() => { logout(); addToast('Signed out', 'info') }}
          onUploadClick={() => setShowUpload(v => !v)}
        />

        {/* Main area */}
        <div className="flex flex-col flex-1 min-w-0">

          {/* Top bar */}
          <div className="flex items-center justify-between h-14 px-4 border-b border-white/8 shrink-0">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-[#555] tracking-widest">
                {active ? active.title.slice(0, 60) : 'HELIOS'}
              </span>
            </div>
            <div className="flex items-center gap-4">
              {user ? (
                <span className="font-mono text-[10px] text-[#444]">{user.username}</span>
              ) : (
                <button
                  onClick={() => setShowAuth(true)}
                  className="font-mono text-[10px] text-crimson hover:text-crimson-light transition-colors"
                >
                  Sign in
                </button>
              )}
            </div>
          </div>

          {/* Upload panel (collapsible) */}
          {showUpload && (
            <div className="border-b border-white/8 overflow-y-auto max-h-64">
              <UploadSection isLoggedIn={!!user} />
            </div>
          )}

          {/* Chat area */}
          <div className="flex-1 min-h-0">
            <ChatView
              conversation={active}
              isLoggedIn={!!user}
              onAuthRequired={() => setShowAuth(true)}
              onAddUserMessage={addUserMessage}
              onAddPlaceholder={addAssistantPlaceholder}
              onUpdateMessage={updateMessage}
              onNeedConversation={newConversation}
            />
          </div>
        </div>
      </div>

      {/* Auth modal */}
      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onLogin={async (u, p) => { await login(u, p); addToast(`Welcome back, ${u}!`, 'success') }}
          onRegister={async (u, e, p) => { await register(u, e, p); addToast(`Welcome, ${u}!`, 'success') }}
        />
      )}

      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map(t => (
          <div
            key={t.id}
            className={`pointer-events-auto flex items-center gap-3 px-4 py-2.5 border font-mono text-xs animate-slide-up max-w-xs ${
              t.type === 'success' ? 'bg-ink border-green-500/30 text-green-400' :
              t.type === 'error'   ? 'bg-ink border-crimson/30 text-crimson' :
                                     'bg-ink border-white/10 text-white/70'
            }`}
          >
            <span className="flex-1">{t.message}</span>
            <button onClick={() => removeToast(t.id)} className="text-[#555] hover:text-white">×</button>
          </div>
        ))}
      </div>
    </>
  )
}
