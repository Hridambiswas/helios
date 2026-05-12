import { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Hero } from './components/Hero'
import { QueryInterface } from './components/QueryInterface'
import { PipelineSection } from './components/PipelineSection'
import { HistorySection } from './components/HistorySection'
import { UploadSection } from './components/UploadSection'
import { UploadPanel } from './components/UploadPanel'
import { MobileBottomNav } from './components/MobileBottomNav'
import { Navbar } from './components/Navbar'
import { Footer } from './components/Footer'
import { ChatSidebar } from './components/ChatSidebar'
import { ChatView } from './components/ChatView'
import { AuthModal } from './components/AuthModal'
import { SplashScreen } from './components/SplashScreen'
import { FireDivider } from './components/DragonDecor'
import { PurpleExplosion } from './components/PurpleExplosion'
import { useAuth } from './hooks/useAuth'
import { useToast } from './hooks/useToast'
import { useConversations } from './hooks/useConversations'
import type { QueryResponse } from './api/client'

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
  } = useConversations(!!user)

  const [chatMode, setChatMode] = useState(false)
  const [showAuth, setShowAuth] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => window.innerWidth < 640)
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [pendingQuery, setPendingQuery] = useState<string | undefined>()
  const [pendingGuestQuery, setPendingGuestQuery] = useState<string | undefined>()
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

  // ?q= deep link → open chat mode with that query
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const q = params.get('q')
    if (q) {
      openChat(q.trim())
      const url = new URL(window.location.href)
      url.searchParams.delete('q')
      window.history.replaceState({}, '', url.toString())
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // After login, run the query the guest was trying to submit
  useEffect(() => {
    if (user && pendingGuestQuery) {
      openChat(pendingGuestQuery)
      setPendingGuestQuery(undefined)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user])

  const openChat = (initialQuery?: string) => {
    let id = activeId
    if (!id || !active || active.messages.length > 0) {
      id = newConversation()
    }
    if (initialQuery && id) {
      // Add user message only — ChatView detects it and fires the API call
      setTimeout(() => addUserMessage(id!, initialQuery), 50)
    }
    setChatMode(true)
    window.scrollTo(0, 0)
  }

  const handleQuerySubmit = (q: string) => {
    // From landing page query interface — switch to chat mode
    openChat(q)
    setPendingQuery(undefined)
  }

  const handleAuthRequired = (q: string) => {
    setPendingGuestQuery(q)
    setShowAuth(true)
  }

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
      {/* 3-D dragon splash */}
      <AnimatePresence>
        {!splashDone && (
          <SplashScreen
            key="splash"
            onComplete={() => { sessionStorage.setItem('helios_splash', '1'); setSplashDone(true) }}
          />
        )}
      </AnimatePresence>

      <div className="scanline" />

      {/* ── CHAT MODE ── */}
      <AnimatePresence mode="wait">
      {chatMode ? (
        <motion.div
          key="chat"
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 40 }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          className="flex h-screen overflow-hidden bg-black pb-14 sm:pb-0"
        >
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
              <div className="flex items-center gap-4">
                {/* Back to landing */}
                <button
                  onClick={() => setChatMode(false)}
                  className="font-mono text-[10px] text-[#444] hover:text-white transition-colors"
                >
                  ← Home
                </button>
                <span className="font-mono text-xs text-[#333] truncate max-w-xs hidden sm:block">
                  {active?.title ?? 'HELIOS'}
                </span>
              </div>
              <div className="flex items-center gap-4">
                {user ? (
                  <span className="font-mono text-[10px] text-[#444]">{user.username}</span>
                ) : (
                  <button onClick={() => setShowAuth(true)} className="font-mono text-[10px] text-crimson hover:text-crimson-light transition-colors">
                    Sign in
                  </button>
                )}
              </div>
            </div>

            {/* Upload panel */}
            {showUpload && (
              <UploadPanel
                isLoggedIn={!!user}
                onAuthClick={() => setShowAuth(true)}
                onClose={() => setShowUpload(false)}
              />
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
        </motion.div>
      ) : (
        /* ── LANDING MODE ── */
        <motion.div
          key="landing"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Navbar
            user={user}
            onAuthClick={() => setShowAuth(true)}
            onLogout={logout}
          />

          {/* Purple explosion — fixed canvas, scroll-driven */}
          <PurpleExplosion />

          <main className="pt-12">
            <Hero
              onQuerySubmit={handleQuerySubmit}
              onAuthClick={() => setShowAuth(true)}
              isLoggedIn={!!user}
            />

            {/* Inline query interface — submitting opens chat */}
            <div id="query-section">
              <QueryInterface
                initialQuery={pendingQuery}
                onNewResult={(_: QueryResponse) => setHistoryRefresh(n => n + 1)}
                isLoggedIn={!!user}
                onAuthRequired={handleAuthRequired}
                onOpenChat={handleQuerySubmit}
              />
            </div>

            <FireDivider flip />

            <div id="pipeline-section">
              <PipelineSection />
            </div>

            <FireDivider />

            <div id="history-section">
              <HistorySection isLoggedIn={!!user} refreshTrigger={historyRefresh} />
            </div>

            <div id="upload-section">
              <UploadSection isLoggedIn={!!user} />
            </div>

            <Footer />
          </main>

          <ScrollToTop />
        </motion.div>
      )}
      </AnimatePresence>

      {/* Mobile bottom nav (shown in both modes) */}
      <MobileBottomNav
        chatMode={chatMode}
        onHome={() => setChatMode(false)}
        onChat={() => openChat()}
        onUpload={() => { if (!chatMode) { setChatMode(true) }; setTimeout(() => setShowUpload(true), 50) }}
        user={user}
        onAuthClick={() => setShowAuth(true)}
        onLogout={() => { logout(); addToast('Signed out', 'info') }}
      />

      {/* Auth modal (shared between both modes) */}
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

function ScrollToTop() {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const handler = () => setVisible(window.scrollY > 400)
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [])
  if (!visible) return null
  return (
    <button
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      className="fixed bottom-6 right-6 z-50 w-10 h-10 bg-ink border border-crimson/30 hover:border-crimson flex items-center justify-center text-crimson hover:bg-crimson/10 transition-all"
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M6 1L11 8H1L6 1Z" /></svg>
    </button>
  )
}

