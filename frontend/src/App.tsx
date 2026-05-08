import { useState, useEffect } from 'react'
import { Hero } from './components/Hero'
import { QueryInterface } from './components/QueryInterface'
import { PipelineSection } from './components/PipelineSection'
import { HistorySection } from './components/HistorySection'
import { UploadSection } from './components/UploadSection'
import { Navbar } from './components/Navbar'
import { Footer } from './components/Footer'
import { AuthModal } from './components/AuthModal'
import { SplashScreen } from './components/SplashScreen'
import { ParticleField } from './components/ParticleField'
import { useAuth } from './hooks/useAuth'
import { useToast } from './hooks/useToast'
import type { QueryResponse } from './api/client'

export default function App() {
  const { user, loading, login, register, logout } = useAuth()
  const { toasts, add: addToast, remove: removeToast } = useToast()
  const [showAuth, setShowAuth] = useState(false)
  const [pendingQuery, setPendingQuery] = useState<string | undefined>()
  const [pendingGuestQuery, setPendingGuestQuery] = useState<string | undefined>()
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [splashDone, setSplashDone] = useState(() => sessionStorage.getItem('helios_splash') === '1')

  const handleSplashComplete = () => {
    sessionStorage.setItem('helios_splash', '1')
    setSplashDone(true)
  }

  // Pre-fill query from ?q= URL parameter so users can share deep links
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const q = params.get('q')
    if (q) {
      handleQuerySubmit(q.trim())
      // Remove the param so refreshing doesn't re-submit
      const url = new URL(window.location.href)
      url.searchParams.delete('q')
      window.history.replaceState({}, '', url.toString())
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // After login/register, auto-run the query the guest was trying to submit
  useEffect(() => {
    if (user && pendingGuestQuery) {
      handleQuerySubmit(pendingGuestQuery)
      setPendingGuestQuery(undefined)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="status-dot" />
          <span className="font-mono text-xs text-[#555] tracking-widest">INITIALIZING...</span>
        </div>
      </div>
    )
  }

  const handleNewResult = (_: QueryResponse) => {
    setHistoryRefresh(n => n + 1)
  }

  const handleQuerySubmit = (q: string) => {
    setPendingQuery(undefined)
    setTimeout(() => setPendingQuery(q), 0)
  }

  const handleAuthRequired = (q: string) => {
    setPendingGuestQuery(q)
    setShowAuth(true)
  }

  return (
    <>
      {/* Oni mask splash intro — only first visit per session */}
      {!splashDone && <SplashScreen onComplete={handleSplashComplete} />}

      {/* Floating ember particles */}
      <ParticleField />

      {/* Scanline effect */}
      <div className="scanline" />

      <Navbar
        user={user}
        onAuthClick={() => setShowAuth(true)}
        onLogout={logout}
      />

      <main className="pt-12">
        <Hero
          onQuerySubmit={handleQuerySubmit}
          onAuthClick={() => setShowAuth(true)}
          isLoggedIn={!!user}
        />

        {/* Ink divider */}
        <InkDivider />

        <div id="query-section">
          <QueryInterface
            initialQuery={pendingQuery}
            onNewResult={handleNewResult}
            isLoggedIn={!!user}
            onAuthRequired={handleAuthRequired}
          />
        </div>

        <InkDivider flip />

        <div id="pipeline-section">
          <PipelineSection />
        </div>

        <InkDivider />

        <div id="history-section">
          <HistorySection isLoggedIn={!!user} refreshTrigger={historyRefresh} />
        </div>

        <div id="upload-section">
          <UploadSection isLoggedIn={!!user} />
        </div>

        <Footer />
      </main>

      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onLogin={async (u, p) => { await login(u, p); addToast(`Welcome back, ${u}!`, 'success') }}
          onRegister={async (u, e, p) => { await register(u, e, p); addToast(`Account created — welcome, ${u}!`, 'success') }}
        />
      )}

      {/* Toast notifications */}
      <div className="fixed top-16 right-4 z-50 flex flex-col gap-2 pointer-events-none">
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

      <ScrollToTop />
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
      aria-label="Scroll to top"
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 1L11 8H1L6 1Z" />
      </svg>
    </button>
  )
}

function InkDivider({ flip = false }: { flip?: boolean }) {
  return (
    <div className={`relative h-16 overflow-hidden ${flip ? 'scale-y-[-1]' : ''}`}>
      <svg
        viewBox="0 0 1400 64"
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="none"
      >
        <path
          d="M0,0 C200,64 400,0 700,32 C1000,64 1200,0 1400,32 L1400,64 L0,64 Z"
          fill="rgba(196,30,58,0.04)"
        />
        <path
          d="M0,20 C150,64 350,8 600,36 C850,64 1100,8 1400,40 L1400,64 L0,64 Z"
          fill="rgba(196,30,58,0.02)"
        />
      </svg>
      <div className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-crimson/20 to-transparent" />
    </div>
  )
}
