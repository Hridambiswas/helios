import { useState, useEffect } from 'react'
import { Hero } from './components/Hero'
import { QueryInterface } from './components/QueryInterface'
import { PipelineSection } from './components/PipelineSection'
import { HistorySection } from './components/HistorySection'
import { UploadSection } from './components/UploadSection'
import { Navbar } from './components/Navbar'
import { Footer } from './components/Footer'
import { AuthModal } from './components/AuthModal'
import { useAuth } from './hooks/useAuth'
import type { QueryResponse } from './api/client'

export default function App() {
  const { user, loading, login, register, logout } = useAuth()
  const [showAuth, setShowAuth] = useState(false)
  const [pendingQuery, setPendingQuery] = useState<string | undefined>()
  const [pendingGuestQuery, setPendingGuestQuery] = useState<string | undefined>()
  const [historyRefresh, setHistoryRefresh] = useState(0)

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
          onLogin={login}
          onRegister={register}
        />
      )}

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
