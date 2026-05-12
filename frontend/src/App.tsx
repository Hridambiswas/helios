import { useState, useEffect, useRef } from 'react'
import { AnimatePresence, motion, useMotionValue, useSpring } from 'framer-motion'
import { VenomOverlay } from './components/VenomOverlay'
import { PromptPage } from './components/PromptPage'
import { ChatPage } from './components/ChatPage'
import { AuthModal } from './components/AuthModal'
import { MobileBottomNav } from './components/MobileBottomNav'
import { useAuth } from './hooks/useAuth'
import { useToast } from './hooks/useToast'
import { useConversations } from './hooks/useConversations'

// ─── Film grain overlay ───────────────────────────────────────────────────────

function FilmGrain() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    let raf = 0, tick = 0

    const resize = () => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    const draw = () => {
      raf = requestAnimationFrame(draw)
      if (++tick % 2 !== 0) return
      const { width, height } = canvas
      const img = ctx.createImageData(width, height)
      const d = img.data
      for (let i = 0; i < d.length; i += 4) {
        const v = (Math.random() * 255) | 0
        d[i] = d[i + 1] = d[i + 2] = v
        d[i + 3] = 16
      }
      ctx.putImageData(img, 0, 0)
    }
    draw()

    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed', inset: 0,
        width: '100%', height: '100%',
        pointerEvents: 'none',
        mixBlendMode: 'overlay',
        opacity: 0.60,
        zIndex: 9990,
      }}
    />
  )
}

// ─── Custom cursor ────────────────────────────────────────────────────────────

function VenomCursor() {
  const mx = useMotionValue(-300)
  const my = useMotionValue(-300)

  const blobX = useSpring(mx, { damping: 15, stiffness: 90 })
  const blobY = useSpring(my, { damping: 15, stiffness: 90 })
  const ringX = useSpring(mx, { damping: 30, stiffness: 600 })
  const ringY = useSpring(my, { damping: 30, stiffness: 600 })
  const dotX  = useSpring(mx, { damping: 40, stiffness: 900 })
  const dotY  = useSpring(my, { damping: 40, stiffness: 900 })

  useEffect(() => {
    const move = (e: MouseEvent) => { mx.set(e.clientX); my.set(e.clientY) }
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [mx, my])

  return (
    <>
      <motion.div style={{
        position: 'fixed', x: blobX, y: blobY,
        translateX: '-50%', translateY: '-50%',
        width: 50, height: 50, borderRadius: '50%',
        background: 'rgba(255,255,255,0.06)',
        pointerEvents: 'none', zIndex: 9997,
        mixBlendMode: 'difference',
      }} />
      <motion.div style={{
        position: 'fixed', x: ringX, y: ringY,
        translateX: '-50%', translateY: '-50%',
        width: 28, height: 28, borderRadius: '50%',
        border: '1px solid rgba(255,255,255,0.55)',
        pointerEvents: 'none', zIndex: 9998,
        mixBlendMode: 'difference',
      }} />
      <motion.div style={{
        position: 'fixed', x: dotX, y: dotY,
        translateX: '-50%', translateY: '-50%',
        width: 4, height: 4, borderRadius: '50%',
        background: '#fff',
        pointerEvents: 'none', zIndex: 9999,
        mixBlendMode: 'difference',
      }} />
    </>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const { user, loading, login, register, logout } = useAuth()
  const { toasts, add: addToast, remove: removeToast } = useToast()
  const {
    conversations, active, activeId,
    selectConversation, newConversation, deleteConversation,
    addUserMessage, addAssistantPlaceholder, updateMessage,
  } = useConversations(!!user)

  // Skip overlay if already seen in this browser session
  const [overlayDone, setOverlayDone] = useState(
    () => sessionStorage.getItem('venom_intro') === '1'
  )
  const [chatMode, setChatMode]   = useState(false)
  const [showAuth, setShowAuth]   = useState(false)

  // Handle OAuth redirect tokens
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const at = params.get('access_token')
    const rt = params.get('refresh_token')
    if (params.get('oauth_error')) {
      addToast('OAuth sign-in failed — please try again', 'error')
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

  const openChat = (initialQuery?: string) => {
    let id = activeId
    if (!id || !active || active.messages.length > 0) id = newConversation()
    if (initialQuery && id) setTimeout(() => addUserMessage(id!, initialQuery), 50)
    setChatMode(true)
    window.scrollTo(0, 0)
  }

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="status-dot" />
          <span style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 10, color: 'rgba(255,255,255,0.2)', letterSpacing: '0.4em' }}>
            INITIALIZING
          </span>
        </div>
      </div>
    )
  }

  return (
    <>
      <VenomCursor />
      <FilmGrain />

      {/* Scanline */}
      <div className="scanline" />

      <AnimatePresence mode="wait">
        {!overlayDone && (
          <VenomOverlay
            key="overlay"
            onComplete={() => {
              sessionStorage.setItem('venom_intro', '1')
              setOverlayDone(true)
            }}
          />
        )}

        {overlayDone && !chatMode && (
          <PromptPage
            key="prompt"
            onSubmit={q => openChat(q)}
            user={user}
            onAuthClick={() => setShowAuth(true)}
          />
        )}

        {overlayDone && chatMode && (
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

      {/* Mobile nav (visible in both modes) */}
      <MobileBottomNav
        chatMode={chatMode}
        onHome={() => setChatMode(false)}
        onChat={() => openChat()}
        onUpload={() => { if (!chatMode) setChatMode(true) }}
        user={user}
        onAuthClick={() => setShowAuth(true)}
        onLogout={() => { logout(); addToast('Signed out', 'info') }}
      />

      {/* Auth modal */}
      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onLogin={async (u, p) => { await login(u, p); addToast(`Welcome back, ${u}!`, 'success') }}
          onRegister={async (u, e, p) => { await register(u, e, p); addToast(`Welcome, ${u}!`, 'success') }}
        />
      )}

      {/* Toasts */}
      <div style={{ position: 'fixed', top: 16, right: 16, zIndex: 9995, display: 'flex', flexDirection: 'column', gap: 8, pointerEvents: 'none' }}>
        {toasts.map(t => (
          <div
            key={t.id}
            className="animate-slide-up"
            style={{
              pointerEvents: 'auto',
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '10px 16px',
              background: 'rgba(0,0,0,0.88)',
              border: `1px solid ${t.type === 'success' ? 'rgba(255,255,255,0.12)' : t.type === 'error' ? 'rgba(255,100,100,0.25)' : 'rgba(255,255,255,0.07)'}`,
              backdropFilter: 'blur(16px)',
              fontFamily: '"IBM Plex Mono", monospace', fontSize: 11,
              color: t.type === 'success' ? 'rgba(255,255,255,0.75)' : t.type === 'error' ? 'rgba(255,120,120,0.9)' : 'rgba(255,255,255,0.55)',
              maxWidth: 320,
            }}
          >
            <span style={{ flex: 1 }}>{t.message}</span>
            <button
              onClick={() => removeToast(t.id)}
              style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)', cursor: 'none', fontSize: 14, lineHeight: 1 }}
            >×</button>
          </div>
        ))}
      </div>
    </>
  )
}
