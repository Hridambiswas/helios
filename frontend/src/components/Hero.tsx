import { useState, useRef, useEffect } from 'react'
import { motion, useScroll, useTransform, useMotionValue, useSpring } from 'framer-motion'

// ── Deterministic blob configs — no layout-shift hydration issues ─────────────
const BLOBS = [
  { left: '12%', top: '22%', w: 380, h: 320, x: [0,90,-60,110,-20,0],  y: [0,70,130,-50,80,0],  dur: 18 },
  { left: '75%', top: '58%', w: 260, h: 290, x: [0,-70,50,-90,40,0],   y: [0,90,-60,110,-30,0],  dur: 14 },
  { left: '48%', top: '78%', w: 440, h: 380, x: [0,60,-80,40,-100,0],  y: [0,-80,40,100,-50,0],  dur: 20 },
  { left: '6%',  top: '68%', w: 200, h: 220, x: [0,50,-30,80,-40,0],   y: [0,-60,90,-40,70,0],   dur: 12 },
  { left: '85%', top: '15%', w: 300, h: 260, x: [0,-80,60,-50,90,0],   y: [0,80,-50,110,-60,0],  dur: 16 },
  { left: '55%', top: '38%', w: 220, h: 240, x: [0,70,-50,60,-80,0],   y: [0,-70,50,-90,60,0],   dur: 13 },
  { left: '30%', top: '12%', w: 280, h: 240, x: [0,-60,80,-40,70,0],   y: [0,50,100,-70,40,0],   dur: 15 },
]

// Magnetic pull wrapper
function Magnetic({ children, strength = 0.28 }: { children: React.ReactNode; strength?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const sx = useSpring(x, { stiffness: 240, damping: 18 })
  const sy = useSpring(y, { stiffness: 240, damping: 18 })

  const onMove = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect()
    x.set((e.clientX - r.left - r.width  / 2) * strength)
    y.set((e.clientY - r.top  - r.height / 2) * strength)
  }
  const onLeave = () => { x.set(0); y.set(0) }

  return (
    <motion.div ref={ref} style={{ x: sx, y: sy, display: 'inline-block' }}
      onMouseMove={onMove} onMouseLeave={onLeave}>
      {children}
    </motion.div>
  )
}

export function Hero({
  onQuerySubmit,
  onAuthClick,
  isLoggedIn,
}: {
  onQuerySubmit: (q: string) => void
  onAuthClick:   () => void
  isLoggedIn:    boolean
}) {
  const [query, setQuery] = useState('')
  const heroRef = useRef<HTMLElement>(null)

  // Scroll-driven parallax + liquid reveal
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] })
  const contentY  = useTransform(scrollYProgress, [0, 0.5], ['0%', '-14%'])
  const contentA  = useTransform(scrollYProgress, [0, 0.42], [1, 0])
  const liquidH   = useTransform(scrollYProgress, [0.08, 0.52], ['0%', '105%'])

  // Cursor tracking inside hero
  const rawX = useMotionValue(-400)
  const rawY = useMotionValue(-400)
  const cursorX = useSpring(rawX, { damping: 26, stiffness: 310 })
  const cursorY = useSpring(rawY, { damping: 26, stiffness: 310 })

  useEffect(() => {
    const hero = heroRef.current
    if (!hero) return
    const onMove = (e: MouseEvent) => {
      const r = hero.getBoundingClientRect()
      rawX.set(e.clientX - r.left)
      rawY.set(e.clientY - r.top)
    }
    hero.addEventListener('mousemove', onMove)
    return () => hero.removeEventListener('mousemove', onMove)
  }, [rawX, rawY])

  const submit = () => {
    const q = query.trim().slice(0, 500)
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
  }

  return (
    <section ref={heroRef} className="relative h-screen overflow-hidden" style={{ background: '#050505' }}>

      {/* ── SVG gooey filter definition ─────────────────────────────────── */}
      <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
          <filter id="hero-goo" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="18" result="blur" />
            <feColorMatrix
              in="blur" mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 24 -12"
              result="goo"
            />
          </filter>
        </defs>
      </svg>

      {/* ── Gooey liquid blob layer ─────────────────────────────────────── */}
      <div style={{ position: 'absolute', inset: 0, filter: 'url(#hero-goo)', zIndex: 1 }}>

        {/* Ambient floating blobs */}
        {BLOBS.map((b, i) => (
          <motion.div
            key={i}
            style={{
              position: 'absolute',
              left: b.left, top: b.top,
              width: b.w, height: b.h,
              borderRadius: '50%',
              background: `radial-gradient(circle at 32% 32%, #1a1a1a, #080808)`,
              translateX: '-50%', translateY: '-50%',
            }}
            animate={{ x: b.x, y: b.y }}
            transition={{ duration: b.dur, repeat: Infinity, repeatType: 'loop', ease: 'easeInOut' }}
          />
        ))}

        {/* Cursor-following blob */}
        <motion.div
          style={{
            position: 'absolute',
            top: 0, left: 0,
            width: 160, height: 160,
            borderRadius: '50%',
            background: 'radial-gradient(circle at 40% 40%, #222, #0a0a0a)',
            x: cursorX, y: cursorY,
            translateX: '-50%', translateY: '-50%',
          }}
        />

        {/* Slow large background blob */}
        <motion.div
          style={{
            position: 'absolute',
            left: '50%', top: '50%',
            width: 600, height: 500,
            borderRadius: '50%',
            background: 'radial-gradient(circle at 50% 50%, #111, #040404)',
            translateX: '-50%', translateY: '-50%',
          }}
          animate={{ x: [-40, 40, -20, 60, -40], y: [-30, 50, -60, 20, -30] }}
          transition={{ duration: 25, repeat: Infinity, repeatType: 'loop', ease: 'easeInOut' }}
        />
      </div>

      {/* ── Edge vignette ───────────────────────────────────────────────── */}
      <div
        style={{
          position: 'absolute', inset: 0, zIndex: 2, pointerEvents: 'none',
          background: 'radial-gradient(ellipse at 50% 50%, transparent 30%, #050505 100%)',
        }}
      />

      {/* ── Hero content ────────────────────────────────────────────────── */}
      <motion.div
        style={{
          position: 'relative', zIndex: 10,
          opacity: contentA as any, y: contentY as any,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          height: '100%', textAlign: 'center', padding: '0 1rem',
        }}
      >
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '6px 16px', marginBottom: '2.5rem',
            border: '1px solid rgba(255,255,255,0.07)',
            background: 'rgba(255,255,255,0.025)',
            backdropFilter: 'blur(8px)',
          }}
        >
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#fff', opacity: 0.5, animation: 'pulse 2s infinite' }} />
          <span style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 9, letterSpacing: '0.4em', color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase' }}>
            Distributed Multi-Agent AI
          </span>
        </motion.div>

        {/* Main heading */}
        <div style={{ overflow: 'hidden', marginBottom: '1rem' }}>
          <motion.h1
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.45, duration: 1.0, ease: [0.16, 1, 0.3, 1] }}
            style={{
              fontFamily: '"Montserrat", sans-serif',
              fontWeight: 900,
              fontSize: 'clamp(76px, 17vw, 210px)',
              letterSpacing: '-0.045em',
              lineHeight: 0.9,
              color: '#ffffff',
              userSelect: 'none',
              margin: 0,
            }}
          >
            HELIOS
          </motion.h1>
        </div>

        {/* Divider line */}
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.9, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          style={{ width: 60, height: 1, background: 'rgba(255,255,255,0.15)', marginBottom: '1.5rem' }}
        />

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.95, duration: 0.8 }}
          style={{
            fontFamily: '"DM Sans", sans-serif',
            fontSize: 'clamp(12px, 1.4vw, 17px)',
            letterSpacing: '0.18em',
            color: 'rgba(255,255,255,0.25)',
            textTransform: 'uppercase',
            marginBottom: '3rem',
          }}
        >
          Five agents &nbsp;&middot;&nbsp; One pipeline &nbsp;&middot;&nbsp; No query escapes
        </motion.p>

        {/* Query input */}
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.05, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          style={{ width: '100%', maxWidth: 560 }}
        >
          <div
            id="hero-input-wrap"
            style={{
              display: 'flex',
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(255,255,255,0.04)',
              backdropFilter: 'blur(20px)',
              transition: 'border-color 0.25s, box-shadow 0.25s',
            }}
            onFocusCapture={e => {
              const el = e.currentTarget as HTMLDivElement
              el.style.borderColor = 'rgba(255,255,255,0.28)'
              el.style.boxShadow   = '0 0 0 1px rgba(255,255,255,0.06)'
            }}
            onBlurCapture={e => {
              const el = e.currentTarget as HTMLDivElement
              el.style.borderColor = 'rgba(255,255,255,0.1)'
              el.style.boxShadow   = 'none'
            }}
          >
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="Ask the intelligence anything…"
              style={{
                flex: 1,
                background: 'transparent',
                padding: '1rem 1.25rem',
                fontFamily: '"DM Sans", sans-serif',
                fontSize: 14,
                color: '#fff',
                outline: 'none',
                caretColor: '#fff',
              }}
            />
            <Magnetic strength={0.12}>
              <button
                onClick={submit}
                style={{
                  padding: '0 1.5rem',
                  fontFamily: '"IBM Plex Mono", monospace',
                  fontSize: 9,
                  letterSpacing: '0.3em',
                  textTransform: 'uppercase',
                  color: query.trim() ? '#fff' : 'rgba(255,255,255,0.22)',
                  borderLeft: '1px solid rgba(255,255,255,0.08)',
                  background: 'transparent',
                  transition: 'color 0.2s',
                  height: '100%',
                  cursor: 'none',
                }}
              >
                RUN
              </button>
            </Magnetic>
          </div>

          {!isLoggedIn && (
            <p style={{ marginTop: '0.6rem', fontFamily: '"IBM Plex Mono", monospace', fontSize: 9, textAlign: 'center', color: 'rgba(255,255,255,0.18)' }}>
              Guest mode ·{' '}
              <button
                onClick={onAuthClick}
                style={{ color: 'rgba(255,255,255,0.5)', background: 'none', border: 'none', fontFamily: 'inherit', fontSize: 'inherit', cursor: 'none', transition: 'opacity 0.2s' }}
                onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.opacity = '0.6')}
                onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.opacity = '1')}
              >
                sign in
              </button>
              {' '}for full access
            </p>
          )}
        </motion.div>

        {/* Suggestion chips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.35, duration: 0.7 }}
          style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 8, marginTop: '1.5rem' }}
        >
          {['Summarize my documents', 'Explain the pipeline', 'Find key insights'].map(s => (
            <Magnetic key={s} strength={0.2}>
              <button
                onClick={() => onQuerySubmit(s)}
                style={{
                  fontFamily: '"IBM Plex Mono", monospace',
                  fontSize: 9,
                  letterSpacing: '0.15em',
                  padding: '6px 14px',
                  border: '1px solid rgba(255,255,255,0.07)',
                  color: 'rgba(255,255,255,0.28)',
                  background: 'transparent',
                  cursor: 'none',
                  transition: 'border-color 0.2s, color 0.2s',
                }}
                onMouseEnter={e => {
                  const b = e.currentTarget as HTMLButtonElement
                  b.style.borderColor = 'rgba(255,255,255,0.22)'
                  b.style.color       = 'rgba(255,255,255,0.72)'
                }}
                onMouseLeave={e => {
                  const b = e.currentTarget as HTMLButtonElement
                  b.style.borderColor = 'rgba(255,255,255,0.07)'
                  b.style.color       = 'rgba(255,255,255,0.28)'
                }}
              >
                {s}
              </button>
            </Magnetic>
          ))}
        </motion.div>
      </motion.div>

      {/* ── Scroll indicator ─────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.9 }}
        style={{
          position: 'absolute', bottom: 32, left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
          zIndex: 10, pointerEvents: 'none',
        }}
      >
        <span style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 8, letterSpacing: '0.4em', color: 'rgba(255,255,255,0.18)', textTransform: 'uppercase' }}>
          scroll
        </span>
        <motion.div
          style={{ width: 1, height: 44, background: 'linear-gradient(to bottom, rgba(255,255,255,0.28), transparent)' }}
          animate={{ scaleY: [1, 0.35, 1], opacity: [0.28, 0.8, 0.28] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
        />
      </motion.div>

      {/* ── Liquid black reveal on scroll — rises from bottom ─────────────── */}
      <motion.div
        style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          height: liquidH,
          background: '#000',
          zIndex: 20,
          originY: 'bottom',
        }}
      />
    </section>
  )
}
