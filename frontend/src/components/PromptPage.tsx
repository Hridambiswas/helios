import { useState, useEffect, useRef, useCallback } from 'react'
import {
  motion, AnimatePresence,
  useScroll, useTransform,
  useMotionValue, useSpring,
} from 'framer-motion'
import type { User } from '../hooks/useAuth'

// ─── Lenis smooth scroll ────────────────────────────────────────────────────

function useLenis() {
  useEffect(() => {
    let lenis: any
    ;(async () => {
      try {
        const { default: Lenis } = await import('lenis')
        lenis = new Lenis({
          duration: 1.2,
          easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        })
        const raf = (time: number) => { lenis.raf(time); requestAnimationFrame(raf) }
        requestAnimationFrame(raf)
      } catch { /* native scroll fallback */ }
    })()
    return () => lenis?.destroy()
  }, [])
}

// ─── Magnetic wrapper ────────────────────────────────────────────────────────

function Magnetic({ children }: { children: React.ReactNode }) {
  const ref  = useRef<HTMLDivElement>(null)
  const rx   = useMotionValue(0)
  const ry   = useMotionValue(0)
  const x    = useSpring(rx, { damping: 12, stiffness: 200 })
  const y    = useSpring(ry, { damping: 12, stiffness: 200 })
  return (
    <motion.div
      ref={ref} style={{ x, y, display: 'inline-block' }}
      onMouseMove={e => {
        const r = ref.current!.getBoundingClientRect()
        rx.set((e.clientX - (r.left + r.width / 2)) * 0.3)
        ry.set((e.clientY - (r.top  + r.height / 2)) * 0.3)
      }}
      onMouseLeave={() => { rx.set(0); ry.set(0) }}
    >{children}</motion.div>
  )
}

// ─── Pipeline data ───────────────────────────────────────────────────────────

const STAGES = [
  { label: 'INGEST',        sub: 'Consume any document format — PDF, DOCX, HTML, video transcripts' },
  { label: 'CHUNK & EMBED', sub: 'Semantic segmentation + vector encoding via fine-tuned bi-encoder' },
  { label: 'HYBRID SEARCH', sub: 'Dense vector + sparse BM25 retrieval, fused via Reciprocal Rank' },
  { label: 'RE-RANK',       sub: 'LLM-as-judge cross-encoder scoring with relevance thresholding' },
  { label: 'GENERATE',      sub: 'Grounded synthesis — citations, confidence scores, source attribution' },
]

// ─── Main component ──────────────────────────────────────────────────────────

interface Props {
  onSubmit:   (q: string) => void
  user:       User | null
  onAuthClick: () => void
}

export function PromptPage({ onSubmit, user, onAuthClick }: Props) {
  useLenis()

  const [query, setQuery]       = useState('')
  const [focused, setFocused]   = useState(false)
  const [swallowing, setSwallowing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Continuous micro-vibration while input is focused
  const shakeX = useMotionValue(0)
  const shakeSp = useSpring(shakeX, { damping: 3, stiffness: 600 })
  useEffect(() => {
    if (!focused) { shakeX.set(0); return }
    let id: number
    const tick = () => {
      shakeX.set(Math.sin(Date.now() * 0.013) * 1.3)
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [focused, shakeX])

  // Pipeline scroll tracking
  const pipelineRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: pipelineRef,
    offset: ['start 90%', 'end 20%'],
  })
  const lineH    = useTransform(scrollYProgress, [0, 1], ['0%', '100%'])
  const dropTop  = useTransform(scrollYProgress, [0, 1], ['0%', '100%'])

  const handleSubmit = useCallback(() => {
    const q = query.trim()
    if (!q) return
    if (!user) { onAuthClick(); return }
    setSwallowing(true)
    setTimeout(() => onSubmit(q), 680)
  }, [query, user, onAuthClick, onSubmit])

  return (
    <motion.div
      exit={{ scale: 1.08, opacity: 0, filter: 'blur(8px)' }}
      transition={{ duration: 0.5, ease: [0.76, 0, 0.24, 1] }}
      style={{ background: '#000', minHeight: '100vh', position: 'relative', overflowX: 'hidden' }}
    >
      {/* ── Swallow transition overlay ──────────────────────────────────────── */}
      <AnimatePresence>
        {swallowing && (
          <motion.div
            key="swallow"
            initial={{ clipPath: 'circle(0% at 50% 55%)' }}
            animate={{ clipPath: 'circle(180% at 50% 55%)' }}
            transition={{ duration: 0.68, ease: [0.76, 0, 0.24, 1] }}
            style={{ position: 'fixed', inset: 0, background: '#000', zIndex: 60, pointerEvents: 'none' }}
          />
        )}
      </AnimatePresence>

      {/* ── HERO ──────────────────────────────────────────────────────────────── */}
      <section
        style={{
          minHeight: '100vh',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          padding: '0 1.5rem',
          position: 'relative',
        }}
      >
        {/* Eyebrow */}
        <motion.p
          initial={{ opacity: 0, y: -14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.9 }}
          style={{
            fontFamily: '"IBM Plex Mono", monospace',
            fontSize: 10, letterSpacing: '0.38em',
            textTransform: 'uppercase',
            color: 'rgba(255,255,255,0.22)',
            marginBottom: 28,
          }}
        >
          Distributed Multi-Agent RAG
        </motion.p>

        {/* HELIOS heading */}
        <motion.h1
          initial={{ opacity: 0, y: 44 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          style={{
            fontFamily: '"Inter Tight", "Montserrat", sans-serif',
            fontWeight: 900,
            fontSize: 'clamp(78px, 17vw, 210px)',
            lineHeight: 0.87,
            letterSpacing: '-0.045em',
            color: '#fff',
            textAlign: 'center',
            marginBottom: 54,
            userSelect: 'none',
          }}
        >
          HELIOS
        </motion.h1>

        {/* Glassmorphism input */}
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 1.0, ease: [0.16, 1, 0.3, 1] }}
          style={{ width: '100%', maxWidth: 640 }}
        >
          <motion.div
            style={{
              x: shakeSp,
              display: 'flex',
              backdropFilter: 'blur(32px) saturate(180%)',
              background: 'rgba(255,255,255,0.028)',
              border: `1px solid ${focused ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.07)'}`,
              boxShadow: focused
                ? '0 0 0 1px rgba(255,255,255,0.03), 0 12px 56px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.05)'
                : '0 8px 48px rgba(0,0,0,0.7)',
              transition: 'border-color 0.3s, box-shadow 0.3s',
            }}
          >
            <input
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="Ask anything about your data..."
              style={{
                flex: 1, padding: '18px 22px',
                background: 'transparent', border: 'none', outline: 'none',
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: 13, color: '#fff', cursor: 'none',
                caretColor: 'rgba(255,255,255,0.6)',
              }}
            />
            <Magnetic>
              <button
                onClick={handleSubmit}
                style={{
                  alignSelf: 'stretch',
                  padding: '0 28px',
                  background: '#fff', border: 'none', cursor: 'none',
                  fontFamily: '"IBM Plex Mono", monospace',
                  fontSize: 10, letterSpacing: '0.28em',
                  textTransform: 'uppercase',
                  color: '#000', fontWeight: 700,
                  transition: 'background 0.2s, color 0.2s',
                  whiteSpace: 'nowrap',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = '#e0e0e0'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = '#fff'
                }}
              >
                Run →
              </button>
            </Magnetic>
          </motion.div>

          {/* Auth hint */}
          {!user && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.0 }}
              style={{
                fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                color: 'rgba(255,255,255,0.18)',
                marginTop: 12, textAlign: 'center', letterSpacing: '0.1em',
              }}
            >
              Sign in required to run queries —{' '}
              <button
                onClick={onAuthClick}
                style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.45)', cursor: 'none', fontFamily: 'inherit', fontSize: 'inherit' }}
              >
                Enter
              </button>
            </motion.p>
          )}
        </motion.div>

        {/* Scroll cue */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.6, duration: 1 }}
          style={{
            position: 'absolute', bottom: 36,
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
          }}
        >
          <span style={{
            fontFamily: '"IBM Plex Mono", monospace', fontSize: 9,
            letterSpacing: '0.45em', color: 'rgba(255,255,255,0.14)',
            textTransform: 'uppercase',
          }}>Pipeline</span>
          <motion.div
            animate={{ y: [0, 9, 0] }}
            transition={{ duration: 1.9, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              width: 1, height: 42,
              background: 'linear-gradient(to bottom, rgba(255,255,255,0.35), transparent)',
            }}
          />
        </motion.div>
      </section>

      {/* ── PIPELINE ──────────────────────────────────────────────────────────── */}
      <section
        ref={pipelineRef}
        style={{ padding: '8vh 1.5rem 20vh', position: 'relative' }}
      >
        <div style={{ maxWidth: 580, margin: '0 auto', position: 'relative' }}>
          {/* Section label */}
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            style={{
              fontFamily: '"IBM Plex Mono", monospace', fontSize: 9,
              letterSpacing: '0.45em', color: 'rgba(255,255,255,0.18)',
              textTransform: 'uppercase', marginBottom: 72,
            }}
          >
            Architecture
          </motion.p>

          <div style={{ position: 'relative' }}>
            {/* Track line */}
            <div style={{
              position: 'absolute', left: 0, top: 0, bottom: 0,
              width: 1, background: 'rgba(255,255,255,0.06)',
            }} />

            {/* Liquid fill line */}
            <motion.div style={{
              position: 'absolute', left: 0, top: 0,
              width: 1, height: lineH,
              background: 'linear-gradient(to bottom, #fff 0%, rgba(255,255,255,0.55) 60%, rgba(255,255,255,0.9) 100%)',
              boxShadow: '0 0 10px rgba(255,255,255,0.45), 0 0 24px rgba(255,255,255,0.12)',
            }} />

            {/* Drip bead */}
            <motion.div style={{
              position: 'absolute', left: -2, top: dropTop,
              width: 5, height: 5, borderRadius: '50%',
              background: '#fff',
              boxShadow: '0 0 10px rgba(255,255,255,0.9)',
              translateY: '-50%',
            }} />

            {/* Pulse glow travelling down line */}
            <motion.div
              animate={{ opacity: [0, 0.7, 0], top: ['0%', '100%'] }}
              transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut', repeatDelay: 0.5 }}
              style={{
                position: 'absolute', left: -1,
                width: 3, height: 60,
                background: 'linear-gradient(to bottom, transparent, rgba(255,255,255,0.8), transparent)',
                pointerEvents: 'none',
              }}
            />

            {/* Stage nodes */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 80, paddingLeft: 44 }}>
              {STAGES.map((stage, i) => (
                <motion.div
                  key={stage.label}
                  initial={{ opacity: 0, x: -22 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: '-80px' }}
                  transition={{ duration: 0.8, delay: i * 0.04, ease: [0.16, 1, 0.3, 1] }}
                  style={{ position: 'relative' }}
                >
                  {/* Node circle */}
                  <motion.div
                    initial={{ scale: 0 }}
                    whileInView={{ scale: 1 }}
                    viewport={{ once: true, margin: '-80px' }}
                    transition={{ duration: 0.4, delay: i * 0.04 + 0.1, type: 'spring', stiffness: 400 }}
                    style={{
                      position: 'absolute', left: -51, top: 6,
                      width: 9, height: 9, borderRadius: '50%',
                      background: '#fff',
                      boxShadow: '0 0 0 3px rgba(255,255,255,0.08), 0 0 16px rgba(255,255,255,0.3)',
                    }}
                  />

                  {/* Step number */}
                  <p style={{
                    fontFamily: '"IBM Plex Mono", monospace', fontSize: 9,
                    letterSpacing: '0.3em', color: 'rgba(255,255,255,0.24)',
                    marginBottom: 10, textTransform: 'uppercase',
                  }}>
                    {String(i + 1).padStart(2, '0')}
                  </p>

                  {/* Stage title */}
                  <h3 style={{
                    fontFamily: '"Inter Tight", "Montserrat", sans-serif',
                    fontWeight: 900, fontSize: 'clamp(22px, 4vw, 30px)',
                    letterSpacing: '-0.025em', color: '#fff',
                    marginBottom: 12,
                  }}>
                    {stage.label}
                  </h3>

                  {/* Description */}
                  <p style={{
                    fontFamily: '"DM Sans", sans-serif', fontSize: 14,
                    color: 'rgba(255,255,255,0.35)', lineHeight: 1.65,
                    maxWidth: 420,
                  }}>
                    {stage.sub}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* CTA at bottom of pipeline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          style={{
            maxWidth: 580, margin: '100px auto 0',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20,
          }}
        >
          <p style={{
            fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
            letterSpacing: '0.28em', color: 'rgba(255,255,255,0.18)',
            textTransform: 'uppercase',
          }}>
            Ready to begin
          </p>
          <button
            onClick={() => inputRef.current?.focus()}
            style={{
              fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
              letterSpacing: '0.25em', textTransform: 'uppercase',
              padding: '14px 36px',
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.15)',
              color: 'rgba(255,255,255,0.55)',
              cursor: 'none',
              transition: 'border-color 0.25s, color 0.25s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.55)'
              e.currentTarget.style.color       = '#fff'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)'
              e.currentTarget.style.color       = 'rgba(255,255,255,0.55)'
            }}
          >
            ↑ Return to prompt
          </button>
        </motion.div>
      </section>
    </motion.div>
  )
}
