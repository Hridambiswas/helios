import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { User } from '../hooks/useAuth'
import { HeroSection } from './hero/HeroSection'
import { ArcSection } from './arc/ArcSection'
import { DemoAnswer } from './answer/DemoAnswer'
import { usePipeline } from '../pipeline/PipelineProvider'

interface Props {
  onSubmit:    (q: string) => void
  user:        User | null
  onAuthClick: () => void
}

export function PromptPage({ onSubmit, user, onAuthClick }: Props) {
  const [query, setQuery]         = useState('')
  const [swallowing, setSwallowing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { state, run, isDemoMode, reset } = usePipeline()

  const submit = useCallback((q?: string) => {
    const value = (q ?? query).trim()
    if (!value) return
    if (isDemoMode) {
      // Stay on the prompt page; play the scripted stream in-place so
      // the arc and Sol react, then show DemoAnswer below.
      reset()
      setTimeout(() => run(value), 40) // one tick so reset() lands first
      // Scroll to the arc so the user sees the animation.
      setTimeout(() => {
        document.getElementById('arc-anchor')?.scrollIntoView({
          behavior: 'smooth', block: 'start',
        })
      }, 80)
      return
    }
    setSwallowing(true)
    setTimeout(() => onSubmit(value), 520)
  }, [query, onSubmit, isDemoMode, run, reset])

  return (
    <motion.div
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35, ease: [0.76, 0, 0.24, 1] }}
      style={{
        background: 'var(--polar-night)',
        minHeight: '100vh',
        position: 'relative',
        overflowX: 'hidden',
      }}
    >
      {/* Wordmark + Demo badge + sign-in — top bar (fixed) */}
      <header style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 40,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '20px 32px',
        pointerEvents: 'none',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, pointerEvents: 'auto' }}>
          <span className="display" style={{
            fontSize: 21, letterSpacing: '-0.01em', color: 'var(--snow)',
          }}>
            Helios
          </span>
          {isDemoMode && (
            <span
              className="text--meta"
              title="Demo mode — the answer is scripted; no backend is contacted."
              style={{
                fontSize: 11,
                border: '1px solid var(--frost-hairline)',
                background: 'var(--frost)',
                color: 'var(--sun)',
                borderRadius: 999,
                padding: '3px 10px',
                letterSpacing: 0.4,
              }}
            >
              Demo
            </span>
          )}
        </div>

        {user ? (
          <span className="text--meta" style={{ pointerEvents: 'auto' }}>
            {user.username}
          </span>
        ) : (
          <button
            onClick={onAuthClick}
            className="text helios-focus"
            style={{
              pointerEvents: 'auto',
              background: 'transparent',
              border: '1px solid var(--frost-hairline)',
              borderRadius: 'var(--radius-md)',
              padding: '8px 18px',
              color: 'var(--snow)',
              fontSize: 'var(--step-0)',
              cursor: 'pointer',
              transition: 'border-color 0.2s, background 0.2s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = 'var(--snow-shadow)'
              e.currentTarget.style.background  = 'var(--frost)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'var(--frost-hairline)'
              e.currentTarget.style.background  = 'transparent'
            }}
          >
            Sign in
          </button>
        )}
      </header>

      {/* Handoff overlay for live-mode navigation to chat */}
      <AnimatePresence>
        {swallowing && (
          <motion.div
            key="handoff"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, ease: [0.76, 0, 0.24, 1] }}
            style={{
              position: 'fixed', inset: 0, zIndex: 60,
              background: 'var(--polar-night)',
              pointerEvents: 'none',
            }}
          />
        )}
      </AnimatePresence>

      <HeroSection
        query={query}
        setQuery={setQuery}
        onSubmit={submit}
        inputRef={inputRef}
      />

      <div id="arc-anchor" />
      <ArcSection />

      {isDemoMode && state.phase !== 'idle' && (
        <DemoAnswer state={state} onReset={reset} />
      )}
    </motion.div>
  )
}
