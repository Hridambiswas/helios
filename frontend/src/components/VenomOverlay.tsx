import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { FluidBackground } from './FluidBackground'

interface Props {
  onComplete: () => void
}

export function VenomOverlay({ onComplete }: Props) {
  const [phase, setPhase] = useState<'idle' | 'dissolving'>('idle')
  const mouseRef    = useRef({ x: 0.5, y: 0.5 })
  // Keep a ref to onComplete so the timer never re-registers if App re-renders
  const onCompleteRef = useRef(onComplete)
  useEffect(() => { onCompleteRef.current = onComplete }, [onComplete])

  useEffect(() => {
    const move = (e: MouseEvent) => {
      mouseRef.current = {
        x: e.clientX / window.innerWidth,
        y: 1 - e.clientY / window.innerHeight,
      }
    }
    window.addEventListener('mousemove', move, { passive: true })

    // 2.5 s display → dissolve; 3.4 s total → hand off
    const t1 = setTimeout(() => setPhase('dissolving'), 2500)
    const t2 = setTimeout(() => onCompleteRef.current(), 3400)

    return () => {
      window.removeEventListener('mousemove', move)
      clearTimeout(t1)
      clearTimeout(t2)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // intentionally empty — timers fire once on mount

  const dissolving = phase === 'dissolving'

  return (
    <motion.div
      style={{ position: 'fixed', inset: 0, zIndex: 100, overflow: 'hidden' }}
      // The venom mass "swallows" the screen then retracts — circle collapse
      animate={dissolving ? {
        clipPath: [
          'circle(150% at 50% 50%)',
          'circle(162% at 50% 50%)', // aggressive expand
          'circle(0%   at 50% 50%)', // collapse to nothing
        ],
        transition: {
          duration: 0.9,
          times: [0, 0.14, 1.0],
          ease: [0.76, 0, 0.24, 1],
        },
      } : {
        clipPath: 'circle(150% at 50% 50%)',
      }}
    >
      {/* WebGL fluid surface */}
      <div style={{ position: 'absolute', inset: 0, background: '#000' }}>
        <FluidBackground mouseRef={mouseRef} />
      </div>

      {/* Centered branding */}
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: 20, pointerEvents: 'none',
      }}>
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: dissolving ? 0 : 1, y: dissolving ? -8 : 0 }}
          transition={{ duration: dissolving ? 0.3 : 0.9, delay: dissolving ? 0 : 0.5 }}
          style={{
            fontFamily: '"Inter Tight", "Montserrat", sans-serif',
            fontWeight: 900,
            fontSize: 'clamp(56px, 12vw, 140px)',
            letterSpacing: '-0.04em',
            color: '#fff',
            lineHeight: 1,
            userSelect: 'none',
            // Subtle glow so it reads against the dark fluid
            textShadow: '0 0 60px rgba(255,255,255,0.25), 0 0 120px rgba(255,255,255,0.08)',
          }}
        >
          HELIOS
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: dissolving ? 0 : 0.45 }}
          transition={{ duration: 0.8, delay: 0.9 }}
          style={{
            fontFamily: '"IBM Plex Mono", monospace',
            fontSize: 10, letterSpacing: '0.45em',
            textTransform: 'uppercase', color: '#fff',
          }}
        >
          Initializing
        </motion.p>

        {/* Thin progress line */}
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: dissolving ? 1 : 1 }}
          style={{
            width: 80, height: 1,
            background: 'rgba(255,255,255,0.25)',
            transformOrigin: 'left',
          }}
        >
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 2.5, ease: 'easeInOut' }}
            style={{ width: '100%', height: '100%', background: '#fff', transformOrigin: 'left' }}
          />
        </motion.div>
      </div>
    </motion.div>
  )
}
