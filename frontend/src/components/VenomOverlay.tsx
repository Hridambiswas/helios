import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { FluidBackground } from './FluidBackground'
import React from 'react'

// Swallow WebGL errors silently — the CSS gradient is the fallback
class FluidErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { failed: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { failed: false }
  }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? null : this.props.children }
}

interface Props {
  onComplete: () => void
}

export function VenomOverlay({ onComplete }: Props) {
  const [dissolving, setDissolving] = useState(false)
  const mouseRef      = useRef({ x: 0.5, y: 0.5 })
  // ref avoids the bug where a new function reference on every App render
  // would clear & restart these timers
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
    const t1 = setTimeout(() => setDissolving(true), 2500)
    const t2 = setTimeout(() => onCompleteRef.current(), 3500)
    return () => {
      window.removeEventListener('mousemove', move)
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, []) // empty — run once on mount only

  return (
    <motion.div
      initial={{ opacity: 1, scale: 1 }}
      animate={
        dissolving
          ? { opacity: 0, scale: 1.06 }
          : { opacity: 1, scale: 1 }
      }
      transition={
        dissolving
          ? { duration: 1.0, ease: [0.4, 0, 1, 1] }
          : { duration: 0 }
      }
      style={{ position: 'fixed', inset: 0, zIndex: 100 }}
    >
      {/* CSS gradient is always visible; WebGL layers on top if supported */}
      <div
        style={{
          position: 'absolute', inset: 0,
          background:
            'radial-gradient(ellipse 80% 90% at 38% 62%, #130424 0%, #08010f 50%, #000 100%)',
        }}
      >
        <FluidErrorBoundary>
          <FluidBackground mouseRef={mouseRef} />
        </FluidErrorBoundary>
      </div>

      {/* Text overlay — always rendered over the fluid/gradient */}
      <div
        style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          gap: 24, pointerEvents: 'none',
        }}
      >
        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.1, delay: 0.25, ease: [0.16, 1, 0.3, 1] }}
          style={{
            fontFamily: '"Inter Tight", "Montserrat", sans-serif',
            fontWeight: 900,
            fontSize: 'clamp(64px, 14vw, 170px)',
            letterSpacing: '-0.04em',
            color: '#fff',
            lineHeight: 1,
            userSelect: 'none',
            textShadow:
              '0 0 60px rgba(255,255,255,0.22), 0 0 120px rgba(139,92,246,0.15)',
          }}
        >
          HELIOS
        </motion.h1>

        {/* Progress bar */}
        <div
          style={{
            width: 72, height: 1,
            background: 'rgba(255,255,255,0.12)',
            overflow: 'hidden',
          }}
        >
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 2.5, ease: 'easeInOut' }}
            style={{
              width: '100%', height: '100%',
              background: 'rgba(255,255,255,0.65)',
              transformOrigin: 'left',
            }}
          />
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.35 }}
          transition={{ delay: 0.7, duration: 0.8 }}
          style={{
            fontFamily: '"IBM Plex Mono", monospace',
            fontSize: 9, letterSpacing: '0.5em',
            textTransform: 'uppercase', color: '#fff',
          }}
        >
          Initializing
        </motion.p>
      </div>
    </motion.div>
  )
}
