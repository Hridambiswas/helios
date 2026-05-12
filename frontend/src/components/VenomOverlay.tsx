import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

// Pure CSS overlay — no WebGL, no callbacks, no timers.
// App.tsx owns the 3-second timer and removes this from the tree.
// AnimatePresence handles the exit animation via the `exit` prop.
export function VenomOverlay() {
  const barRef = useRef<HTMLDivElement>(null)

  // Animate the progress bar width via plain DOM so it's independent
  // of any React state and can't crash
  useEffect(() => {
    const el = barRef.current
    if (!el) return
    el.style.transition = 'transform 3.2s linear'
    requestAnimationFrame(() => { el.style.transform = 'scaleX(1)' })
  }, [])

  return (
    <motion.div
      key="venom-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.05, transition: { duration: 0.8, ease: [0.4, 0, 1, 1] } }}
      transition={{ duration: 0.6 }}
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: '#000',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: 28,
      }}
    >
      {/* Large title — impossible to miss */}
      <h1
        style={{
          fontFamily: '"Inter Tight", "Montserrat", sans-serif',
          fontWeight: 900,
          fontSize: 'clamp(72px, 16vw, 200px)',
          letterSpacing: '-0.045em',
          color: '#fff',
          lineHeight: 1,
          userSelect: 'none',
        }}
      >
        HELIOS
      </h1>

      {/* Thin progress bar */}
      <div
        style={{
          width: 80, height: 1,
          background: 'rgba(255,255,255,0.12)',
          overflow: 'hidden',
        }}
      >
        <div
          ref={barRef}
          style={{
            width: '100%', height: '100%',
            background: 'rgba(255,255,255,0.7)',
            transform: 'scaleX(0)',
            transformOrigin: 'left',
          }}
        />
      </div>

      {/* Subtext */}
      <p
        style={{
          fontFamily: '"IBM Plex Mono", monospace',
          fontSize: 9, letterSpacing: '0.5em',
          textTransform: 'uppercase',
          color: 'rgba(255,255,255,0.35)',
        }}
      >
        Initializing
      </p>
    </motion.div>
  )
}
