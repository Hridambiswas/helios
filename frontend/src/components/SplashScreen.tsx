import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HeroScene } from '../three/HeroScene'

export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const [showTitle,   setShowTitle]   = useState(false)
  const [showTagline, setShowTagline] = useState(false)
  const [showBar,     setShowBar]     = useState(false)
  const [exiting,     setExiting]     = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setShowBar(true),     100)
    const t2 = setTimeout(() => setShowTitle(true),   1400)
    const t3 = setTimeout(() => setShowTagline(true), 2000)
    const t4 = setTimeout(() => setExiting(true),     3200)
    const t5 = setTimeout(() => onComplete(),          4000)
    return () => { [t1, t2, t3, t4, t5].forEach(clearTimeout) }
  }, [onComplete])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: exiting ? 0 : 1, scale: exiting ? 1.04 : 1 }}
      transition={{ duration: exiting ? 0.8 : 0.5, ease: [0.76, 0, 0.24, 1] }}
      className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden bg-black"
    >
      {/* Full-screen 3D orb */}
      <div className="absolute inset-0 pointer-events-none">
        <HeroScene />
      </div>

      {/* Radial violet glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse 60% 50% at 50% 50%, rgba(109,40,217,0.18) 0%, transparent 70%)',
        }}
      />

      {/* Vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, transparent 30%, rgba(0,0,0,0.6) 70%, #000 100%)',
        }}
      />

      {/* Centered content */}
      <div className="relative z-10 text-center select-none pointer-events-none">
        <AnimatePresence>
          {showTitle && (
            <motion.div
              initial={{ opacity: 0, scale: 0.88, y: 30 }}
              animate={{ opacity: 1,  scale: 1,    y: 0  }}
              transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Divider line above */}
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: 0.1, duration: 0.5 }}
                className="h-px w-32 mx-auto mb-5"
                style={{ background: 'linear-gradient(90deg, transparent, #8b5cf6, transparent)' }}
              />

              {/* HELIOS wordmark */}
              <h1
                style={{
                  fontFamily:    'Impact, "Arial Black", sans-serif',
                  fontSize:      'clamp(80px, 20vw, 240px)',
                  lineHeight:     1,
                  letterSpacing: '-0.02em',
                  color:         '#fff',
                  textShadow:
                    '0 0 80px rgba(139,92,246,0.9), 0 0 160px rgba(139,92,246,0.5), 0 0 260px rgba(139,92,246,0.2)',
                }}
              >
                HEL<span style={{ color: '#8b5cf6' }}>IOS</span>
              </h1>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showTagline && (
            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1,  y: 0  }}
              transition={{ duration: 0.5 }}
              className="mt-4 space-y-1"
            >
              <p
                className="font-mono text-sm tracking-[0.4em] uppercase"
                style={{ color: 'rgba(255,255,255,0.6)' }}
              >
                Distributed Multi-Agent AI
              </p>
              <p
                className="font-mono text-[10px] tracking-[0.6em] uppercase"
                style={{ color: 'rgba(139,92,246,0.5)' }}
              >
                ⟡ We Transcend Dimensions ⟡
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Progress bar */}
      <AnimatePresence>
        {showBar && (
          <motion.div
            className="absolute bottom-0 left-0 h-[2px] origin-left"
            style={{
              background: 'linear-gradient(90deg, #7c3aed, #8b5cf6, #a78bfa, #c026d3, #8b5cf6)',
              boxShadow: '0 0 14px rgba(139,92,246,0.9)',
            }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 3.2, ease: 'linear' }}
          />
        )}
      </AnimatePresence>

      {/* Bottom label */}
      <div
        className="absolute bottom-6 left-1/2 -translate-x-1/2 font-mono text-[9px] tracking-[0.8em] uppercase pointer-events-none"
        style={{ color: 'rgba(139,92,246,0.25)' }}
      >
        ⟡ INITIALIZING ⟡
      </div>
    </motion.div>
  )
}
