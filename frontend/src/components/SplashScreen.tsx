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
    const t2 = setTimeout(() => setShowTitle(true),   1200)
    const t3 = setTimeout(() => setShowTagline(true), 1900)
    const t4 = setTimeout(() => setExiting(true),     3100)
    const t5 = setTimeout(() => onComplete(),          3900)
    return () => { [t1, t2, t3, t4, t5].forEach(clearTimeout) }
  }, [onComplete])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: exiting ? 0 : 1, scale: exiting ? 1.03 : 1 }}
      transition={{ duration: exiting ? 0.8 : 0.5, ease: [0.76, 0, 0.24, 1] }}
      className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden bg-black"
    >
      {/* 3D shattered crystal fills the splash */}
      <div className="absolute inset-0 pointer-events-none">
        <HeroScene />
      </div>

      {/* Subtle dark vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, transparent 20%, rgba(0,0,0,0.55) 65%, #000 100%)',
        }}
      />

      {/* Content */}
      <div className="relative z-10 text-center select-none pointer-events-none">
        <AnimatePresence>
          {showTitle && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1,  y: 0  }}
              transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
            >
              <h1
                style={{
                  fontFamily:    'Impact, "Arial Black", sans-serif',
                  fontSize:      'clamp(80px, 20vw, 240px)',
                  lineHeight:     1,
                  letterSpacing: '-0.02em',
                  color:         '#ffffff',
                }}
              >
                HEL<span style={{ color: '#a78bfa' }}>IOS</span>
              </h1>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showTagline && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1,  y: 0  }}
              transition={{ duration: 0.5 }}
              className="mt-4 space-y-1"
            >
              <p className="font-mono text-sm tracking-[0.4em] uppercase" style={{ color: 'rgba(255,255,255,0.4)' }}>
                Distributed Multi-Agent AI
              </p>
              <p className="font-mono text-[10px] tracking-[0.5em] uppercase" style={{ color: 'rgba(167,139,250,0.45)' }}>
                Born of Intelligence
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Progress bar */}
      <AnimatePresence>
        {showBar && (
          <motion.div
            className="absolute bottom-0 left-0 h-[1px] origin-left"
            style={{
              background: 'linear-gradient(90deg, transparent, #8b5cf6, #c026d3, #8b5cf6, transparent)',
              boxShadow: '0 0 10px rgba(139,92,246,0.8)',
            }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 3.1, ease: 'linear' }}
          />
        )}
      </AnimatePresence>
    </motion.div>
  )
}
