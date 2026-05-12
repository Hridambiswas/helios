import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { DragonScene } from '../three/DragonScene'

export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const [showTitle, setShowTitle] = useState(false)
  const [showSub,   setShowSub]   = useState(false)
  const [exiting,   setExiting]   = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setShowTitle(true), 1600)
    const t2 = setTimeout(() => setShowSub(true),   2100)
    const t3 = setTimeout(() => setExiting(true),   3000)
    const t4 = setTimeout(() => onComplete(),        3700)
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4) }
  }, [onComplete])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: exiting ? 0 : 1, y: exiting ? -60 : 0 }}
      transition={{ duration: exiting ? 0.7 : 0.55, ease: [0.76, 0, 0.24, 1] }}
      className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden"
      style={{ background: '#000' }}
    >
      {/* 3-D dragon fills entire background */}
      <div className="absolute inset-0 pointer-events-none">
        <DragonScene />
      </div>

      {/* Deep vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse at 50% 50%, transparent 30%, rgba(0,0,0,0.72) 75%, #000 100%)',
        }}
      />

      {/* HELIOS reveal */}
      <div className="relative z-10 text-center select-none pointer-events-none">
        <AnimatePresence>
          {showTitle && (
            <motion.h1
              initial={{ opacity: 0, scale: 0.86, y: 24 }}
              animate={{ opacity: 1,  scale: 1,    y: 0  }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              style={{
                fontFamily:   'Impact, "Arial Black", sans-serif',
                fontSize:     'clamp(80px, 18vw, 220px)',
                lineHeight:    1,
                letterSpacing: '-0.02em',
                color:         'transparent',
                WebkitTextStroke: '2px rgba(255,255,255,0.9)',
                textShadow:
                  '0 0 80px rgba(196,30,58,0.6), 0 0 160px rgba(196,30,58,0.25)',
              }}
            >
              HELIOS
            </motion.h1>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showSub && (
            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1,  y: 0  }}
              transition={{ duration: 0.5 }}
              className="font-mono text-sm tracking-[0.45em] uppercase mt-3"
              style={{ color: 'rgba(196,30,58,0.82)' }}
            >
              Distributed Multi-Agent AI
            </motion.p>
          )}
        </AnimatePresence>
      </div>

      {/* Progress bar */}
      <motion.div
        className="absolute bottom-0 left-0 h-[2px]"
        style={{
          background: 'linear-gradient(90deg, transparent, #c41e3a, #ff4444, #c41e3a, transparent)',
        }}
        initial={{ width: '0%' }}
        animate={{ width: '100%' }}
        transition={{ duration: 3.0, ease: 'linear' }}
      />
    </motion.div>
  )
}
