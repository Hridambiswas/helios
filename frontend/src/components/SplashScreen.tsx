import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { DragonScene } from '../three/DragonScene'

export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const [showTitle,   setShowTitle]   = useState(false)
  const [showTagline, setShowTagline] = useState(false)
  const [showBar,     setShowBar]     = useState(false)
  const [exiting,     setExiting]     = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setShowBar(true),     100)
    const t2 = setTimeout(() => setShowTitle(true),   1500)
    const t3 = setTimeout(() => setShowTagline(true), 2000)
    const t4 = setTimeout(() => setExiting(true),     3000)
    const t5 = setTimeout(() => onComplete(),          3750)
    return () => { [t1,t2,t3,t4,t5].forEach(clearTimeout) }
  }, [onComplete])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: exiting ? 0 : 1, y: exiting ? -80 : 0 }}
      transition={{ duration: exiting ? 0.75 : 0.5, ease: [0.76, 0, 0.24, 1] }}
      className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden"
      style={{ background: '#000' }}
    >
      {/* Full-screen dragon — bigger, closer, filling the viewport */}
      <div className="absolute inset-0 pointer-events-none">
        <DragonScene />
      </div>

      {/* Dragon scale background pattern */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.04]">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="splashScales" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse">
              <path d="M0,26 Q15,0 30,26 Q45,0 60,26" fill="none" stroke="#C9A227" strokeWidth="0.8"/>
              <path d="M-30,52 Q-15,26 0,52 Q15,26 30,52 Q45,26 60,52 Q75,26 90,52" fill="none" stroke="#C9A227" strokeWidth="0.8"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#splashScales)"/>
        </svg>
      </div>

      {/* Radial vignette to focus on dragon */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, transparent 28%, rgba(0,0,0,0.65) 70%, #000 100%)',
        }}
      />

      {/* Corner claw marks */}
      {[
        { corner: 'tl', style: { top: 0, left: 0 } },
        { corner: 'tr', style: { top: 0, right: 0, transform: 'scaleX(-1)' } },
      ].map(({ corner, style }) => (
        <div key={corner} className="absolute pointer-events-none opacity-[0.12]"
          style={{ ...style, width: 120, height: 120 }}>
          <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
            <path d="M5,5 Q30,20 20,55"  stroke="#C9A227" strokeWidth="2.5" strokeLinecap="round"/>
            <path d="M22,5 Q45,18 38,52" stroke="#C9A227" strokeWidth="2"   strokeLinecap="round"/>
            <path d="M40,5 Q60,16 58,48" stroke="#C9A227" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>
      ))}

      {/* HELIOS wordmark */}
      <div className="relative z-10 text-center select-none pointer-events-none">
        <AnimatePresence>
          {showTitle && (
            <motion.div
              initial={{ opacity: 0, scale: 0.84, y: 28 }}
              animate={{ opacity: 1,  scale: 1,    y: 0  }}
              transition={{ duration: 0.72, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Dragon eye ornament above title */}
              <div className="flex items-center justify-center gap-4 mb-3">
                <div className="h-px w-16 bg-gradient-to-r from-transparent to-gold/60" />
                <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
                  <ellipse cx="16" cy="16" rx="14" ry="9" stroke="#C9A227" strokeWidth="1.5"/>
                  <ellipse cx="16" cy="16" rx="5"  ry="9" fill="#C9A227" fillOpacity="0.2" stroke="#C9A227" strokeWidth="1.2"/>
                  <circle  cx="16" cy="16" r="3.5" fill="#FF6B00"/>
                  <circle  cx="16" cy="16" r="1.8" fill="#000"/>
                </svg>
                <div className="h-px w-16 bg-gradient-to-l from-transparent to-gold/60" />
              </div>

              <h1 style={{
                fontFamily:   'Impact, "Arial Black", sans-serif',
                fontSize:     'clamp(80px, 19vw, 230px)',
                lineHeight:    1,
                letterSpacing: '-0.02em',
                color:         'transparent',
                WebkitTextStroke: '2px #C9A227',
                textShadow:
                  '0 0 80px rgba(201,162,39,0.7), 0 0 160px rgba(201,162,39,0.3), 0 0 240px rgba(255,107,0,0.15)',
              }}>
                HELIOS
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
              className="mt-3 space-y-1"
            >
              <p className="font-mono text-sm tracking-[0.45em] uppercase"
                style={{ color: 'rgba(201,162,39,0.85)' }}>
                Distributed Multi-Agent AI
              </p>
              <p className="font-mono text-[10px] tracking-[0.6em] uppercase"
                style={{ color: 'rgba(255,107,0,0.55)' }}>
                ⟡ Born of Dragon Fire ⟡
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Gold progress bar */}
      <AnimatePresence>
        {showBar && (
          <motion.div
            className="absolute bottom-0 left-0 h-[2px] origin-left"
            style={{
              background: 'linear-gradient(90deg, #8B6914, #C9A227, #FFD700, #FF6B00, #C9A227, #8B6914)',
              boxShadow: '0 0 12px rgba(201,162,39,0.8)',
            }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 3.0, ease: 'linear' }}
          />
        )}
      </AnimatePresence>

      {/* Bottom rune-like text */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 font-mono text-[9px] tracking-[0.8em] uppercase pointer-events-none"
        style={{ color: 'rgba(201,162,39,0.25)' }}>
        ⟡ ⟡ ⟡ IGNITING ⟡ ⟡ ⟡
      </div>
    </motion.div>
  )
}
