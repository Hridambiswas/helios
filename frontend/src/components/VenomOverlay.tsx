import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { FluidBackground } from './FluidBackground'

// 24-point clip-path keyframes — same point count in every frame so Framer
// Motion can interpolate each coordinate pair individually.
const P_FULL = `polygon(
  0% 0%,   17% 0%,  33% 0%,  50% 0%,  67% 0%,  83% 0%,
  100% 0%, 100% 17%,100% 33%,100% 50%,100% 67%,100% 83%,
  100% 100%,83% 100%,67% 100%,50% 100%,33% 100%,17% 100%,
  0% 100%, 0% 83%,  0% 67%,  0% 50%,  0% 33%,  0% 17%
)`

// Slightly past viewport – the "aggressive expand"
const P_EXPAND = `polygon(
  -7% -7%,  17% -7%,  33% -7%,  50% -7%,  67% -7%,  83% -7%,
  107% -7%, 107% 17%, 107% 33%, 107% 50%, 107% 67%, 107% 83%,
  107% 107%, 83% 107%, 67% 107%, 50% 107%, 33% 107%, 17% 107%,
  -7% 107%,-7% 83%,  -7% 67%,  -7% 50%,  -7% 33%,  -7% 17%
)`

// Same 24 points but mid-edge points bitten inward — jagged dissolution
const P_JAGGED = `polygon(
  0% 0%,   17% 17%, 33% 5%,  50% 14%, 67% 6%,  83% 18%,
  100% 0%, 85% 17%, 95% 33%, 87% 50%, 96% 67%, 86% 83%,
  100% 100%, 83% 85%, 67% 95%, 50% 88%, 33% 95%, 17% 86%,
  0% 100%, 13% 83%, 5% 67%,  14% 50%, 5% 33%,  15% 17%
)`

// All 24 points collapsed to center
const P_NONE = `polygon(
  50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%,
  50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%,
  50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%,
  50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%, 50% 50%
)`

interface Props {
  onComplete: () => void
}

export function VenomOverlay({ onComplete }: Props) {
  const [dissolving, setDissolving] = useState(false)
  const mouseRef = useRef({ x: 0.5, y: 0.5 })

  useEffect(() => {
    const move = (e: MouseEvent) => {
      mouseRef.current = {
        x: e.clientX / window.innerWidth,
        y: 1 - e.clientY / window.innerHeight,
      }
    }
    window.addEventListener('mousemove', move, { passive: true })

    // 3 s idle → start dissolve; 4.4 s total → hand off to next page
    const t1 = setTimeout(() => setDissolving(true), 3000)
    const t2 = setTimeout(() => onComplete(), 4400)

    return () => {
      window.removeEventListener('mousemove', move)
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [onComplete])

  return (
    <motion.div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        background: '#000',
        clipPath: P_FULL,
      }}
      animate={
        dissolving
          ? { clipPath: [P_FULL, P_EXPAND, P_JAGGED, P_NONE] }
          : { clipPath: P_FULL }
      }
      transition={
        dissolving
          ? { duration: 1.4, times: [0, 0.12, 0.52, 1.0], ease: 'easeInOut' }
          : { duration: 0 }
      }
    >
      <FluidBackground mouseRef={mouseRef} />

      {/* Eyebrow text */}
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        pointerEvents: 'none',
      }}>
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: dissolving ? 0 : 0.18 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          style={{
            fontFamily: '"IBM Plex Mono", monospace',
            fontSize: 10,
            letterSpacing: '0.5em',
            textTransform: 'uppercase',
            color: '#fff',
          }}
        >
          HELIOS
        </motion.span>
      </div>
    </motion.div>
  )
}
