import { useEffect, useRef } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

export function LiquidCursor() {
  const mx = useMotionValue(-300)
  const my = useMotionValue(-300)

  // Snappy inner dot
  const dotX = useSpring(mx, { damping: 35, stiffness: 700 })
  const dotY = useSpring(my, { damping: 35, stiffness: 700 })

  // Laggy outer ring — the "liquid" drag
  const ringX = useSpring(mx, { damping: 22, stiffness: 160 })
  const ringY = useSpring(my, { damping: 22, stiffness: 160 })

  // Blob that lags even more
  const blobX = useSpring(mx, { damping: 18, stiffness: 80 })
  const blobY = useSpring(my, { damping: 18, stiffness: 80 })

  useEffect(() => {
    const move = (e: MouseEvent) => { mx.set(e.clientX); my.set(e.clientY) }
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [mx, my])

  return (
    <>
      {/* Trailing venom blob */}
      <motion.div
        style={{
          position: 'fixed',
          x: blobX, y: blobY,
          translateX: '-50%', translateY: '-50%',
          width: 48, height: 48,
          borderRadius: '50%',
          background: 'rgba(255,255,255,0.06)',
          pointerEvents: 'none',
          zIndex: 9998,
          mixBlendMode: 'difference',
        }}
      />

      {/* Outer ring */}
      <motion.div
        style={{
          position: 'fixed',
          x: ringX, y: ringY,
          translateX: '-50%', translateY: '-50%',
          width: 28, height: 28,
          borderRadius: '50%',
          border: '1px solid rgba(255,255,255,0.5)',
          pointerEvents: 'none',
          zIndex: 9999,
          mixBlendMode: 'difference',
        }}
      />

      {/* Inner dot */}
      <motion.div
        style={{
          position: 'fixed',
          x: dotX, y: dotY,
          translateX: '-50%', translateY: '-50%',
          width: 4, height: 4,
          borderRadius: '50%',
          background: '#fff',
          pointerEvents: 'none',
          zIndex: 9999,
          mixBlendMode: 'difference',
        }}
      />
    </>
  )
}
