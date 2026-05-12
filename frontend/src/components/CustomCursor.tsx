import { useEffect, useRef } from 'react'
import { useMotionValue, useSpring, motion } from 'framer-motion'

export function CustomCursor() {
  // Exact mouse position (for the small trailing dot)
  const mx = useMotionValue(-300)
  const my = useMotionValue(-300)

  // Blob position: lags behind with spring physics
  const blobX = useSpring(mx, { damping: 22, stiffness: 280, mass: 0.5 })
  const blobY = useSpring(my, { damping: 22, stiffness: 280, mass: 0.5 })

  // Stretch: raw values → smoothed via spring
  const rawSX  = useMotionValue(1)
  const rawSY  = useMotionValue(1)
  const rawRot = useMotionValue(0)
  const scaleX = useSpring(rawSX,  { damping: 10, stiffness: 200, mass: 0.3 })
  const scaleY = useSpring(rawSY,  { damping: 10, stiffness: 200, mass: 0.3 })
  const rotate = useSpring(rawRot, { damping: 12, stiffness: 120, mass: 0.4 })

  const prevRef = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      // Initialise on first event to avoid giant velocity jump
      if (!prevRef.current) {
        prevRef.current = { x: e.clientX, y: e.clientY }
        mx.set(e.clientX)
        my.set(e.clientY)
        return
      }

      const vx = e.clientX - prevRef.current.x
      const vy = e.clientY - prevRef.current.y
      prevRef.current = { x: e.clientX, y: e.clientY }

      mx.set(e.clientX)
      my.set(e.clientY)

      const speed = Math.sqrt(vx * vx + vy * vy)
      if (speed > 2) {
        const stretch = Math.min(1 + speed * 0.10, 2.6)
        rawSX.set(stretch)
        rawSY.set(Math.max(0.38, 1 / Math.sqrt(stretch)))
        rawRot.set(Math.atan2(vy, vx) * (180 / Math.PI))
      } else {
        rawSX.set(1)
        rawSY.set(1)
      }
    }

    window.addEventListener('mousemove', onMove, { passive: true })
    return () => window.removeEventListener('mousemove', onMove)
  }, [mx, my, rawSX, rawSY, rawRot])

  return (
    <>
      {/* SVG filter definition — zero-size, just provides the filter */}
      <svg
        aria-hidden="true"
        style={{ position: 'fixed', width: 0, height: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: -1 }}
      >
        <defs>
          <filter id="cursor-goo" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="blur" />
            <feColorMatrix
              in="blur"
              mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 22 -9"
              result="goo"
            />
            <feComposite in="SourceGraphic" in2="goo" operator="atop" />
          </filter>
        </defs>
      </svg>

      {/* Cursor layer:
          - pointer-events: none  → clicks pass through to page elements
          - z-index: 9999         → always on top
          - mix-blend-mode: difference → white on black = white, white on white = black (always visible)
      */}
      <div
        style={{
          position: 'fixed',
          top: 0, left: 0,
          width: '100%', height: '100%',
          pointerEvents: 'none',
          zIndex: 9999,
        }}
      >
        {/* Goo-filtered container — merges nearby circles into blobs */}
        <div
          style={{
            position: 'absolute', inset: 0,
            filter: 'url(#cursor-goo)',
            mixBlendMode: 'difference',
          }}
        >
          {/* Large blob — trails behind the cursor */}
          <motion.div
            style={{
              position: 'absolute',
              x: blobX,
              y: blobY,
              translateX: '-50%',
              translateY: '-50%',
              scaleX,
              scaleY,
              rotate,
              width: 34,
              height: 34,
              borderRadius: '50%',
              background: '#ffffff',
            }}
          />

          {/* Small dot — follows the pointer exactly */}
          <motion.div
            style={{
              position: 'absolute',
              x: mx,
              y: my,
              translateX: '-50%',
              translateY: '-50%',
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: '#ffffff',
            }}
          />
        </div>
      </div>
    </>
  )
}
