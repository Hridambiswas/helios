import { useEffect, useRef } from 'react'
import { useMotionValue, useSpring, motion } from 'framer-motion'

export function CustomCursor() {
  // Exact pointer position
  const mx = useMotionValue(-300)
  const my = useMotionValue(-300)

  // Blob: lags behind with spring
  const blobX = useSpring(mx, { damping: 22, stiffness: 280, mass: 0.5 })
  const blobY = useSpring(my, { damping: 22, stiffness: 280, mass: 0.5 })

  // Stretch + rotation from velocity
  const rawSX  = useMotionValue(1)
  const rawSY  = useMotionValue(1)
  const rawRot = useMotionValue(0)
  const scaleX = useSpring(rawSX,  { damping: 10, stiffness: 200, mass: 0.3 })
  const scaleY = useSpring(rawSY,  { damping: 10, stiffness: 200, mass: 0.3 })
  const rotate = useSpring(rawRot, { damping: 12, stiffness: 120, mass: 0.4 })

  const prevRef = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
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
      {/*
        CRITICAL: do NOT apply filter + mix-blend-mode on the same
        full-screen element — this triggers a WebKit/Safari bug that
        paints a solid black layer over the entire page.

        Fix: scope the SVG goo filter to a small (200×200) container
        that moves with the blob, and keep it completely separate from
        any blend-mode layers.
      */}

      {/* SVG filter definition — zero-size, invisible */}
      <svg
        aria-hidden="true"
        style={{
          position: 'fixed', width: 0, height: 0,
          overflow: 'hidden', pointerEvents: 'none',
        }}
      >
        <defs>
          <filter id="cursor-goo" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
            <feColorMatrix
              in="blur"
              mode="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -8"
              result="goo"
            />
            <feComposite in="SourceGraphic" in2="goo" operator="atop" />
          </filter>
        </defs>
      </svg>

      {/*
        Blob inside a small (200×200) filtered container.
        The filter never touches a full-viewport element → no Safari black bug.
        No mix-blend-mode anywhere near a filter.
      */}
      <motion.div
        style={{
          position: 'fixed',
          x: blobX,
          y: blobY,
          translateX: '-50%',
          translateY: '-50%',
          width: 200,
          height: 200,
          pointerEvents: 'none',
          zIndex: 9999,
          filter: 'url(#cursor-goo)',
        }}
      >
        <motion.div
          style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            translateX: '-50%',
            translateY: '-50%',
            scaleX,
            scaleY,
            rotate,
            width: 30,
            height: 30,
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.90)',
          }}
        />
      </motion.div>

      {/* Exact-position dot — outside filter, always crisp */}
      <motion.div
        style={{
          position: 'fixed',
          x: mx,
          y: my,
          translateX: '-50%',
          translateY: '-50%',
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: '#ffffff',
          pointerEvents: 'none',
          zIndex: 9999,
        }}
      />
    </>
  )
}
