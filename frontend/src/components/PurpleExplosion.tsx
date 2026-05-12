import { useEffect, useRef } from 'react'
import { useScroll, useMotionValueEvent } from 'framer-motion'

// Seeded random so particles don't jump around on re-renders
function seededRandom(seed: number) {
  const x = Math.sin(seed + 1) * 10000
  return x - Math.floor(x)
}

const NUM_PARTICLES = 260

// Pre-generate particle data
const PARTICLES = Array.from({ length: NUM_PARTICLES }, (_, i) => ({
  angle:  seededRandom(i * 3.1)  * Math.PI * 2,
  speed:  seededRandom(i * 7.7)  * 0.6 + 0.4,
  size:   seededRandom(i * 13.3) * 4   + 1,
  hue:    Math.round(seededRandom(i * 17.9) * 60 + 260), // 260–320 (purple→fuchsia)
  layer:  seededRandom(i * 5.5),
}))

export function PurpleExplosion() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { scrollYProgress } = useScroll()

  const draw = (progress: number) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Size canvas to viewport
    const W = window.innerWidth
    const H = window.innerHeight
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width  = W
      canvas.height = H
    }

    ctx.clearRect(0, 0, W, H)

    // Only draw when in range
    if (progress < 0.04 || progress > 0.72) return

    // Normalise: 0 → appears, 1 → full blast, fade out toward end
    const p = Math.max(0, Math.min(1, (progress - 0.04) / 0.30))
    const fadeOut = progress > 0.55 ? 1 - (progress - 0.55) / 0.17 : 1
    const alpha   = Math.min(1, p * 2) * fadeOut

    if (alpha <= 0) return

    const cx = W / 2
    const cy = H / 2
    const maxR = Math.sqrt(cx * cx + cy * cy) * 1.3

    // ── Core radial glow ─────────────────────────────────────────────────────
    const r0 = maxR * p * 0.08
    const r1 = maxR * p
    const grd = ctx.createRadialGradient(cx, cy, r0, cx, cy, r1)
    grd.addColorStop(0.00, `rgba(220,100,255,${0.95 * alpha})`)
    grd.addColorStop(0.15, `rgba(180,60,240,${0.90 * alpha})`)
    grd.addColorStop(0.35, `rgba(139,92,246,${0.75 * alpha})`)
    grd.addColorStop(0.58, `rgba(109,40,217,${0.50 * alpha})`)
    grd.addColorStop(0.80, `rgba(76,29,149,${0.25 * alpha})`)
    grd.addColorStop(1.00, 'rgba(0,0,0,0)')
    ctx.fillStyle = grd
    ctx.fillRect(0, 0, W, H)

    // ── Secondary outer bloom ─────────────────────────────────────────────────
    if (p > 0.3) {
      const bloom = ctx.createRadialGradient(cx, cy, maxR * p * 0.5, cx, cy, maxR * p * 1.1)
      bloom.addColorStop(0, `rgba(192,38,211,${0.15 * alpha})`)
      bloom.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = bloom
      ctx.fillRect(0, 0, W, H)
    }

    // ── Tendrils (long thin streaks from center) ─────────────────────────────
    ctx.save()
    ctx.globalCompositeOperation = 'screen'
    const numTendrils = 18
    for (let i = 0; i < numTendrils; i++) {
      const angle = (i / numTendrils) * Math.PI * 2 + p * 0.3
      const len   = maxR * p * (0.5 + seededRandom(i * 2.3) * 0.7)
      const x2    = cx + Math.cos(angle) * len
      const y2    = cy + Math.sin(angle) * len
      const tGrd  = ctx.createLinearGradient(cx, cy, x2, y2)
      tGrd.addColorStop(0, `rgba(200, 80, 255, ${0.35 * alpha})`)
      tGrd.addColorStop(1, 'rgba(139,92,246,0)')
      ctx.strokeStyle = tGrd
      ctx.lineWidth   = 2 + seededRandom(i * 9.1) * 4
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(x2, y2)
      ctx.stroke()
    }
    ctx.restore()

    // ── Particles ────────────────────────────────────────────────────────────
    ctx.save()
    ctx.globalCompositeOperation = 'screen'
    for (const pt of PARTICLES) {
      const dist = maxR * p * pt.speed * (0.3 + pt.layer * 0.7)
      const x    = cx + Math.cos(pt.angle) * dist
      const y    = cy + Math.sin(pt.angle) * dist
      const a    = Math.max(0, (1 - dist / maxR) * alpha * 0.85)
      ctx.beginPath()
      ctx.arc(x, y, pt.size, 0, Math.PI * 2)
      ctx.fillStyle = `hsla(${pt.hue},90%,70%,${a})`
      ctx.fill()
    }
    ctx.restore()
  }

  useMotionValueEvent(scrollYProgress, 'change', draw)

  useEffect(() => {
    draw(scrollYProgress.get())
    const onResize = () => draw(scrollYProgress.get())
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 20 }}
    />
  )
}
