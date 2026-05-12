import { useEffect, useRef } from 'react'

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3)
const easeIn  = (t: number) => t * t * t

function drawCreature(ctx: CanvasRenderingContext2D, W: number, H: number, elapsed: number) {
  const cx   = W / 2
  const cy   = H / 2 + H * 0.025
  const size = Math.min(W, H) * 0.1

  const FORM_DUR   = 0.9
  const HOLD_END   = 2.1
  const TOTAL      = 3.0

  const formP     = Math.min(1, elapsed / FORM_DUR)
  const dissolveP = elapsed > HOLD_END ? Math.min(1, (elapsed - HOLD_END) / (TOTAL - HOLD_END)) : 0
  const gA        = easeOut(formP) * (1 - easeIn(dissolveP))

  if (gA <= 0.01) return

  const t = elapsed

  // ── Atmosphere ──────────────────────────────────────────────────────────
  const atm = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 6.5)
  atm.addColorStop(0,    `rgba(28,  0, 46, ${0.88 * gA})`)
  atm.addColorStop(0.35, `rgba(10,  0, 18, ${0.62 * gA})`)
  atm.addColorStop(0.7,  `rgba( 4,  0,  8, ${0.38 * gA})`)
  atm.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = atm
  ctx.fillRect(0, 0, W, H)

  // ── Main tentacles (12) ─────────────────────────────────────────────────
  for (let i = 0; i < 12; i++) {
    const baseAngle  = (i / 12) * Math.PI * 2
    const lenFactor  = 0.88 + (i % 4) * 0.13
    const tentLen    = size * 4.8 * lenFactor * easeOut(formP)
    const w1 = Math.sin(t * 1.1 + i * 2.09) * 0.28
    const w2 = Math.sin(t * 0.75 + i * 3.14) * 0.17

    const cp1x = cx + Math.cos(baseAngle + w2 * 0.5) * tentLen * 0.36
    const cp1y = cy + Math.sin(baseAngle + w2 * 0.5) * tentLen * 0.36
    const cp2x = cx + Math.cos(baseAngle + w2)       * tentLen * 0.72
    const cp2y = cy + Math.sin(baseAngle + w2)       * tentLen * 0.72
    const ex   = cx + Math.cos(baseAngle + w1)       * tentLen
    const ey   = cy + Math.sin(baseAngle + w1)       * tentLen

    const alpha = gA * (0.70 + Math.sin(t * 1.8 + i * 0.8) * 0.16)

    const grd = ctx.createLinearGradient(cx, cy, ex, ey)
    grd.addColorStop(0,   `rgba(55, 12, 82, ${alpha})`)
    grd.addColorStop(0.5, `rgba(22,  4, 38, ${alpha * 0.7})`)
    grd.addColorStop(1,   'rgba(0,0,0,0)')

    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, ex, ey)
    ctx.strokeStyle = grd
    ctx.lineWidth   = Math.max(1, (11.5 - (i % 5) * 1.6) * (1 - dissolveP * 0.85))
    ctx.lineCap     = 'round'
    ctx.stroke()
  }

  // ── Secondary thin tendrils ─────────────────────────────────────────────
  ctx.save()
  ctx.globalCompositeOperation = 'screen'
  for (let i = 0; i < 22; i++) {
    const a   = (i / 22) * Math.PI * 2 + t * 0.09 + Math.sin(t * 1.5 + i * 0.55) * 0.26
    const len = size * (1.9 + (i % 4) * 0.38) * easeOut(formP)
    const ex  = cx + Math.cos(a) * len
    const ey  = cy + Math.sin(a) * len
    const grd = ctx.createLinearGradient(cx, cy, ex, ey)
    grd.addColorStop(0, `rgba(100, 20, 162, ${0.22 * gA})`)
    grd.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.strokeStyle = grd
    ctx.lineWidth   = 0.8 + (i % 3) * 0.6
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(ex, ey)
    ctx.stroke()
  }
  ctx.restore()

  // ── Body blob ───────────────────────────────────────────────────────────
  const bodyA = formP > 0.35 ? easeOut(Math.min(1, (formP - 0.35) / 0.45)) : 0
  if (bodyA > 0.01) {
    ctx.beginPath()
    for (let i = 0; i <= 36; i++) {
      const angle = (i / 36) * Math.PI * 2
      const noise = Math.sin(angle * 3 + t * 2.2) * 0.10
                  + Math.sin(angle * 7 + t * 1.5) * 0.06
                  + Math.sin(angle * 11+ t * 0.9) * 0.03
      const r = size * (1 + noise) * (1 - dissolveP * 0.45)
      const x = cx + Math.cos(angle) * r
      const y = cy + Math.sin(angle) * r
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.closePath()

    const bGrd = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 1.3)
    bGrd.addColorStop(0,    `rgba(92, 22, 138, ${0.96 * bodyA * gA})`)
    bGrd.addColorStop(0.45, `rgba(46,  9,  70, ${0.90 * bodyA * gA})`)
    bGrd.addColorStop(0.85, `rgba(18,  3,  28, ${0.80 * bodyA * gA})`)
    bGrd.addColorStop(1,    `rgba( 5,  0,  10, ${0.60 * bodyA * gA})`)
    ctx.fillStyle = bGrd

    ctx.save()
    ctx.shadowColor = 'rgba(139,92,246,0.7)'
    ctx.shadowBlur  = 30 * bodyA
    ctx.fill()
    ctx.strokeStyle = `rgba(100,40,162, ${0.4 * bodyA * gA})`
    ctx.lineWidth   = 2
    ctx.stroke()
    ctx.restore()
  }

  // ── Eyes ────────────────────────────────────────────────────────────────
  const eyeA = formP > 0.68 ? easeOut(Math.min(1, (formP - 0.68) / 0.25)) : 0
  if (eyeA > 0.01) {
    const pulse   = 1 + Math.sin(t * 5.2) * 0.13
    const spacing = size * 0.44

    for (const side of [-1, 1]) {
      const ex = cx + side * spacing
      const ey = cy - size * 0.07

      // Blood halo
      const halo = ctx.createRadialGradient(ex, ey, 0, ex, ey, size * 0.6)
      halo.addColorStop(0,   `rgba(255, 35,  0, ${0.78 * eyeA * gA * (1 - dissolveP)})`)
      halo.addColorStop(0.4, `rgba(180,  8,  0, ${0.36 * eyeA * gA * (1 - dissolveP)})`)
      halo.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.fillStyle = halo
      ctx.beginPath()
      ctx.arc(ex, ey, size * 0.6, 0, Math.PI * 2)
      ctx.fill()

      // Iris
      ctx.save()
      ctx.shadowColor = 'rgba(255,80,0,1)'
      ctx.shadowBlur  = 22
      ctx.fillStyle   = `rgba(255, 88, 8, ${eyeA * gA * (1 - dissolveP)})`
      ctx.beginPath()
      ctx.ellipse(ex, ey, size * 0.19 * pulse, size * 0.13 * pulse, 0, 0, Math.PI * 2)
      ctx.fill()

      // Slit pupil
      ctx.fillStyle = `rgba(255, 215, 60, ${eyeA * gA})`
      ctx.beginPath()
      ctx.ellipse(ex, ey, size * 0.055, size * 0.11 * pulse, 0, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
    }
  }

  // ── Floating spore particles ─────────────────────────────────────────────
  if (formP > 0.5 && dissolveP < 0.8) {
    const sporeA = easeOut(Math.min(1, (formP - 0.5) / 0.3)) * (1 - dissolveP) * gA
    for (let i = 0; i < 40; i++) {
      const angle = (i / 40) * Math.PI * 2 + t * (0.15 + (i % 5) * 0.04)
      const dist  = size * (1.2 + (i % 7) * 0.45 + Math.sin(t * 1.2 + i) * 0.3)
      const px    = cx + Math.cos(angle) * dist
      const py    = cy + Math.sin(angle) * dist
      const r     = 1 + (i % 3) * 0.8
      ctx.fillStyle = `rgba(${100 + (i % 60)}, ${10 + (i % 20)}, ${150 + (i % 50)}, ${sporeA * 0.5})`
      ctx.beginPath()
      ctx.arc(px, py, r, 0, Math.PI * 2)
      ctx.fill()
    }
  }
}

export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef    = useRef(0)
  const startRef  = useRef<number | null>(null)
  const doneRef   = useRef(false)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx    = canvas.getContext('2d')!

    const resize = () => {
      canvas.width  = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const frame = (ts: number) => {
      if (!startRef.current) startRef.current = ts
      const elapsed = (ts - startRef.current) / 1000
      const W = canvas.width
      const H = canvas.height

      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, W, H)
      drawCreature(ctx, W, H, elapsed)

      if (elapsed < 3.0) {
        rafRef.current = requestAnimationFrame(frame)
      } else if (!doneRef.current) {
        doneRef.current = true
        onComplete()
      }
    }

    rafRef.current = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [onComplete])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        width: '100%',
        height: '100%',
        zIndex: 200,
        background: '#000',
        display: 'block',
      }}
    />
  )
}
