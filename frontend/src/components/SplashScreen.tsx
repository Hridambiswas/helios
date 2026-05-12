import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

// ── Easing ────────────────────────────────────────────────────────────────────
const lerp    = (a: number, b: number, t: number) => a + (b - a) * t
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3)
const easeIn  = (t: number) => t * t * t

// ── Creature renderer ─────────────────────────────────────────────────────────
function drawScene(ctx: CanvasRenderingContext2D, W: number, H: number, elapsed: number) {
  const cx = W / 2

  // Timeline
  const RISE_DUR  = 0.55
  const MOUTH_AT  = 0.90
  const MOUTH_DUR = 1.05
  const FADE_AT   = 2.45

  const riseP  = easeOut(Math.min(1, elapsed / RISE_DUR))
  const bodyA  = easeOut(Math.min(1, elapsed / 0.75))
  const mouthP = elapsed > MOUTH_AT
    ? easeOut(Math.min(1, (elapsed - MOUTH_AT) / MOUTH_DUR))
    : 0
  const fadeP  = elapsed > FADE_AT
    ? easeIn(Math.min(1, (elapsed - FADE_AT) / (3.0 - FADE_AT)))
    : 0

  const gA   = bodyA * (1 - fadeP)
  const yOff = (1 - riseP) * H * 0.12
  const sway = Math.sin(elapsed * 1.1) * (2.5 + mouthP * 2.5)

  // Creature metrics
  const creatureH = Math.min(H * 0.82, W * 1.55)
  const footY     = H * 0.94 + yOff
  const headCY    = footY - creatureH * 0.91
  const headR     = creatureH * 0.105
  const torsoW    = creatureH * 0.155
  const hipW      = creatureH * 0.125
  const legW      = creatureH * 0.052
  const armW      = creatureH * 0.036
  const torsoTop  = headCY + headR * 2.6
  const hipY      = footY - creatureH * 0.36
  const kneeY     = footY - creatureH * 0.19

  // ── Atmospheric glow ──────────────────────────────────────────────────────
  const atmCY = headCY + creatureH * 0.42
  const atmR  = creatureH * 0.72
  const atm   = ctx.createRadialGradient(cx, atmCY, 0, cx, atmCY, atmR)
  atm.addColorStop(0,    `rgba(40, 4, 4, ${0.68 * gA})`)
  atm.addColorStop(0.35, `rgba(16, 1, 1, ${0.48 * gA})`)
  atm.addColorStop(0.70, `rgba( 5, 0, 0, ${0.28 * gA})`)
  atm.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = atm
  ctx.fillRect(0, 0, W, H)

  ctx.save()
  ctx.translate(sway, 0)
  ctx.globalAlpha = gA

  // ── Body silhouette ───────────────────────────────────────────────────────
  ctx.beginPath()
  ctx.moveTo(cx - torsoW * 0.30, torsoTop)
  ctx.bezierCurveTo(cx - torsoW, torsoTop + creatureH * 0.02, cx - torsoW * 1.05, hipY - creatureH * 0.08, cx - hipW * 0.95, hipY)
  ctx.bezierCurveTo(cx - hipW * 1.0, kneeY - creatureH * 0.03, cx - legW * 2.8, kneeY + creatureH * 0.07, cx - legW * 1.6, footY)
  ctx.lineTo(cx + legW * 1.6, footY)
  ctx.bezierCurveTo(cx + legW * 2.8, kneeY + creatureH * 0.07, cx + hipW * 1.0, kneeY - creatureH * 0.03, cx + hipW * 0.95, hipY)
  ctx.bezierCurveTo(cx + torsoW * 1.05, hipY - creatureH * 0.08, cx + torsoW, torsoTop + creatureH * 0.02, cx + torsoW * 0.30, torsoTop)
  ctx.closePath()

  const bGrd = ctx.createLinearGradient(cx - torsoW, torsoTop, cx, footY)
  bGrd.addColorStop(0,   'rgba(34, 21, 15, 0.97)')
  bGrd.addColorStop(0.5, 'rgba(22, 13,  9, 0.97)')
  bGrd.addColorStop(1,   'rgba(13,  8,  5, 0.97)')
  ctx.fillStyle = bGrd
  ctx.fill()

  // ── Arms ──────────────────────────────────────────────────────────────────
  const armTopY = torsoTop + creatureH * 0.025
  const armEndY = hipY + creatureH * 0.195

  for (const side of [-1, 1] as const) {
    const sx   = side * torsoW * 0.85
    const midX = side * torsoW * 1.45
    const endX = side * torsoW * 1.58

    ctx.beginPath()
    ctx.moveTo(cx + sx, armTopY)
    ctx.bezierCurveTo(cx + midX, armTopY + creatureH * 0.11, cx + endX, armEndY - creatureH * 0.07, cx + endX, armEndY)
    ctx.lineTo(cx + endX - side * armW * 2, armEndY)
    ctx.bezierCurveTo(cx + endX - side * armW * 2, armEndY - creatureH * 0.07, cx + midX - side * armW, armTopY + creatureH * 0.11, cx + sx - side * armW, armTopY)
    ctx.closePath()
    ctx.fillStyle = 'rgba(28, 17, 11, 0.97)'
    ctx.fill()

    // Claws
    for (let k = 0; k < 4; k++) {
      const clawX = cx + endX + side * (k - 1.5) * armW * 0.9 - side * armW
      ctx.beginPath()
      ctx.moveTo(clawX - side * armW * 0.3, armEndY)
      ctx.bezierCurveTo(clawX - side * armW * 0.4, armEndY + armW, clawX + side * armW * 0.05, armEndY + armW * 1.9, clawX + side * armW * 0.35, armEndY + armW * 1.6)
      ctx.lineTo(clawX + side * armW * 0.5, armEndY)
      ctx.closePath()
      ctx.fillStyle = 'rgba(16, 10, 6, 0.97)'
      ctx.fill()
    }
  }

  // ── Neck ──────────────────────────────────────────────────────────────────
  const neckW   = creatureH * 0.065
  const neckTop = headCY + headR * 1.05
  const neckBot = torsoTop + creatureH * 0.005

  ctx.beginPath()
  ctx.moveTo(cx - neckW, neckBot)
  ctx.lineTo(cx - neckW * 0.75, neckTop)
  ctx.lineTo(cx + neckW * 0.75, neckTop)
  ctx.lineTo(cx + neckW, neckBot)
  ctx.closePath()
  ctx.fillStyle = 'rgba(30, 18, 12, 0.97)'
  ctx.fill()

  // ── Petal Mouth (the iconic Demogorgon feature) ───────────────────────────
  const NUM         = 5
  const openLen     = headR * 1.95
  const closedLen   = headR * 0.16
  const openW       = headR * 0.74
  const closedW     = headR * 0.38
  const petalLen    = lerp(closedLen, openLen, mouthP)
  const petalW      = lerp(closedW, openW, mouthP)

  // Draw in back-to-front order: bottom petals drawn first
  const drawOrder = [2, 3, 4, 0, 1]

  for (const i of drawOrder) {
    const angle = (i / NUM) * Math.PI * 2 - Math.PI / 2

    ctx.save()
    ctx.translate(cx, headCY)
    ctx.rotate(angle)

    // Petal shape — pointed teardrop
    ctx.beginPath()
    ctx.moveTo(-petalW * 0.42, 0)
    ctx.bezierCurveTo(-petalW, petalLen * 0.24, -petalW * 0.88, petalLen * 0.64, -petalW * 0.12, petalLen)
    ctx.bezierCurveTo(-petalW * 0.04, petalLen + petalW * 0.13, petalW * 0.04, petalLen + petalW * 0.13, petalW * 0.12, petalLen)
    ctx.bezierCurveTo(petalW * 0.88, petalLen * 0.64, petalW, petalLen * 0.24, petalW * 0.42, 0)
    ctx.closePath()

    const red = Math.round(lerp(20, 98, mouthP))
    const pg  = ctx.createLinearGradient(0, 0, 0, petalLen)
    pg.addColorStop(0,   `rgba(${42 + red}, 16, 10, 0.97)`)
    pg.addColorStop(0.4, `rgba(${55 + red}, 20, 12, 0.95)`)
    pg.addColorStop(1,   `rgba(${30 + red}, 10,  6, 0.92)`)
    ctx.fillStyle = pg
    ctx.fill()

    // Ridge down centre of petal
    if (mouthP > 0.04) {
      ctx.beginPath()
      ctx.moveTo(0, petalW * 0.06)
      ctx.lineTo(0, petalLen * 0.9)
      ctx.strokeStyle = `rgba(10, 4, 2, ${0.4 + mouthP * 0.3})`
      ctx.lineWidth   = Math.max(0.5, petalW * 0.12)
      ctx.stroke()
    }

    // Teeth — appear once mouth is 25% open
    if (mouthP > 0.25) {
      const toothA = Math.min(1, (mouthP - 0.25) / 0.4)
      for (let j = 1; j <= 4; j++) {
        const tp = j / 5
        const ty = tp * petalLen * 0.88
        const tw = Math.sin(tp * Math.PI) * petalW * 0.58
        const tl = petalW * 0.23 * (1 - tp * 0.15)
        for (const s of [-1, 1] as const) {
          ctx.beginPath()
          ctx.moveTo(s * tw, ty)
          ctx.lineTo(s * (tw + tl), ty - tl * 0.55)
          ctx.lineTo(s * (tw + tl * 0.28), ty + tl * 0.65)
          ctx.closePath()
          ctx.fillStyle = `rgba(220, 206, 185, ${toothA * 0.88})`
          ctx.fill()
        }
      }
    }

    ctx.restore()
  }

  // ── Inner mouth void ──────────────────────────────────────────────────────
  {
    const voidR = lerp(headR * 0.22, headR * 0.90, mouthP)
    const vg    = ctx.createRadialGradient(cx, headCY, 0, cx, headCY, voidR)
    vg.addColorStop(0,    `rgba(0, 0, 0, ${0.28 + mouthP * 0.68})`)
    vg.addColorStop(0.25, `rgba(32, 2, 0, ${mouthP * 0.9})`)
    vg.addColorStop(0.55, `rgba(72, 9, 3, ${mouthP * 0.72})`)
    vg.addColorStop(1,    'rgba(0,0,0,0)')
    ctx.fillStyle = vg
    ctx.beginPath()
    ctx.arc(cx, headCY, voidR, 0, Math.PI * 2)
    ctx.fill()
  }

  // ── Slime drips (after mouth is mostly open) ──────────────────────────────
  if (mouthP > 0.65 && elapsed < FADE_AT) {
    const slimeA = Math.min(1, (mouthP - 0.65) / 0.28)
    for (let d = 0; d < 5; d++) {
      const dX    = cx + (d - 2) * headR * 0.4
      const dBase = headCY + openLen * mouthP * 0.44 + headR * 0.08
      const dLen  = (Math.sin(elapsed * 0.75 + d * 1.3) * 0.5 + 0.5) * headR * 0.28
      ctx.beginPath()
      ctx.moveTo(dX, dBase)
      ctx.lineTo(dX + Math.sin(elapsed + d) * 2.5, dBase + dLen)
      ctx.strokeStyle = `rgba(88, 13, 5, ${slimeA * 0.52})`
      ctx.lineWidth   = 1.5 + d % 2
      ctx.stroke()
      ctx.beginPath()
      ctx.arc(dX + Math.sin(elapsed + d) * 2.5, dBase + dLen, 2.8, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(88, 13, 5, ${slimeA * 0.42})`
      ctx.fill()
    }
  }

  // ── Dark spores orbiting creature ────────────────────────────────────────
  if (bodyA > 0.35) {
    const sporeA = easeOut(Math.min(1, (bodyA - 0.35) / 0.5)) * (1 - fadeP)
    for (let i = 0; i < 30; i++) {
      const a    = (i / 30) * Math.PI * 2 + elapsed * (0.09 + (i % 5) * 0.025)
      const dist = creatureH * (0.11 + (i % 8) * 0.028 + Math.sin(elapsed * 0.65 + i) * 0.018)
      const px   = cx + Math.cos(a) * dist * 1.85
      const py   = headCY + creatureH * 0.44 + Math.sin(a) * dist * 0.88
      ctx.fillStyle = `hsla(5, 65%, ${14 + (i % 14)}%, ${sporeA * 0.42})`
      ctx.beginPath()
      ctx.arc(px, py, 1.1 + (i % 4) * 0.4, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  ctx.restore()
}

// ── Component ─────────────────────────────────────────────────────────────────
export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef    = useRef(0)
  const startRef  = useRef<number | null>(null)
  const doneRef   = useRef(false)
  const fadingRef = useRef(false)
  const [fading, setFading] = useState(false)

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

      if (elapsed >= 2.45 && !fadingRef.current) {
        fadingRef.current = true
        setFading(true)
      }

      const W = canvas.width
      const H = canvas.height
      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, W, H)
      drawScene(ctx, W, H, elapsed)

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
    <motion.div
      style={{ position: 'fixed', inset: 0, zIndex: 200, background: '#000' }}
      animate={{ opacity: fading ? 0 : 1 }}
      transition={{ duration: 0.55, ease: 'easeInOut' }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </motion.div>
  )
}
