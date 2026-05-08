import { useEffect, useRef } from 'react'

interface Ember {
  x: number
  y: number
  size: number
  opacity: number
  speed: number
  drift: number
  hue: number
  age: number
  maxAge: number
}

function spawnEmber(w: number, h: number): Ember {
  return {
    x: Math.random() * w,
    y: h + 10,
    size: 1 + Math.random() * 2.5,
    opacity: 0,
    speed: 0.3 + Math.random() * 0.7,
    drift: (Math.random() - 0.5) * 0.8,
    hue: Math.random() > 0.7 ? 45 : 0,   // gold or crimson
    age: 0,
    maxAge: 200 + Math.random() * 300,
  }
}

export function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const embers: Ember[] = []
    let raf: number

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = document.body.scrollHeight
    }
    resize()

    const resizeObs = new ResizeObserver(resize)
    resizeObs.observe(document.body)

    // Seed
    for (let i = 0; i < 40; i++) {
      const e = spawnEmber(canvas.width, canvas.height)
      e.y = Math.random() * canvas.height
      e.age = Math.random() * e.maxAge
      embers.push(e)
    }

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (embers.length < 80) embers.push(spawnEmber(canvas.width, canvas.height))

      for (let i = embers.length - 1; i >= 0; i--) {
        const e = embers[i]
        const life = e.age / e.maxAge
        e.opacity = Math.sin(life * Math.PI) * 0.6

        ctx.beginPath()
        ctx.arc(e.x, e.y, e.size, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${e.hue}, 90%, 60%, ${e.opacity})`
        ctx.shadowBlur = e.size * 4
        ctx.shadowColor = `hsla(${e.hue}, 90%, 60%, ${e.opacity * 0.5})`
        ctx.fill()
        ctx.shadowBlur = 0

        e.x += e.drift + Math.sin(e.age * 0.02) * 0.4
        e.y -= e.speed
        e.age++

        if (e.age >= e.maxAge || e.y < -10) embers.splice(i, 1)
      }

      raf = requestAnimationFrame(tick)
    }
    tick()

    return () => {
      cancelAnimationFrame(raf)
      resizeObs.disconnect()
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
        opacity: 0.5,
      }}
    />
  )
}
