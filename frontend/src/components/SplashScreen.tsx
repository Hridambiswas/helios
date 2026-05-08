import { useEffect, useRef, useState, useCallback } from 'react'
import oniMask from '../assets/oni-mask.png'

interface Particle {
  x: number
  y: number
  radius: number
  opacity: number
  speed: number
  drift: number
  age: number
  maxAge: number
}

function createParticle(width: number, height: number): Particle {
  return {
    x: Math.random() * width,
    y: height + Math.random() * 100,
    radius: 4 + Math.random() * 20,
    opacity: 0.05 + Math.random() * 0.18,
    speed: 0.4 + Math.random() * 0.8,
    drift: (Math.random() - 0.5) * 0.6,
    age: 0,
    maxAge: 180 + Math.random() * 200,
  }
}

export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskRef = useRef<HTMLDivElement>(null)
  const particlesRef = useRef<Particle[]>([])
  const rafRef = useRef<number>(0)

  const [phase, setPhase] = useState<'enter' | 'glow' | 'exit' | 'done'>('enter')
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  // Smoke canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    // Seed initial particles
    for (let i = 0; i < 60; i++) {
      const p = createParticle(canvas.width, canvas.height)
      p.y = Math.random() * canvas.height  // spread initial smoke across screen
      p.age = Math.random() * p.maxAge
      particlesRef.current.push(p)
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Spawn new particles
      if (particlesRef.current.length < 120) {
        particlesRef.current.push(createParticle(canvas.width, canvas.height))
      }

      particlesRef.current = particlesRef.current.filter(p => {
        const lifeRatio = p.age / p.maxAge
        const alpha = p.opacity * Math.sin(lifeRatio * Math.PI)

        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius)
        grad.addColorStop(0, `rgba(180, 180, 180, ${alpha})`)
        grad.addColorStop(1, `rgba(80, 0, 0, 0)`)

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
        ctx.fillStyle = grad
        ctx.fill()

        p.y -= p.speed
        p.x += p.drift + Math.sin(p.age * 0.03) * 0.3
        p.radius += 0.08
        p.age++

        return p.age < p.maxAge && p.y > -100
      })

      rafRef.current = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [])

  // Phase timing
  useEffect(() => {
    const t1 = setTimeout(() => setPhase('glow'), 1400)
    const t2 = setTimeout(() => setPhase('exit'), 3000)
    const t3 = setTimeout(() => {
      setPhase('done')
      onComplete()
    }, 4200)
    return () => [t1, t2, t3].forEach(clearTimeout)
  }, [onComplete])

  // Mouse parallax
  const handleMouseMove = useCallback((e: MouseEvent) => {
    const x = (e.clientX / window.innerWidth - 0.5) * 2
    const y = (e.clientY / window.innerHeight - 0.5) * 2
    setMousePos({ x, y })
  }, [])

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [handleMouseMove])

  if (phase === 'done') return null

  const tiltX = mousePos.y * -8
  const tiltY = mousePos.x * 8

  return (
    <div
      className={`splash-screen ${phase}`}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'radial-gradient(ellipse at center, #1a0005 0%, #080808 60%, #000 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'none',
        overflow: 'hidden',
      }}
    >
      {/* Smoke canvas */}
      <canvas
        ref={canvasRef}
        style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
      />

      {/* Red vignette pulse */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(ellipse at center, transparent 30%, rgba(196,30,58,0.25) 100%)',
          animation: phase === 'glow' ? 'vignetteBreath 1.2s ease-in-out infinite alternate' : 'none',
          pointerEvents: 'none',
        }}
      />

      {/* Oni Mask container */}
      <div
        ref={maskRef}
        style={{
          position: 'relative',
          width: 'min(520px, 85vw)',
          transformStyle: 'preserve-3d',
          transform: `
            perspective(900px)
            rotateX(${tiltX}deg)
            rotateY(${tiltY}deg)
          `,
          transition: 'transform 0.1s ease-out',
          animation: phase === 'enter'
            ? 'maskRise 1.2s cubic-bezier(0.16,1,0.3,1) forwards'
            : phase === 'exit'
            ? 'maskExplode 1s cubic-bezier(0.4,0,1,1) forwards'
            : 'maskFloat 3s ease-in-out infinite',
        }}
      >
        {/* Glow behind mask */}
        <div style={{
          position: 'absolute',
          inset: '-20px',
          background: 'radial-gradient(ellipse, rgba(196,30,58,0.4) 0%, transparent 70%)',
          filter: 'blur(30px)',
          opacity: phase === 'glow' || phase === 'exit' ? 1 : 0,
          transition: 'opacity 0.8s ease',
          animation: phase === 'glow' ? 'glowBreath 1.5s ease-in-out infinite alternate' : 'none',
        }} />

        {/* Mask image */}
        <img
          src={oniMask}
          alt="Oni"
          style={{
            width: '100%',
            display: 'block',
            filter: phase === 'glow' || phase === 'exit'
              ? 'drop-shadow(0 0 30px rgba(196,30,58,0.8)) drop-shadow(0 0 60px rgba(196,30,58,0.4)) brightness(1.1)'
              : 'drop-shadow(0 0 10px rgba(196,30,58,0.3)) brightness(0.9)',
            transition: 'filter 0.8s ease',
            transform: 'translateZ(20px)',
          }}
        />

        {/* Left eye glow */}
        <div style={{
          position: 'absolute',
          left: '31%',
          top: '34%',
          width: '12%',
          height: '10%',
          borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(255,200,0,0.9) 0%, rgba(196,30,58,0.8) 40%, transparent 70%)',
          filter: 'blur(4px)',
          opacity: phase === 'glow' || phase === 'exit' ? 1 : 0,
          transition: 'opacity 0.5s ease',
          animation: phase === 'glow' ? 'eyePulse 0.8s ease-in-out infinite alternate' : 'none',
          transform: 'translateZ(30px)',
          mixBlendMode: 'screen',
        }} />

        {/* Right eye glow */}
        <div style={{
          position: 'absolute',
          right: '31%',
          top: '34%',
          width: '12%',
          height: '10%',
          borderRadius: '50%',
          background: 'radial-gradient(ellipse, rgba(255,200,0,0.9) 0%, rgba(196,30,58,0.8) 40%, transparent 70%)',
          filter: 'blur(4px)',
          opacity: phase === 'glow' || phase === 'exit' ? 1 : 0,
          transition: 'opacity 0.5s ease',
          animation: phase === 'glow' ? 'eyePulse 0.8s ease-in-out infinite alternate 0.1s' : 'none',
          transform: 'translateZ(30px)',
          mixBlendMode: 'screen',
        }} />

        {/* Eye flare streaks */}
        {(phase === 'glow' || phase === 'exit') && (
          <>
            <div style={{
              position: 'absolute',
              left: '37%',
              top: '39%',
              width: '2px',
              height: '40px',
              background: 'linear-gradient(to bottom, rgba(255,200,0,0.8), transparent)',
              transform: 'translateZ(35px)',
              animation: 'tearDrop 1.5s ease-in-out infinite',
            }} />
            <div style={{
              position: 'absolute',
              right: '37%',
              top: '39%',
              width: '2px',
              height: '40px',
              background: 'linear-gradient(to bottom, rgba(255,200,0,0.8), transparent)',
              transform: 'translateZ(35px)',
              animation: 'tearDrop 1.5s ease-in-out infinite 0.3s',
            }} />
          </>
        )}
      </div>

      {/* Bottom text */}
      <div style={{
        position: 'absolute',
        bottom: '8%',
        left: 0,
        right: 0,
        textAlign: 'center',
        fontFamily: 'monospace',
        letterSpacing: '0.4em',
        fontSize: '11px',
        color: 'rgba(196,30,58,0.7)',
        textTransform: 'uppercase',
        opacity: phase === 'enter' ? 0 : phase === 'exit' ? 0 : 1,
        transition: 'opacity 0.8s ease',
        animation: phase === 'glow' ? 'textFlicker 3s ease-in-out infinite' : 'none',
      }}>
        HELIOS · AGENTIC AI PLATFORM
      </div>

      {/* Exit flash */}
      {phase === 'exit' && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'white',
          animation: 'exitFlash 0.4s ease-out forwards',
          pointerEvents: 'none',
        }} />
      )}

      <style>{`
        @keyframes maskRise {
          0%   { opacity: 0; transform: perspective(900px) translateY(120px) scale(0.6) rotateX(20deg); }
          60%  { opacity: 1; transform: perspective(900px) translateY(-8px) scale(1.02) rotateX(-2deg); }
          100% { opacity: 1; transform: perspective(900px) translateY(0) scale(1) rotateX(0deg); }
        }
        @keyframes maskFloat {
          0%, 100% { transform: perspective(900px) translateY(0px) rotateX(${tiltX}deg) rotateY(${tiltY}deg); }
          50%       { transform: perspective(900px) translateY(-10px) rotateX(${tiltX}deg) rotateY(${tiltY}deg); }
        }
        @keyframes maskExplode {
          0%   { opacity: 1; transform: perspective(900px) scale(1) translateZ(0); }
          40%  { opacity: 1; transform: perspective(900px) scale(1.15) translateZ(100px); }
          100% { opacity: 0; transform: perspective(900px) scale(2.5) translateZ(400px); }
        }
        @keyframes eyePulse {
          0%   { opacity: 0.7; filter: blur(3px); transform: translateZ(30px) scale(0.9); }
          100% { opacity: 1;   filter: blur(6px); transform: translateZ(30px) scale(1.2); }
        }
        @keyframes glowBreath {
          0%   { opacity: 0.6; transform: scale(0.95); }
          100% { opacity: 1.0; transform: scale(1.05); }
        }
        @keyframes vignetteBreath {
          0%   { opacity: 0.6; }
          100% { opacity: 1.0; }
        }
        @keyframes tearDrop {
          0%, 100% { opacity: 0; height: 20px; }
          50%       { opacity: 1; height: 50px; }
        }
        @keyframes textFlicker {
          0%, 95%, 100% { opacity: 1; }
          96%            { opacity: 0.3; }
          97%            { opacity: 1; }
          98%            { opacity: 0.5; }
        }
        @keyframes exitFlash {
          0%   { opacity: 0.9; }
          100% { opacity: 0; }
        }
      `}</style>
    </div>
  )
}
