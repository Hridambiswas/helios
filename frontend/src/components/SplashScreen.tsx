import { useEffect, useRef, useState, useCallback } from 'react'
import oniMask from '../assets/oni-mask.png'

export function SplashScreen({ onComplete }: { onComplete: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const [phase, setPhase] = useState<'rise' | 'glow' | 'exit' | 'done'>('rise')
  const [mouse, setMouse] = useState({ x: 0, y: 0 })

  /* ── Smoke canvas ── */
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight }
    resize()
    window.addEventListener('resize', resize)

    type P = { x: number; y: number; r: number; op: number; sp: number; dr: number; age: number; max: number }
    const spawn = (): P => ({
      x: Math.random() * canvas.width,
      y: canvas.height + 40,
      r: 30 + Math.random() * 60,
      op: 0.06 + Math.random() * 0.12,
      sp: 0.3 + Math.random() * 0.5,
      dr: (Math.random() - 0.5) * 0.4,
      age: 0,
      max: 300 + Math.random() * 200,
    })

    const ps: P[] = Array.from({ length: 80 }, () => {
      const p = spawn(); p.y = Math.random() * canvas.height; p.age = Math.random() * p.max; return p
    })

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      if (ps.length < 120) ps.push(spawn())
      for (let i = ps.length - 1; i >= 0; i--) {
        const p = ps[i]
        const life = Math.sin((p.age / p.max) * Math.PI)
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r)
        g.addColorStop(0, `rgba(120,10,10,${p.op * life})`)
        g.addColorStop(0.5, `rgba(60,0,0,${p.op * life * 0.4})`)
        g.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = g; ctx.fill()
        p.y -= p.sp; p.x += p.dr + Math.sin(p.age * 0.02) * 0.5
        p.r += 0.15; p.age++
        if (p.age >= p.max || p.y < -100) ps.splice(i, 1)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    tick()
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener('resize', resize) }
  }, [])

  /* ── Phase timing ── */
  useEffect(() => {
    const t1 = setTimeout(() => setPhase('glow'), 1600)
    const t2 = setTimeout(() => setPhase('exit'), 3400)
    const t3 = setTimeout(() => { setPhase('done'); onComplete() }, 4600)
    return () => [t1, t2, t3].forEach(clearTimeout)
  }, [onComplete])

  /* ── Mouse parallax ── */
  const onMove = useCallback((e: MouseEvent) => {
    setMouse({ x: (e.clientX / window.innerWidth - 0.5) * 2, y: (e.clientY / window.innerHeight - 0.5) * 2 })
  }, [])
  useEffect(() => { window.addEventListener('mousemove', onMove); return () => window.removeEventListener('mousemove', onMove) }, [onMove])

  if (phase === 'done') return null

  const tiltX = mouse.y * -10
  const tiltY = mouse.x * 10

  /* ── Animation styles per phase ── */
  const maskAnim = phase === 'rise'
    ? 'maskRise 1.4s cubic-bezier(0.16,1,0.3,1) forwards'
    : phase === 'exit'
    ? 'maskExit 1s cubic-bezier(0.4,0,1,1) forwards'
    : 'maskBreathe 4s ease-in-out infinite'

  const isGlowing = phase === 'glow' || phase === 'exit'

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999, overflow: 'hidden', cursor: 'none',
      background: '#000',
    }}>

      {/* ── Deep background: full-screen mask at very low opacity as wallpaper ── */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: `url(${oniMask})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        opacity: 0.06,
        filter: 'blur(2px) saturate(0.3) brightness(0.5)',
        transform: `scale(1.05) translate(${mouse.x * -6}px, ${mouse.y * -6}px)`,
        transition: 'transform 0.3s ease-out',
      }} />

      {/* ── Atmospheric red glow from center-bottom ── */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 80% 60% at 50% 85%, rgba(140,0,0,0.55) 0%, rgba(80,0,0,0.25) 40%, transparent 70%)',
        animation: isGlowing ? 'atmospherePulse 1.8s ease-in-out infinite alternate' : 'none',
      }} />
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 60% 50% at 50% 100%, rgba(196,30,58,0.3) 0%, transparent 60%)',
      }} />

      {/* ── Smoke ── */}
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />

      {/* ── Vignette ── */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.85) 100%)',
        pointerEvents: 'none',
      }} />

      {/* ── Mask stage ── */}
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        perspective: '1000px',
      }}>

        {/* Cast shadow on "floor" — same shape, blurred below the mask */}
        <div style={{
          position: 'absolute',
          top: '50%', left: '50%',
          transform: 'translate(-50%, 20%) scaleY(0.18) scaleX(0.9)',
          width: 'min(600px, 88vw)',
          opacity: isGlowing ? 0.9 : 0.4,
          filter: 'blur(28px)',
          transition: 'opacity 0.8s ease',
          backgroundImage: `url(${oniMask})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
          aspectRatio: '1.6',
          mixBlendMode: 'normal',
        }}>
          <div style={{ position:'absolute',inset:0, background:'rgba(196,30,58,0.9)', borderRadius:'50%' }} />
        </div>

        {/* The mask itself — 3D container */}
        <div style={{
          position: 'relative',
          width: 'min(600px, 88vw)',
          transformStyle: 'preserve-3d',
          animation: maskAnim,
          transform: phase === 'glow'
            ? `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`
            : undefined,
          transition: phase === 'glow' ? 'transform 0.12s ease-out' : undefined,
        }}>

          {/* Bloom glow layer behind mask */}
          <img src={oniMask} alt="" style={{
            position: 'absolute', inset: '-8%',
            width: '116%',
            opacity: isGlowing ? 0.5 : 0.15,
            filter: 'blur(24px) saturate(2) brightness(0.8)',
            transition: 'opacity 0.8s ease',
            animation: isGlowing ? 'bloomBreath 1.6s ease-in-out infinite alternate' : 'none',
            transform: 'translateZ(-10px)',
          }} />

          {/* Secondary softer bloom */}
          <img src={oniMask} alt="" style={{
            position: 'absolute', inset: '-3%',
            width: '106%',
            opacity: isGlowing ? 0.35 : 0.08,
            filter: 'blur(12px) saturate(3)',
            transition: 'opacity 0.8s ease',
            transform: 'translateZ(-5px)',
          }} />

          {/* ── Main mask image ── */}
          <img
            src={oniMask}
            alt="Oni"
            style={{
              width: '100%',
              display: 'block',
              position: 'relative',
              transform: 'translateZ(0)',
              filter: isGlowing
                ? 'drop-shadow(0 0 40px rgba(196,30,58,1)) drop-shadow(0 0 80px rgba(196,30,58,0.6)) drop-shadow(0 40px 60px rgba(0,0,0,0.95)) brightness(1.15) contrast(1.05)'
                : 'drop-shadow(0 20px 50px rgba(0,0,0,0.95)) drop-shadow(0 0 20px rgba(196,30,58,0.3)) brightness(1.0)',
              transition: 'filter 0.8s ease',
            }}
          />

          {/* Top lighting — simulates a light from above */}
          <div style={{
            position: 'absolute', inset: 0,
            background: 'linear-gradient(to bottom, rgba(255,220,180,0.08) 0%, transparent 40%, rgba(0,0,0,0.3) 100%)',
            pointerEvents: 'none',
            transform: 'translateZ(2px)',
            borderRadius: '4px',
          }} />

          {/* ── Left eye glow ── */}
          <div style={{
            position: 'absolute',
            left: '30%', top: '33%',
            width: '13%', height: '11%',
            borderRadius: '50%',
            background: 'radial-gradient(ellipse, rgba(255,220,60,1) 0%, rgba(255,80,0,0.9) 35%, rgba(196,30,58,0.6) 60%, transparent 80%)',
            filter: isGlowing ? 'blur(5px)' : 'blur(2px)',
            opacity: isGlowing ? 1 : 0,
            transition: 'opacity 0.5s ease, filter 0.5s ease',
            animation: isGlowing ? 'eyeBlaze 0.9s ease-in-out infinite alternate' : 'none',
            transform: 'translateZ(8px)',
            mixBlendMode: 'screen',
          }} />

          {/* ── Right eye glow ── */}
          <div style={{
            position: 'absolute',
            right: '30%', top: '33%',
            width: '13%', height: '11%',
            borderRadius: '50%',
            background: 'radial-gradient(ellipse, rgba(255,220,60,1) 0%, rgba(255,80,0,0.9) 35%, rgba(196,30,58,0.6) 60%, transparent 80%)',
            filter: isGlowing ? 'blur(5px)' : 'blur(2px)',
            opacity: isGlowing ? 1 : 0,
            transition: 'opacity 0.5s ease 0.1s, filter 0.5s ease',
            animation: isGlowing ? 'eyeBlaze 0.9s ease-in-out infinite alternate 0.15s' : 'none',
            transform: 'translateZ(8px)',
            mixBlendMode: 'screen',
          }} />

          {/* Eye light rays */}
          {isGlowing && <>
            <div style={{
              position: 'absolute', left: '36%', top: '44%',
              width: '1px', height: '60px',
              background: 'linear-gradient(to bottom, rgba(255,200,0,0.7), transparent)',
              transform: 'translateZ(10px)', animation: 'rayFade 1.2s ease-in-out infinite',
            }} />
            <div style={{
              position: 'absolute', right: '36%', top: '44%',
              width: '1px', height: '60px',
              background: 'linear-gradient(to bottom, rgba(255,200,0,0.7), transparent)',
              transform: 'translateZ(10px)', animation: 'rayFade 1.2s ease-in-out infinite 0.2s',
            }} />
          </>}
        </div>
      </div>

      {/* Bottom text */}
      <div style={{
        position: 'absolute', bottom: '7%', left: 0, right: 0, textAlign: 'center',
        fontFamily: 'monospace', fontSize: '10px', letterSpacing: '0.5em',
        color: 'rgba(196,30,58,0.6)', textTransform: 'uppercase',
        opacity: phase === 'rise' ? 0 : phase === 'exit' ? 0 : 1,
        transition: 'opacity 1s ease',
        animation: isGlowing ? 'textGlitch 4s ease-in-out infinite' : 'none',
      }}>
        HELIOS · AGENTIC AI PLATFORM
      </div>

      {/* Exit flash */}
      {phase === 'exit' && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(196,30,58,0.15)',
          animation: 'exitFlash 1s ease-out forwards',
          pointerEvents: 'none',
        }} />
      )}

      <style>{`
        @keyframes maskRise {
          0%   { opacity:0; transform:perspective(1000px) translateY(110vh) rotateX(35deg) scale(0.7); }
          50%  { opacity:1; }
          80%  { transform:perspective(1000px) translateY(-12px) rotateX(-3deg) scale(1.02); }
          100% { opacity:1; transform:perspective(1000px) translateY(0) rotateX(0deg) scale(1); }
        }
        @keyframes maskBreathe {
          0%,100% { transform:perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(0); }
          50%      { transform:perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-8px); }
        }
        @keyframes maskExit {
          0%   { opacity:1; transform:perspective(1000px) scale(1) translateZ(0); }
          30%  { opacity:1; transform:perspective(1000px) scale(1.08) translateZ(60px); }
          100% { opacity:0; transform:perspective(1000px) scale(3) translateZ(600px); }
        }
        @keyframes eyeBlaze {
          0%   { opacity:0.75; transform:translateZ(8px) scale(0.9); filter:blur(4px); }
          100% { opacity:1.0;  transform:translateZ(8px) scale(1.3); filter:blur(7px); }
        }
        @keyframes bloomBreath {
          0%   { opacity:0.35; transform:translateZ(-10px) scale(1.0); }
          100% { opacity:0.6;  transform:translateZ(-10px) scale(1.04); }
        }
        @keyframes atmospherePulse {
          0%   { opacity:0.8; }
          100% { opacity:1.0; }
        }
        @keyframes rayFade {
          0%,100% { opacity:0; height:20px; }
          50%      { opacity:0.8; height:70px; }
        }
        @keyframes textGlitch {
          0%,92%,100% { opacity:1; letter-spacing:0.5em; }
          93%  { opacity:0.2; letter-spacing:0.6em; }
          94%  { opacity:1; }
          96%  { opacity:0.5; letter-spacing:0.45em; }
          97%  { opacity:1; }
        }
        @keyframes exitFlash {
          0%   { opacity:1; }
          100% { opacity:0; }
        }
      `}</style>
    </div>
  )
}
