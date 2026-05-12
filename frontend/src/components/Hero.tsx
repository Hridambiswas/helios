import { useState, useRef } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { HeroScene } from '../three/HeroScene'

export function Hero({
  onQuerySubmit,
  onAuthClick,
  isLoggedIn,
}: {
  onQuerySubmit: (q: string) => void
  onAuthClick: () => void
  isLoggedIn: boolean
}) {
  const [query, setQuery] = useState('')
  const sectionRef = useRef<HTMLElement>(null)
  const { scrollYProgress } = useScroll({ target: sectionRef, offset: ['start start', 'end start'] })
  const textY  = useTransform(scrollYProgress, [0, 1], ['0%',   '-30%'])
  const sceneY = useTransform(scrollYProgress, [0, 1], ['0%',   '-15%'])
  const opacity = useTransform(scrollYProgress, [0, 0.6], [1, 0])

  const submit = () => {
    const q = query.trim().slice(0, 500)
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
  }

  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-black"
    >
      {/* Radial violet backdrop */}
      <div className="absolute inset-0 pointer-events-none hero-bg" />

      {/* 3D Orb — fills the hero, pointer-events off so text/input works */}
      <motion.div
        style={{ y: sceneY }}
        className="absolute inset-0 pointer-events-none"
      >
        <HeroScene />
      </motion.div>

      {/* Content */}
      <motion.div
        style={{ y: textY, opacity }}
        className="relative z-10 w-full flex flex-col items-center text-center select-none px-4"
      >
        {/* Ghost number */}
        <div
          className="font-display leading-none mb-1 pointer-events-none"
          style={{
            fontSize: 'clamp(40px, 7vw, 100px)',
            color: 'transparent',
            WebkitTextStroke: '1px rgba(139,92,246,0.1)',
          }}
        >
          01
        </div>

        {/* Main wordmark — HEL | (orb) | IOS split */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="flex items-center justify-center w-full"
        >
          <h1
            className="leading-none tracking-tighter pointer-events-none"
            style={{
              fontFamily: '"Impact", "Arial Black", sans-serif',
              fontSize: 'clamp(72px, 18vw, 220px)',
              letterSpacing: '-0.03em',
              color: '#fff',
            }}
          >
            <span>HEL</span>
            <span
              className="glow-violet"
              style={{
                color: '#8b5cf6',
                textShadow: '0 0 60px rgba(139,92,246,0.9), 0 0 120px rgba(139,92,246,0.5), 0 0 200px rgba(139,92,246,0.2)',
              }}
            >
              IOS
            </span>
          </h1>
        </motion.div>

        {/* Tagline */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.7 }}
          className="font-mono text-xs sm:text-sm tracking-[0.35em] uppercase mt-4 mb-2 pointer-events-none"
          style={{ color: 'rgba(255,255,255,0.3)' }}
        >
          Distributed Multi-Agent AI Platform
        </motion.p>

        <motion.div
          initial={{ opacity: 0, scaleX: 0 }}
          animate={{ opacity: 1, scaleX: 1 }}
          transition={{ delay: 0.85, duration: 0.6 }}
          className="h-px w-40 mb-10"
          style={{ background: 'linear-gradient(90deg, transparent, #8b5cf6, transparent)' }}
        />

        {/* Query input */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.7 }}
          className="w-full max-w-xl pointer-events-auto"
        >
          <div
            className="flex items-stretch transition-all duration-300"
            style={{
              border: '1px solid rgba(139,92,246,0.3)',
              background: 'rgba(0,0,0,0.6)',
              backdropFilter: 'blur(12px)',
            }}
            onFocusCapture={e => {
              (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(139,92,246,0.8)'
              ;(e.currentTarget as HTMLDivElement).style.boxShadow  = '0 0 30px rgba(139,92,246,0.2)'
            }}
            onBlurCapture={e => {
              (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(139,92,246,0.3)'
              ;(e.currentTarget as HTMLDivElement).style.boxShadow  = 'none'
            }}
          >
            <div className="px-4 flex items-center" style={{ borderRight: '1px solid rgba(139,92,246,0.2)' }}>
              <span className="font-mono text-sm" style={{ color: '#8b5cf6' }}>⟡</span>
            </div>
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="Query the intelligence..."
              className="flex-1 bg-transparent px-4 py-4 text-white text-sm font-mono outline-none"
              style={{ caretColor: '#8b5cf6' }}
            />
            <button
              onClick={submit}
              className="px-7 font-mono text-xs tracking-widest uppercase font-bold text-white transition-all duration-200"
              style={{ background: 'linear-gradient(135deg, #7c3aed, #8b5cf6)' }}
              onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 28px rgba(139,92,246,0.7)')}
              onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}
            >
              RUN
            </button>
          </div>

          {!isLoggedIn && (
            <p className="text-[11px] mt-2 font-mono text-center" style={{ color: 'rgba(255,255,255,0.2)' }}>
              Guest mode ·{' '}
              <button
                onClick={onAuthClick}
                className="transition-colors hover:opacity-80"
                style={{ color: '#8b5cf6' }}
              >
                sign in
              </button>
              {' '}for full access
            </p>
          )}
        </motion.div>
      </motion.div>

      {/* Built-for tags — top right like the video */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute top-20 right-6 text-right hidden sm:block pointer-events-none"
      >
        <div className="font-mono text-[10px] tracking-widest uppercase" style={{ color: 'rgba(255,255,255,0.18)' }}>
          BUILT FOR
        </div>
        <div className="font-mono text-[10px] tracking-widest uppercase" style={{ color: 'rgba(255,255,255,0.18)' }}>
          INTELLIGENCE
        </div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 pointer-events-none"
      >
        <span className="font-mono text-[9px] tracking-[0.5em] uppercase" style={{ color: 'rgba(139,92,246,0.35)' }}>
          SCROLL
        </span>
        <div
          className="w-px h-14"
          style={{
            background: 'linear-gradient(to bottom, rgba(139,92,246,0.6), transparent)',
            animation: 'float 2s ease-in-out infinite',
          }}
        />
      </motion.div>
    </section>
  )
}
