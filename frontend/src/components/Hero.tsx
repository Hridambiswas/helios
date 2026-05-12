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
  const contentOpacity = useTransform(scrollYProgress, [0, 0.4], [1, 0])
  const contentY       = useTransform(scrollYProgress, [0, 0.4], ['0%', '-12%'])

  const submit = () => {
    const q = query.trim().slice(0, 500)
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
  }

  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen bg-black overflow-hidden flex flex-col justify-center"
    >
      {/* 3D shattered crystal — fills hero, centered */}
      <div className="absolute inset-0 pointer-events-none z-10">
        <HeroScene />
      </div>

      {/* TOP RIGHT label */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="absolute top-20 right-6 text-right pointer-events-none z-20"
      >
        <div className="font-mono text-[10px] tracking-widest uppercase" style={{ color: 'rgba(255,255,255,0.25)' }}>BUILT FOR</div>
        <div className="font-mono text-[10px] tracking-widest uppercase" style={{ color: 'rgba(255,255,255,0.25)' }}>THE FUTURE</div>
      </motion.div>

      {/* MAIN HERO TEXT — split layout like the reference */}
      <motion.div
        style={{ opacity: contentOpacity, y: contentY }}
        className="relative z-20 w-full select-none pointer-events-none px-0"
      >
        {/* Row 1: "HEL" — left aligned, massive */}
        <motion.div
          initial={{ opacity: 0, x: -60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="leading-none overflow-hidden"
          style={{ paddingLeft: '3vw' }}
        >
          <span
            style={{
              fontFamily: '"Impact", "Arial Black", sans-serif',
              fontSize:   'clamp(96px, 22vw, 300px)',
              letterSpacing: '-0.03em',
              color: '#ffffff',
              lineHeight: 0.88,
              display: 'block',
            }}
          >
            HEL
          </span>
        </motion.div>

        {/* Row 2: "IOS" — right aligned, massive */}
        <motion.div
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="leading-none overflow-hidden text-right"
          style={{ paddingRight: '3vw' }}
        >
          <span
            style={{
              fontFamily: '"Impact", "Arial Black", sans-serif',
              fontSize:   'clamp(96px, 22vw, 300px)',
              letterSpacing: '-0.03em',
              color: '#ffffff',
              lineHeight: 0.88,
              display: 'block',
            }}
          >
            IOS
          </span>
        </motion.div>
      </motion.div>

      {/* BOTTOM LEFT — WE TRANSCEND + query input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9, duration: 0.7 }}
        className="absolute bottom-10 left-6 z-20 pointer-events-auto"
        style={{ maxWidth: 420 }}
      >
        <p className="font-mono text-[11px] tracking-[0.3em] uppercase mb-0.5" style={{ color: 'rgba(255,255,255,0.45)' }}>
          WE TRANSCEND
        </p>
        <p className="font-mono text-[11px] tracking-[0.3em] uppercase mb-5" style={{ color: 'rgba(255,255,255,0.45)' }}>
          INTELLIGENCE
        </p>

        {/* Query input */}
        <div
          className="flex items-stretch transition-all duration-300"
          style={{
            border: '1px solid rgba(255,255,255,0.15)',
            background: 'rgba(0,0,0,0.7)',
            backdropFilter: 'blur(12px)',
          }}
          onFocusCapture={e => {
            (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(139,92,246,0.7)'
            ;(e.currentTarget as HTMLDivElement).style.boxShadow  = '0 0 24px rgba(139,92,246,0.2)'
          }}
          onBlurCapture={e => {
            (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.15)'
            ;(e.currentTarget as HTMLDivElement).style.boxShadow  = 'none'
          }}
        >
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="Query the intelligence..."
            className="flex-1 bg-transparent px-4 py-3 text-white text-sm font-mono outline-none placeholder:text-white/20"
            style={{ caretColor: '#8b5cf6' }}
          />
          <button
            onClick={submit}
            className="px-5 font-mono text-[10px] tracking-widest uppercase text-white/70 hover:text-white transition-colors border-l"
            style={{ borderColor: 'rgba(255,255,255,0.1)' }}
          >
            RUN
          </button>
        </div>

        {!isLoggedIn && (
          <p className="text-[10px] mt-2 font-mono" style={{ color: 'rgba(255,255,255,0.2)' }}>
            Guest mode ·{' '}
            <button onClick={onAuthClick} className="hover:opacity-70 transition-opacity" style={{ color: '#a78bfa' }}>
              sign in
            </button>
            {' '}for full access
          </p>
        )}
      </motion.div>

      {/* BOTTOM RIGHT — small description */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.1 }}
        className="absolute bottom-10 right-6 z-20 text-right pointer-events-none"
        style={{ maxWidth: 220 }}
      >
        <p className="font-mono text-[10px] leading-relaxed" style={{ color: 'rgba(255,255,255,0.2)' }}>
          Distributed multi-agent AI platform. Five agents. One relentless pipeline. No query escapes.
        </p>
      </motion.div>

      {/* Scroll line indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-20 pointer-events-none"
      >
        <motion.div
          className="w-px"
          style={{
            height: 48,
            background: 'linear-gradient(to bottom, rgba(255,255,255,0.4), transparent)',
          }}
          animate={{ scaleY: [1, 0.4, 1], opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
        />
      </motion.div>
    </section>
  )
}
