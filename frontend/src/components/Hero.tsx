import { useState, useRef } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { HeroScene } from '../three/HeroScene'
import { ArrowRight } from 'lucide-react'

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
  const contentY       = useTransform(scrollYProgress, [0, 0.4], ['0%', '-8%'])
  const bgTextOpacity  = useTransform(scrollYProgress, [0, 0.3], [0.06, 0])

  const submit = () => {
    const q = query.trim().slice(0, 500)
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
  }

  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen bg-black overflow-hidden flex flex-col items-center justify-center"
    >
      {/* 3D crystal — fills hero */}
      <div className="absolute inset-0 pointer-events-none z-10">
        <HeroScene />
      </div>

      {/* Decorative HEL / IOS — huge background letterform */}
      <motion.div
        style={{ opacity: bgTextOpacity }}
        className="absolute inset-0 z-10 flex flex-col justify-center pointer-events-none select-none overflow-hidden"
      >
        <div
          style={{
            fontFamily: '"Orbitron", sans-serif',
            fontSize: 'clamp(120px, 28vw, 380px)',
            letterSpacing: '-0.03em',
            color: '#ffffff',
            lineHeight: 0.88,
            paddingLeft: '2vw',
          }}
        >
          HEL
        </div>
        <div
          style={{
            fontFamily: '"Orbitron", sans-serif',
            fontSize: 'clamp(120px, 28vw, 380px)',
            letterSpacing: '-0.03em',
            color: '#ffffff',
            lineHeight: 0.88,
            textAlign: 'right',
            paddingRight: '2vw',
          }}
        >
          IOS
        </div>
      </motion.div>

      {/* TOP RIGHT label */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="absolute top-20 right-6 text-right pointer-events-none z-30"
      >
        <div className="font-mono text-[9px] tracking-[0.3em] uppercase" style={{ color: 'rgba(255,255,255,0.2)' }}>BUILT FOR</div>
        <div className="font-mono text-[9px] tracking-[0.3em] uppercase" style={{ color: 'rgba(255,255,255,0.2)' }}>THE FUTURE</div>
      </motion.div>

      {/* TOP LEFT — badge */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="absolute top-20 left-6 pointer-events-none z-30"
      >
        <div
          className="inline-flex items-center gap-2 px-3 py-1 font-mono text-[9px] tracking-[0.3em] uppercase border"
          style={{ borderColor: 'rgba(139,92,246,0.3)', color: 'rgba(139,92,246,0.7)', background: 'rgba(139,92,246,0.05)' }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-[#8b5cf6] animate-pulse" />
          MULTI-AGENT AI
        </div>
      </motion.div>

      {/* CENTERED CONTENT — main hero body */}
      <motion.div
        style={{ opacity: contentOpacity, y: contentY }}
        className="relative z-30 w-full flex flex-col items-center text-center px-4"
      >
        {/* Eyebrow */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.7 }}
          className="mb-6"
        >
          <p className="font-mono text-[10px] tracking-[0.5em] uppercase mb-1" style={{ color: 'rgba(255,255,255,0.3)' }}>
            WE TRANSCEND
          </p>
          <p className="font-mono text-[10px] tracking-[0.5em] uppercase" style={{ color: 'rgba(139,92,246,0.5)' }}>
            INTELLIGENCE
          </p>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="mb-10 select-none"
          style={{
            fontFamily: '"Orbitron", sans-serif',
            fontSize: 'clamp(52px, 10vw, 120px)',
            letterSpacing: '-0.02em',
            lineHeight: 1,
            color: '#ffffff',
          }}
        >
          HEL<span style={{ color: '#a78bfa' }}>IOS</span>
        </motion.h1>

        {/* Query input — center, full width up to 640px */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.7 }}
          className="w-full pointer-events-auto"
          style={{ maxWidth: 620 }}
        >
          <div
            className="group flex items-stretch transition-all duration-300"
            style={{
              border: '1px solid rgba(255,255,255,0.12)',
              background: 'rgba(0,0,0,0.75)',
              backdropFilter: 'blur(16px)',
              boxShadow: '0 0 60px rgba(139,92,246,0.08)',
            }}
            onFocusCapture={e => {
              ;(e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(139,92,246,0.6)'
              ;(e.currentTarget as HTMLDivElement).style.boxShadow  = '0 0 40px rgba(139,92,246,0.18), 0 0 0 1px rgba(139,92,246,0.15)'
            }}
            onBlurCapture={e => {
              ;(e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.12)'
              ;(e.currentTarget as HTMLDivElement).style.boxShadow  = '0 0 60px rgba(139,92,246,0.08)'
            }}
          >
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="Ask anything — documents, research, analysis…"
              className="flex-1 bg-transparent px-5 py-4 text-white text-sm font-sans outline-none placeholder:text-white/20"
              style={{ caretColor: '#8b5cf6' }}
              autoFocus={false}
            />
            <button
              onClick={submit}
              className="flex items-center gap-2 px-5 font-mono text-[10px] tracking-widest uppercase transition-all border-l"
              style={{
                borderColor: 'rgba(255,255,255,0.08)',
                color: query.trim() ? '#a78bfa' : 'rgba(255,255,255,0.3)',
              }}
            >
              <span className="hidden sm:inline">RUN</span>
              <ArrowRight size={13} />
            </button>
          </div>

          {/* Sub-caption */}
          <div className="mt-3 flex items-center justify-center gap-4">
            {!isLoggedIn ? (
              <p className="font-mono text-[9px]" style={{ color: 'rgba(255,255,255,0.2)' }}>
                Guest mode ·{' '}
                <button onClick={onAuthClick} className="hover:opacity-80 transition-opacity" style={{ color: '#a78bfa' }}>
                  sign in
                </button>
                {' '}for full access
              </p>
            ) : (
              <p className="font-mono text-[9px]" style={{ color: 'rgba(255,255,255,0.18)' }}>
                Five agents · One pipeline · No query escapes
              </p>
            )}
          </div>
        </motion.div>

        {/* Quick suggestion chips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1, duration: 0.6 }}
          className="mt-6 flex flex-wrap justify-center gap-2 pointer-events-auto"
        >
          {['Summarize my documents', 'Explain the pipeline', 'Find key insights'].map(s => (
            <button
              key={s}
              onClick={() => { setQuery(s); submit() }}
              className="font-mono text-[9px] tracking-wider px-3 py-1.5 border transition-all"
              style={{
                borderColor: 'rgba(255,255,255,0.07)',
                color: 'rgba(255,255,255,0.3)',
                background: 'rgba(255,255,255,0.02)',
              }}
              onMouseEnter={e => {
                ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.35)'
                ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(167,139,250,0.8)'
              }}
              onMouseLeave={e => {
                ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.07)'
                ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.3)'
              }}
            >
              {s}
            </button>
          ))}
        </motion.div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-30 pointer-events-none"
      >
        <motion.div
          className="w-px"
          style={{
            height: 44,
            background: 'linear-gradient(to bottom, rgba(139,92,246,0.5), transparent)',
          }}
          animate={{ scaleY: [1, 0.4, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
        />
      </motion.div>
    </section>
  )
}
