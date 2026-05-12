import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import { api } from '../api/client'
import { DragonScales, ClawMark, DragonEye } from './DragonDecor'

const SUGGESTIONS = [
  'Explain the retrieval pipeline',
  'What is LLM-as-judge evaluation?',
  'How does hybrid search work?',
  'Analyse my uploaded documents',
]

const TAGLINES = [
  'FORGE YOUR QUESTIONS IN DRAGON FIRE',
  'HYBRID SEMANTIC RETRIEVAL',
  'LLM-AS-JUDGE EVALUATION',
  'FIVE AGENTS. ONE ANSWER.',
]

export function Hero({
  onQuerySubmit,
  onAuthClick,
  isLoggedIn,
}: {
  onQuerySubmit: (q: string) => void
  onAuthClick: () => void
  isLoggedIn: boolean
}) {
  const [query,       setQuery]       = useState('')
  const [tagIdx,      setTagIdx]      = useState(0)
  const [displayText, setDisplayText] = useState('')
  const [charIdx,     setCharIdx]     = useState(0)
  const [liveStats,   setLiveStats]   = useState<{ total_queries: number; total_documents: number } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.get('/stats').then(({ data }) => setLiveStats(data)).catch(() => {})
  }, [])

  useEffect(() => {
    const target = TAGLINES[tagIdx]
    if (charIdx < target.length) {
      const t = setTimeout(() => { setDisplayText(target.slice(0, charIdx + 1)); setCharIdx(c => c + 1) }, 48)
      return () => clearTimeout(t)
    }
    const t = setTimeout(() => { setCharIdx(0); setDisplayText(''); setTagIdx(i => (i + 1) % TAGLINES.length) }, 2600)
    return () => clearTimeout(t)
  }, [charIdx, tagIdx])

  const submit = () => {
    const q = query.trim().slice(0, 500)
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
  }

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden"
      style={{ background: 'radial-gradient(ellipse 80% 70% at 50% 40%, #0D0800 0%, #050505 60%, #000 100%)' }}>

      {/* Dragon scale texture overlay */}
      <DragonScales opacity={0.09} />

      {/* Animated gold grid (subtle) */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.025]"
        style={{
          backgroundImage: 'linear-gradient(rgba(201,162,39,1) 1px, transparent 1px), linear-gradient(90deg, rgba(201,162,39,1) 1px, transparent 1px)',
          backgroundSize: '80px 80px',
        }}
      />

      {/* Radial gold glow behind text */}
      <div className="absolute inset-0 pointer-events-none"
        style={{ background: 'radial-gradient(ellipse 65% 45% at 50% 42%, rgba(201,162,39,0.07) 0%, transparent 70%)' }} />

      {/* Corner claw marks */}
      <ClawMark corner="tl" size={140} opacity={0.10} />
      <ClawMark corner="tr" size={140} opacity={0.10} />
      <ClawMark corner="bl" size={100} opacity={0.06} />
      <ClawMark corner="br" size={100} opacity={0.06} />

      {/* Top status */}
      <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="absolute top-8 left-8 flex items-center gap-3">
        <div className="status-dot" />
        <span className="font-mono text-[11px] tracking-widest uppercase" style={{ color: '#C9A227' }}>
          Helios v1.1
        </span>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="absolute top-8 right-8 text-right">
        <div className="font-mono text-[11px] tracking-widest uppercase" style={{ color: 'rgba(201,162,39,0.3)' }}>AGENTIC AI</div>
        <div className="font-mono text-[11px] tracking-widest uppercase" style={{ color: 'rgba(201,162,39,0.3)' }}>PLATFORM</div>
      </motion.div>

      {/* Main content */}
      <div className="relative z-10 text-center px-6 max-w-4xl w-full">

        {/* Ghost number */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3, duration: 0.8 }}
          className="font-mono leading-none mb-2 select-none"
          style={{ fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 'clamp(48px,8vw,96px)', color: 'transparent', WebkitTextStroke: '1px rgba(201,162,39,0.1)' }}>
          01
        </motion.div>

        {/* Dragon eye ornament */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          className="flex items-center justify-center gap-4 mb-4">
          <div className="h-px flex-1 max-w-[80px]" style={{ background: 'linear-gradient(90deg, transparent, rgba(201,162,39,0.5))' }} />
          <DragonEye size={28} opacity={0.7} />
          <div className="h-px flex-1 max-w-[80px]" style={{ background: 'linear-gradient(90deg, rgba(201,162,39,0.5), transparent)' }} />
        </motion.div>

        {/* HELIOS wordmark */}
        <motion.h1
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1,  y: 0  }}
          transition={{ delay: 0.4, duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
          className="leading-none animate-flicker"
          style={{
            fontFamily: 'Impact, "Arial Black", sans-serif',
            fontSize: 'clamp(72px, 16vw, 185px)',
            letterSpacing: '-0.02em',
          }}
        >
          <span className="text-white">HEL</span>
          <span style={{
            color: '#C9A227',
            textShadow: '0 0 50px rgba(201,162,39,0.7), 0 0 100px rgba(201,162,39,0.3), 0 0 160px rgba(255,107,0,0.15)',
          }}>IOS</span>
        </motion.h1>

        {/* Typewriter tagline */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }}
          className="h-8 mt-4 mb-3">
          <span className="font-mono text-sm tracking-[0.25em] uppercase cursor"
            style={{ color: 'rgba(201,162,39,0.55)' }}>
            {displayText}
          </span>
        </motion.div>

        {/* Gold divider */}
        <motion.div
          initial={{ opacity: 0, scaleX: 0 }} animate={{ opacity: 1, scaleX: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="flex items-center gap-4 mb-8 justify-center">
          <div className="h-px flex-1 max-w-[100px]" style={{ background: 'linear-gradient(90deg, transparent, #C9A227)' }} />
          <div className="w-2 h-2 rotate-45" style={{ background: '#C9A227', boxShadow: '0 0 10px rgba(201,162,39,0.9)' }} />
          <div className="h-px flex-1 max-w-[100px]" style={{ background: 'linear-gradient(90deg, #C9A227, transparent)' }} />
        </motion.div>

        {/* Prompt input */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="relative max-w-2xl mx-auto">

          {/* Fire glow on container */}
          <div className="relative">
            <div className="flex items-stretch transition-all duration-300 group"
              style={{
                border: '1px solid rgba(201,162,39,0.3)',
                background: 'rgba(201,162,39,0.03)',
                boxShadow: '0 8px 40px rgba(0,0,0,0.7)',
              }}
              onFocus={() => {}} >

              {/* Dragon fang prefix */}
              <div className="px-4 flex items-center"
                style={{ borderRight: '1px solid rgba(201,162,39,0.18)' }}>
                <span className="font-mono text-sm select-none" style={{ color: '#C9A227' }}>⟡</span>
              </div>

              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && submit()}
                onFocus={e => {
                  e.currentTarget.parentElement!.style.borderColor = 'rgba(201,162,39,0.7)'
                  e.currentTarget.parentElement!.style.boxShadow   = '0 0 30px rgba(201,162,39,0.2), 0 8px 40px rgba(0,0,0,0.7)'
                }}
                onBlur={e => {
                  e.currentTarget.parentElement!.style.borderColor = 'rgba(201,162,39,0.3)'
                  e.currentTarget.parentElement!.style.boxShadow   = '0 8px 40px rgba(0,0,0,0.7)'
                }}
                placeholder="Breathe fire into your query…"
                className="flex-1 bg-transparent px-4 py-4 text-white text-sm font-mono outline-none"
                style={{ caretColor: '#C9A227' }}
              />

              <button onClick={submit}
                className="px-7 font-mono text-xs tracking-widest text-black uppercase font-bold transition-all duration-200"
                style={{ background: 'linear-gradient(135deg, #C9A227, #FFD700)' }}
                onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 24px rgba(201,162,39,0.6)')}
                onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}>
                IGNITE
              </button>
            </div>
          </div>

          {!isLoggedIn && (
            <p className="text-[11px] mt-2.5 font-mono text-center"
              style={{ color: 'rgba(201,162,39,0.25)' }}>
              Dragon guest mode ·{' '}
              <button onClick={onAuthClick}
                className="transition-colors hover:opacity-80"
                style={{ color: '#C9A227' }}>
                sign in
              </button>
              {' '}for full power
            </p>
          )}
        </motion.div>

        {/* Suggestion chips */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.1 }}
          className="flex flex-wrap gap-2 mt-5 justify-center">
          {SUGGESTIONS.map(s => (
            <button key={s}
              onClick={() => { setQuery(s); inputRef.current?.focus() }}
              className="font-mono text-[10px] px-3 py-1.5 transition-all duration-200"
              style={{
                color: 'rgba(201,162,39,0.35)',
                border: '1px solid rgba(201,162,39,0.12)',
                backdropFilter: 'blur(4px)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.color       = 'rgba(201,162,39,0.75)'
                e.currentTarget.style.borderColor = 'rgba(201,162,39,0.35)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.color       = 'rgba(201,162,39,0.35)'
                e.currentTarget.style.borderColor = 'rgba(201,162,39,0.12)'
              }}>
              {s}
            </button>
          ))}
        </motion.div>

        {/* Live stats */}
        {liveStats && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.3 }}
            className="flex flex-wrap items-center justify-center gap-6 sm:gap-8 mt-12 font-mono text-[11px] tracking-wider"
            style={{ color: 'rgba(201,162,39,0.3)' }}>
            {[['PLANNER','LLM DECOMPOSE'],['RETRIEVER','HYBRID RAG'],['EXECUTOR','SANDBOXED PY'],['CRITIC','LLM-AS-JUDGE']].map(([label,sub]) => (
              <div key={label} className="text-center">
                <div className="text-[10px] mb-0.5" style={{ color: '#C9A227' }}>{label}</div>
                <div className="text-[10px]">{sub}</div>
              </div>
            ))}
            <div className="w-px h-6 bg-white/10 hidden sm:block" />
            <div className="text-center">
              <div className="text-[10px] mb-0.5" style={{ color: '#FF6B00' }}>{liveStats.total_queries.toLocaleString()}</div>
              <div className="text-[10px]">QUERIES</div>
            </div>
            <div className="text-center">
              <div className="text-[10px] mb-0.5" style={{ color: '#FF6B00' }}>{liveStats.total_documents.toLocaleString()}</div>
              <div className="text-[10px]">DOCS INDEXED</div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Scroll indicator */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.6 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        style={{ color: 'rgba(201,162,39,0.25)' }}>
        <span className="font-mono text-[10px] tracking-widest uppercase">Behold The Pipeline</span>
        <ChevronDown size={14} className="animate-bounce" />
      </motion.div>

      {/* Bottom gradient blend into next section */}
      <div className="absolute bottom-0 left-0 right-0 h-28"
        style={{ background: 'linear-gradient(to bottom, transparent, #050505)' }} />
    </section>
  )
}
