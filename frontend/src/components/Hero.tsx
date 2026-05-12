import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import { api } from '../api/client'

const SUGGESTIONS = [
  'Explain the retrieval pipeline',
  'What is LLM-as-judge evaluation?',
  'How does hybrid search work?',
  'Summarize my uploaded documents',
]

const TAGLINES = [
  'MULTI-AGENT RAG PIPELINE',
  'HYBRID SEMANTIC RETRIEVAL',
  'LLM-AS-JUDGE EVALUATION',
  'SANDBOXED CODE EXECUTION',
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

  // Live stats
  useEffect(() => {
    api.get('/stats').then(({ data }) => setLiveStats(data)).catch(() => {})
  }, [])

  // Typewriter tagline
  useEffect(() => {
    const target = TAGLINES[tagIdx]
    if (charIdx < target.length) {
      const t = setTimeout(() => {
        setDisplayText(target.slice(0, charIdx + 1))
        setCharIdx(c => c + 1)
      }, 55)
      return () => clearTimeout(t)
    }
    const t = setTimeout(() => {
      setCharIdx(0)
      setDisplayText('')
      setTagIdx(i => (i + 1) % TAGLINES.length)
    }, 2600)
    return () => clearTimeout(t)
  }, [charIdx, tagIdx])

  const submit = () => {
    const q = query.trim().slice(0, 500)
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
  }

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-black">

      {/* Subtle grid */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.04]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(196,30,58,1) 1px, transparent 1px), linear-gradient(90deg, rgba(196,30,58,1) 1px, transparent 1px)',
          backgroundSize: '80px 80px',
        }}
      />

      {/* Radial glow behind title */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 70% 50% at 50% 40%, rgba(196,30,58,0.07) 0%, transparent 70%)',
        }}
      />

      {/* Top status badge */}
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1,  y: 0   }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="absolute top-8 left-8 flex items-center gap-3"
      >
        <div className="status-dot" />
        <span className="font-mono text-[11px] text-crimson tracking-widest uppercase">
          Helios v1.1
        </span>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1,  y: 0   }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="absolute top-8 right-8 text-right"
      >
        <div className="font-mono text-[11px] text-white/20 tracking-widest uppercase">
          AGENTIC AI
        </div>
        <div className="font-mono text-[11px] text-white/20 tracking-widest uppercase">
          PLATFORM
        </div>
      </motion.div>

      {/* Main content */}
      <div className="relative z-10 text-center px-6 max-w-4xl w-full">

        {/* Ghost section number */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="font-mono text-[clamp(48px,8vw,96px)] leading-none tracking-tight mb-2 select-none"
          style={{ color: 'transparent', WebkitTextStroke: '1px rgba(196,30,58,0.12)' }}
        >
          01
        </motion.div>

        {/* HELIOS wordmark */}
        <motion.h1
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1,  y: 0  }}
          transition={{ delay: 0.4, duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
          style={{
            fontFamily:    'Impact, "Arial Black", sans-serif',
            fontSize:      'clamp(72px, 16vw, 180px)',
            lineHeight:    1,
            letterSpacing: '-0.02em',
          }}
        >
          <span className="text-white">HEL</span>
          <span
            className="text-crimson"
            style={{ textShadow: '0 0 50px rgba(196,30,58,0.6), 0 0 100px rgba(196,30,58,0.25)' }}
          >
            IOS
          </span>
        </motion.h1>

        {/* Typewriter tagline */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="h-8 mt-4 mb-10"
        >
          <span className="font-mono text-sm tracking-[0.3em] text-white/40 uppercase cursor">
            {displayText}
          </span>
        </motion.div>

        {/* Divider */}
        <motion.div
          initial={{ opacity: 0, scaleX: 0 }}
          animate={{ opacity: 1, scaleX: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
          className="flex items-center gap-4 mb-10 justify-center"
        >
          <div className="h-px flex-1 max-w-[100px] bg-gradient-to-r from-transparent to-crimson/60" />
          <div
            className="w-1.5 h-1.5 bg-crimson rotate-45"
            style={{ boxShadow: '0 0 8px rgba(196,30,58,0.9)' }}
          />
          <div className="h-px flex-1 max-w-[100px] bg-gradient-to-l from-transparent to-crimson/60" />
        </motion.div>

        {/* Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1,  y: 0  }}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="relative max-w-2xl mx-auto"
        >
          <div
            className="flex items-stretch border transition-all duration-300"
            style={{
              borderColor: 'rgba(196,30,58,0.25)',
              background:  'rgba(255,255,255,0.02)',
              boxShadow:   '0 8px 40px rgba(0,0,0,0.6)',
            }}
            onFocus={() => {}}
          >
            <div className="px-4 flex items-center border-r border-crimson/20">
              <span className="font-mono text-crimson text-sm select-none">›</span>
            </div>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="Ask Helios anything…"
              className="flex-1 bg-transparent px-4 py-4 text-white text-sm font-mono placeholder-white/20 outline-none"
            />
            <button
              onClick={submit}
              className="px-7 font-mono text-xs tracking-widest text-white uppercase transition-all duration-200"
              style={{ background: 'rgba(196,30,58,1)' }}
              onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 24px rgba(196,30,58,0.55)')}
              onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}
            >
              RUN →
            </button>
          </div>

          {!isLoggedIn && (
            <p className="text-[11px] text-white/25 mt-2.5 font-mono text-center">
              Guest mode ·{' '}
              <button onClick={onAuthClick} className="text-crimson hover:text-crimson/80 transition-colors">
                sign in
              </button>
              {' '}for history & uploads
            </p>
          )}
        </motion.div>

        {/* Suggestion chips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1 }}
          className="flex flex-wrap gap-2 mt-5 justify-center"
        >
          {SUGGESTIONS.map(s => (
            <button
              key={s}
              onClick={() => { setQuery(s); inputRef.current?.focus() }}
              className="font-mono text-[10px] text-white/25 hover:text-white/55 border border-white/8 hover:border-white/20 px-3 py-1.5 transition-all duration-200"
              style={{ backdropFilter: 'blur(4px)' }}
            >
              {s}
            </button>
          ))}
        </motion.div>

        {/* Live stats */}
        {liveStats && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.3 }}
            className="flex items-center justify-center gap-8 mt-12 font-mono text-[11px] text-white/30 tracking-wider"
          >
            {[
              ['PLANNER',   'LLM DECOMPOSE'],
              ['RETRIEVER', 'HYBRID RAG'],
              ['EXECUTOR',  'SANDBOXED PY'],
              ['CRITIC',    'LLM-AS-JUDGE'],
            ].map(([label, sub]) => (
              <div key={label} className="text-center">
                <div className="text-crimson text-[10px] mb-0.5">{label}</div>
                <div className="text-[10px]">{sub}</div>
              </div>
            ))}
            <div className="w-px h-6 bg-white/10 hidden sm:block" />
            <div className="text-center">
              <div className="text-crimson text-[10px] mb-0.5">{liveStats.total_queries.toLocaleString()}</div>
              <div className="text-[10px]">QUERIES</div>
            </div>
            <div className="text-center">
              <div className="text-crimson text-[10px] mb-0.5">{liveStats.total_documents.toLocaleString()}</div>
              <div className="text-[10px]">DOCS</div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-white/20"
      >
        <span className="font-mono text-[10px] tracking-widest uppercase">The Pipeline</span>
        <ChevronDown size={14} className="animate-bounce" />
      </motion.div>
    </section>
  )
}
