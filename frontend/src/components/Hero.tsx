import { useState, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'
import { api } from '../api/client'

const TAGLINES = [
  'MULTI-AGENT RAG PIPELINE',
  'HYBRID SEMANTIC RETRIEVAL',
  'LLM-AS-JUDGE EVALUATION',
  'DISTRIBUTED INTELLIGENCE',
]

export function Hero({ onQuerySubmit, onAuthClick, isLoggedIn }: {
  onQuerySubmit: (q: string) => void
  onAuthClick: () => void
  isLoggedIn: boolean
}) {
  const [query, setQuery] = useState('')
  const [tagline, setTagline] = useState(0)
  const [displayText, setDisplayText] = useState('')
  const [charIdx, setCharIdx] = useState(0)
  const [liveStats, setLiveStats] = useState<{ total_queries: number; total_documents: number } | null>(null)

  useEffect(() => {
    api.get('/stats').then(({ data }) => setLiveStats(data)).catch(() => {})
  }, [])

  useEffect(() => {
    const target = TAGLINES[tagline]
    if (charIdx < target.length) {
      const t = setTimeout(() => {
        setDisplayText(target.slice(0, charIdx + 1))
        setCharIdx(c => c + 1)
      }, 60)
      return () => clearTimeout(t)
    } else {
      const t = setTimeout(() => {
        setCharIdx(0)
        setDisplayText('')
        setTagline(i => (i + 1) % TAGLINES.length)
      }, 2400)
      return () => clearTimeout(t)
    }
  }, [charIdx, tagline])

  const submit = () => {
    const q = query.trim()
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
    document.getElementById('query-section')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      {/* Radial gradient background */}
      <div className="absolute inset-0 bg-gradient-radial from-[#1a0008] via-ink to-ink" />

      {/* Grid lines */}
      <div
        className="absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* Top-left version tag */}
      <div className="absolute top-8 left-8 flex items-center gap-3">
        <div className="status-dot" />
        <span className="font-mono text-xs text-crimson tracking-widest uppercase">Helios v1.0</span>
      </div>

      {/* Top-right label */}
      <div className="absolute top-8 right-8 text-right">
        <div className="text-xs text-[#555] tracking-widest uppercase font-mono">AGENTIC AI</div>
        <div className="text-xs text-[#555] tracking-widest uppercase font-mono">PLATFORM</div>
      </div>

      {/* Main content */}
      <div className="relative z-10 text-center px-4 max-w-5xl w-full">
        {/* Large number accent */}
        <div className="section-number mb-2 select-none" style={{ color: 'transparent', WebkitTextStroke: '1px rgba(196,30,58,0.2)' }}>
          01
        </div>

        {/* Title */}
        <h1 className="font-display text-[clamp(64px,14vw,180px)] leading-none tracking-tight mb-4 animate-flicker"
          style={{ fontFamily: 'Impact, Arial Black, sans-serif' }}>
          <span className="text-white">HEL</span>
          <span className="text-crimson" style={{ textShadow: '0 0 40px rgba(196,30,58,0.5)' }}>IOS</span>
        </h1>

        {/* Tagline typewriter */}
        <div className="h-8 mb-8">
          <span className="font-mono text-sm tracking-[0.3em] text-[#888] uppercase cursor">
            {displayText}
          </span>
        </div>

        {/* Divider */}
        <div className="flex items-center gap-4 mb-10 justify-center">
          <div className="h-px flex-1 max-w-[120px] bg-gradient-to-r from-transparent to-crimson" />
          <div className="w-1.5 h-1.5 bg-crimson rotate-45" />
          <div className="h-px flex-1 max-w-[120px] bg-gradient-to-l from-transparent to-crimson" />
        </div>

        {/* Query input */}
        <div className="relative max-w-2xl mx-auto">
          <div className="card-dark rounded-none border border-crimson/30 flex items-stretch group transition-all duration-300 hover:border-crimson/60 hover:glow-red">
            <div className="px-4 flex items-center border-r border-crimson/20">
              <span className="font-mono text-crimson text-xs">{'>'}</span>
            </div>
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="Ask anything — HELIOS will plan, retrieve, and synthesize..."
              className="flex-1 bg-transparent px-4 py-4 text-white text-sm font-mono placeholder-[#444] outline-none"
            />
            <button
              onClick={submit}
              className="px-6 bg-crimson hover:bg-crimson-light transition-colors text-white font-mono text-xs tracking-widest uppercase border-l border-crimson/50"
            >
              RUN
            </button>
          </div>
          {!isLoggedIn && (
            <p className="text-xs text-[#555] mt-2 font-mono text-center">
              1 free query — {' '}
              <button onClick={onAuthClick} className="text-crimson hover:text-crimson-light transition-colors">
                sign in
              </button>
              {' '}to unlock unlimited
            </p>
          )}
        </div>

        {/* Stats row */}
        <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-8 mt-12 text-xs font-mono text-[#555] tracking-wider">
          {[
            ['PLANNER', 'LLM DECOMPOSE'],
            ['RETRIEVER', 'HYBRID RAG'],
            ['EXECUTOR', 'SANDBOXED PY'],
            ['CRITIC', 'LLM-AS-JUDGE'],
          ].map(([label, sub]) => (
            <div key={label} className="text-center">
              <div className="text-crimson text-[10px]">{label}</div>
              <div className="text-[10px] mt-0.5">{sub}</div>
            </div>
          ))}
          {liveStats && (
            <>
              <div className="w-px h-6 bg-white/10 hidden sm:block" />
              <div className="text-center">
                <div className="text-crimson text-[10px]">{liveStats.total_queries.toLocaleString()}</div>
                <div className="text-[10px] mt-0.5">QUERIES RUN</div>
              </div>
              <div className="text-center">
                <div className="text-crimson text-[10px]">{liveStats.total_documents.toLocaleString()}</div>
                <div className="text-[10px] mt-0.5">DOCS INDEXED</div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-[#444]">
        <span className="font-mono text-[10px] tracking-widest uppercase">SCROLL</span>
        <ChevronDown size={16} className="animate-bounce" />
      </div>

      {/* Bottom ink stroke */}
      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-ink to-transparent" />
    </section>
  )
}
