import { useState, useEffect, useRef, useCallback } from 'react'
import { ChevronDown } from 'lucide-react'
import { api } from '../api/client'
import oniMask from '../assets/oni-mask.png'

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
  const [mouse, setMouse] = useState({ x: 0, y: 0 })
  const sectionRef = useRef<HTMLElement>(null)

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

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!sectionRef.current) return
    const rect = sectionRef.current.getBoundingClientRect()
    setMouse({
      x: ((e.clientX - rect.left) / rect.width - 0.5) * 2,
      y: ((e.clientY - rect.top) / rect.height - 0.5) * 2,
    })
  }, [])

  useEffect(() => {
    const el = sectionRef.current
    if (!el) return
    el.addEventListener('mousemove', handleMouseMove)
    return () => el.removeEventListener('mousemove', handleMouseMove)
  }, [handleMouseMove])

  const submit = () => {
    const q = query.trim()
    if (!q) return
    onQuerySubmit(q)
    setQuery('')
    document.getElementById('query-section')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden"
    >
      {/* Deep radial background */}
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 50% 40%, #1f0008 0%, #0d0005 40%, #080808 100%)' }}
      />

      {/* Animated grid */}
      <div className="absolute inset-0"
        style={{
          backgroundImage: 'linear-gradient(rgba(196,30,58,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(196,30,58,0.06) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
          animation: 'gridDrift 20s linear infinite',
        }}
      />

      {/* Oni mask ghost — 3D parallax background */}
      <div
        style={{
          position: 'absolute',
          right: '-5%',
          top: '50%',
          transform: `translateY(-50%) perspective(1200px) rotateY(${-15 + mouse.x * 6}deg) rotateX(${mouse.y * 4}deg)`,
          width: 'min(55vw, 680px)',
          opacity: 0.12,
          filter: 'drop-shadow(0 0 60px rgba(196,30,58,0.6)) saturate(0.4) brightness(0.7)',
          transition: 'transform 0.15s ease-out',
          pointerEvents: 'none',
          mixBlendMode: 'screen',
        }}
      >
        <img src={oniMask} alt="" style={{ width: '100%', display: 'block' }} />
      </div>

      {/* Horizontal scan lines */}
      <div className="absolute inset-0 pointer-events-none"
        style={{
          background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)',
        }}
      />

      {/* Top meta */}
      <div className="absolute top-8 left-8 flex items-center gap-3 z-10">
        <div className="status-dot" />
        <span className="font-mono text-xs text-crimson tracking-widest uppercase">Helios v1.0</span>
      </div>
      <div className="absolute top-8 right-8 text-right z-10">
        <div className="text-xs text-[#555] tracking-widest uppercase font-mono">AGENTIC AI</div>
        <div className="text-xs text-[#555] tracking-widest uppercase font-mono">PLATFORM</div>
      </div>

      {/* Main content */}
      <div className="relative z-10 text-center px-4 max-w-5xl w-full">
        <div className="section-number mb-2 select-none"
          style={{ color: 'transparent', WebkitTextStroke: '1px rgba(196,30,58,0.15)' }}>
          01
        </div>

        {/* Title with 3D depth */}
        <div style={{ perspective: '800px' }}>
          <h1
            className="font-display leading-none tracking-tight mb-4 animate-flicker"
            style={{
              fontFamily: 'Impact, Arial Black, sans-serif',
              fontSize: 'clamp(64px,14vw,180px)',
              transform: `perspective(800px) rotateX(${mouse.y * 3}deg) rotateY(${mouse.x * 3}deg)`,
              transition: 'transform 0.15s ease-out',
              textShadow: '0 0 80px rgba(196,30,58,0.3), 0 20px 40px rgba(0,0,0,0.8)',
            }}
          >
            <span className="text-white">HEL</span>
            <span className="text-crimson" style={{ textShadow: '0 0 40px rgba(196,30,58,0.7), 0 0 80px rgba(196,30,58,0.3)' }}>IOS</span>
          </h1>
        </div>

        {/* Tagline */}
        <div className="h-8 mb-8">
          <span className="font-mono text-sm tracking-[0.3em] text-[#888] uppercase cursor">
            {displayText}
          </span>
        </div>

        {/* Divider */}
        <div className="flex items-center gap-4 mb-10 justify-center">
          <div className="h-px flex-1 max-w-[120px] bg-gradient-to-r from-transparent to-crimson" />
          <div className="w-1.5 h-1.5 bg-crimson rotate-45" style={{ boxShadow: '0 0 8px rgba(196,30,58,0.8)' }} />
          <div className="h-px flex-1 max-w-[120px] bg-gradient-to-l from-transparent to-crimson" />
        </div>

        {/* Query input */}
        <div className="relative max-w-2xl mx-auto">
          <div
            className="card-dark rounded-none border border-crimson/30 flex items-stretch group transition-all duration-300 hover:border-crimson/60"
            style={{ boxShadow: '0 0 0 1px transparent, 0 8px 40px rgba(0,0,0,0.6)' }}
            onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 20px rgba(196,30,58,0.2), 0 8px 40px rgba(0,0,0,0.6)')}
            onMouseLeave={e => (e.currentTarget.style.boxShadow = '0 0 0 1px transparent, 0 8px 40px rgba(0,0,0,0.6)')}
          >
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
              className="px-6 bg-crimson hover:bg-crimson-light transition-all text-white font-mono text-xs tracking-widest uppercase border-l border-crimson/50"
              style={{ transition: 'background 0.2s, box-shadow 0.2s' }}
              onMouseEnter={e => (e.currentTarget.style.boxShadow = '0 0 20px rgba(196,30,58,0.5)')}
              onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}
            >
              RUN
            </button>
          </div>
          {!isLoggedIn && (
            <p className="text-xs text-[#555] mt-2 font-mono text-center">
              1 free query —{' '}
              <button onClick={onAuthClick} className="text-crimson hover:text-crimson-light transition-colors">
                sign in
              </button>
              {' '}to unlock unlimited
            </p>
          )}
        </div>

        {/* Stats */}
        <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-8 mt-12 text-xs font-mono text-[#555] tracking-wider">
          {[
            ['PLANNER', 'LLM DECOMPOSE'],
            ['RETRIEVER', 'HYBRID RAG'],
            ['EXECUTOR', 'SANDBOXED PY'],
            ['CRITIC', 'LLM-AS-JUDGE'],
          ].map(([label, sub]) => (
            <div key={label} className="text-center group cursor-default"
              style={{ transition: 'transform 0.2s' }}
              onMouseEnter={e => (e.currentTarget.style.transform = 'translateY(-2px)')}
              onMouseLeave={e => (e.currentTarget.style.transform = 'translateY(0)')}
            >
              <div className="text-crimson text-[10px] group-hover:text-crimson transition-colors"
                style={{ textShadow: '0 0 8px rgba(196,30,58,0)' }}
                onMouseEnter={e => ((e.currentTarget as HTMLElement).style.textShadow = '0 0 8px rgba(196,30,58,0.6)')}
                onMouseLeave={e => ((e.currentTarget as HTMLElement).style.textShadow = 'none')}
              >{label}</div>
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
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-[#444] z-10">
        <span className="font-mono text-[10px] tracking-widest uppercase">SCROLL</span>
        <ChevronDown size={16} className="animate-bounce" />
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-ink to-transparent" />

      <style>{`
        @keyframes gridDrift {
          0%   { background-position: 0 0; }
          100% { background-position: 60px 60px; }
        }
      `}</style>
    </section>
  )
}
