import { useRef } from 'react'
import type React from 'react'
import { motion, useInView } from 'framer-motion'

const V  = '#8b5cf6'
const VD = 'rgba(139,92,246,0.35)'
const VM = 'rgba(139,92,246,0.12)'
const VL = 'rgba(139,92,246,0.06)'

const AGENTS = [
  {
    num: '01', name: 'PLANNER', role: 'Query Decomposition',
    desc: 'Llama 3.3 70B deconstructs your query into sub-tasks — mapping every angle before the first move.',
    tech: ['Llama 3.3 70B', 'LangGraph', 'T=0'],
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <polygon points="16,2 30,10 30,22 16,30 2,22 2,10" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <circle cx="16" cy="16" r="4" fill="currentColor" fillOpacity="0.4"/>
        <line x1="16" y1="2"  x2="16" y2="12" stroke="currentColor" strokeWidth="1" strokeOpacity="0.5"/>
        <line x1="16" y1="20" x2="16" y2="30" stroke="currentColor" strokeWidth="1" strokeOpacity="0.5"/>
        <line x1="2"  y1="10" x2="10" y2="14" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4"/>
        <line x1="30" y1="10" x2="22" y2="14" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4"/>
      </svg>
    ),
  },
  {
    num: '02', name: 'RETRIEVER', role: 'Hybrid Search',
    desc: 'CLIP + BAAI/bge + BM25 fused with RRF — semantic and keyword retrieval working in unison.',
    tech: ['BAAI/bge', 'CLIP', 'BM25', 'ChromaDB'],
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="13" cy="13" r="9" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <line x1="20" y1="20" x2="30" y2="30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <line x1="13" y1="6"  x2="13" y2="20" stroke="currentColor" strokeWidth="1" strokeOpacity="0.45"/>
        <line x1="6"  y1="13" x2="20" y2="13" stroke="currentColor" strokeWidth="1" strokeOpacity="0.45"/>
        <circle cx="13" cy="13" r="3" fill="currentColor" fillOpacity="0.3"/>
      </svg>
    ),
  },
  {
    num: '03', name: 'EXECUTOR', role: 'Sandboxed Python',
    desc: 'AST-restricted Python runs safely in isolation — pure computation without side effects.',
    tech: ['AST sandbox', 'RestrictedPython', 'NumPy'],
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <rect x="3" y="5" width="26" height="22" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <path d="M8 12 L14 17 L8 22" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        <line x1="16" y1="22" x2="24" y2="22" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        <line x1="3"  y1="10" x2="29" y2="10" stroke="currentColor" strokeWidth="1" strokeOpacity="0.35"/>
        <circle cx="7"  cy="7.5" r="1" fill="currentColor" fillOpacity="0.5"/>
        <circle cx="11" cy="7.5" r="1" fill="currentColor" fillOpacity="0.5"/>
        <circle cx="15" cy="7.5" r="1" fill="currentColor" fillOpacity="0.5"/>
      </svg>
    ),
  },
  {
    num: '04', name: 'SYNTHESIZER', role: 'Streaming Answer',
    desc: 'Token-by-token via Groq + WebSocket — knowledge delivered word by word in real time.',
    tech: ['Llama 3.3 70B', 'Groq', 'WebSocket'],
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <path d="M3 16 Q9 6 16 16 Q23 26 29 16" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
        <circle cx="3"  cy="16" r="2" fill="currentColor" fillOpacity="0.5"/>
        <circle cx="29" cy="16" r="2" fill="currentColor" fillOpacity="0.5"/>
        <circle cx="16" cy="16" r="3" fill="currentColor"/>
        <line x1="16" y1="3"  x2="16" y2="10" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.4"/>
        <line x1="16" y1="22" x2="16" y2="29" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.4"/>
      </svg>
    ),
  },
  {
    num: '05', name: 'CRITIC', role: 'LLM-as-Judge',
    desc: 'Scores groundedness, faithfulness & completeness — quality control at every step.',
    tech: ['Llama 3.3 70B', 'Min score 0.5', 'Retry'],
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <path d="M16 3 L19.5 12H29L21.5 17.5L24 27L16 22L8 27L10.5 17.5L3 12H12.5Z"
          stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
        <path d="M16 3 L19.5 12H29L21.5 17.5L24 27L16 22L8 27L10.5 17.5L3 12H12.5Z"
          fill="currentColor" fillOpacity="0.08"/>
      </svg>
    ),
  },
]

function AgentCard({ agent, index }: { agent: typeof AGENTS[0]; index: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 52 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className="relative flex flex-col p-6 group cursor-default overflow-hidden"
      style={{ border: `1px solid ${VM}`, background: VL }}
      onMouseEnter={(e: React.MouseEvent<HTMLElement>) => {
        e.currentTarget.style.borderColor = VD
        e.currentTarget.style.background  = 'rgba(139,92,246,0.05)'
        e.currentTarget.style.boxShadow   = `0 0 40px rgba(139,92,246,0.08), inset 0 1px 0 ${VM}`
      }}
      onMouseLeave={(e: React.MouseEvent<HTMLElement>) => {
        e.currentTarget.style.borderColor = VM
        e.currentTarget.style.background  = VL
        e.currentTarget.style.boxShadow   = 'none'
      }}
    >
      {/* Ghost number */}
      <div
        className="absolute top-4 right-4 font-mono text-[11px] select-none"
        style={{ color: 'transparent', WebkitTextStroke: '1px rgba(139,92,246,0.18)' }}
      >
        {agent.num}
      </div>

      {/* Icon */}
      <div className="mb-5 transition-all duration-300" style={{ color: V }}>
        {agent.icon}
      </div>

      {/* Labels */}
      <div className="font-mono text-xs tracking-widest mb-0.5" style={{ color: V }}>{agent.name}</div>
      <div className="font-mono text-[10px] mb-4 tracking-wider" style={{ color: VD }}>{agent.role}</div>

      {/* Description */}
      <p className="text-[12px] leading-relaxed flex-1 mb-5" style={{ color: 'rgba(255,255,255,0.45)' }}>{agent.desc}</p>

      {/* Tech pills */}
      <div className="flex flex-wrap gap-1.5">
        {agent.tech.map(t => (
          <span
            key={t}
            className="font-mono text-[9px] px-2 py-0.5 tracking-wider"
            style={{ border: `1px solid ${VM}`, color: VD }}
          >
            {t}
          </span>
        ))}
      </div>

      {/* Hover sweep line */}
      <motion.div
        className="absolute bottom-0 left-0 h-[1px]"
        style={{ background: `linear-gradient(90deg, transparent, ${V}, #a78bfa, transparent)`, width: '0%' }}
        whileHover={{ width: '100%' }}
        transition={{ duration: 0.4 }}
      />
    </motion.div>
  )
}

function Connector({ index }: { index: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-100px' })

  return (
    <div ref={ref} className="hidden lg:flex items-center justify-center flex-shrink-0 mx-0.5 self-center">
      <svg width="48" height="24" viewBox="0 0 48 24" fill="none">
        <motion.line x1="0" y1="12" x2="48" y2="12"
          stroke={V} strokeWidth="0.8" strokeOpacity="0.3"
          initial={{ pathLength: 0 }} animate={inView ? { pathLength: 1 } : {}}
          transition={{ duration: 0.5, delay: index * 0.12 }} />
        <motion.circle cx="24" cy="12" r="3"
          stroke={V} strokeWidth="1" fill="none" strokeOpacity="0.6"
          initial={{ scale: 0, opacity: 0 }} animate={inView ? { scale: 1, opacity: 1 } : {}}
          transition={{ delay: index * 0.12 + 0.3 }} />
        <motion.path d="M40 8 L48 12 L40 16" stroke={V} strokeWidth="1" fill="none" strokeOpacity="0.6"
          strokeLinecap="round" strokeLinejoin="round"
          initial={{ opacity: 0 }} animate={inView ? { opacity: 1 } : {}}
          transition={{ delay: index * 0.12 + 0.5 }} />
      </svg>
    </div>
  )
}

export function PipelineSection() {
  const titleRef    = useRef(null)
  const titleInView = useInView(titleRef, { once: true, margin: '-60px' })

  return (
    <section
      id="pipeline-section"
      className="relative min-h-screen flex flex-col justify-center py-24 px-6 overflow-hidden bg-black"
    >
      {/* Subtle violet radial */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: 'radial-gradient(ellipse 80% 60% at 50% 50%, rgba(109,40,217,0.05) 0%, transparent 70%)' }}
      />

      {/* Faint grid */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.018]"
        style={{
          backgroundImage: `linear-gradient(${V} 1px, transparent 1px), linear-gradient(90deg, ${V} 1px, transparent 1px)`,
          backgroundSize: '80px 80px',
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto w-full">

        {/* Header */}
        <motion.div
          ref={titleRef}
          initial={{ opacity: 0, y: 24 }}
          animate={titleInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="mb-16 text-center lg:text-left"
        >
          <div
            className="font-mono leading-none mb-2 select-none"
            style={{ fontFamily: 'Impact, Arial Black, sans-serif', fontSize: 'clamp(40px,6vw,72px)', color: 'transparent', WebkitTextStroke: '1px rgba(139,92,246,0.1)' }}
          >
            02
          </div>

          <h2
            className="font-mono tracking-widest uppercase"
            style={{ fontSize: 'clamp(22px,4vw,38px)', color: 'rgba(255,255,255,0.9)' }}
          >
            The Pipeline
          </h2>

          <p className="font-mono text-sm tracking-wider mt-2" style={{ color: VD }}>
            Five agents. One answer. No query escapes.
          </p>
          <div
            className="mt-4 h-px"
            style={{ background: `linear-gradient(90deg, ${V} 0%, rgba(139,92,246,0.1) 40%, transparent 100%)` }}
          />
        </motion.div>

        {/* Agent cards */}
        <div className="flex flex-col lg:flex-row items-stretch gap-0">
          {AGENTS.map((agent, i) => (
            <div key={agent.name} className="flex flex-col lg:flex-row items-stretch flex-1">
              <div className="flex-1"><AgentCard agent={agent} index={i} /></div>
              {i < AGENTS.length - 1 && <Connector index={i} />}
            </div>
          ))}
        </div>

        {/* Footer metrics */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={titleInView ? { opacity: 1 } : {}}
          transition={{ delay: 1.0, duration: 0.6 }}
          className="mt-12 flex items-center gap-6 justify-center font-mono text-[10px] tracking-widest"
          style={{ color: 'rgba(255,255,255,0.18)' }}
        >
          <span>OPENTELEMETRY TRACED</span>
          <span style={{ color: VD }}>⟡</span>
          <span>PROMETHEUS METRICS</span>
          <span style={{ color: VD }}>⟡</span>
          <span>CELERY ASYNC WORKERS</span>
        </motion.div>
      </div>
    </section>
  )
}
