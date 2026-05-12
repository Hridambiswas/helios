import { useRef } from 'react'
import type React from 'react'
import { motion, useInView } from 'framer-motion'

const AGENTS = [
  {
    num: '01',
    name: 'PLANNER',
    role: 'Query Decomposition',
    desc: 'Llama 3.3 70B analyses the query and produces a structured plan of sub-tasks — choosing between semantic, code, and web retrieval strategies.',
    tech: ['Llama 3.3 70B', 'LangGraph', 'T=0'],
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 2 L26 8 L26 20 L14 26 L2 20 L2 8 Z" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <circle cx="14" cy="14" r="3" fill="currentColor" opacity="0.6"/>
        <line x1="14" y1="2"  x2="14" y2="11" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
        <line x1="14" y1="17" x2="14" y2="26" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
      </svg>
    ),
  },
  {
    num: '02',
    name: 'RETRIEVER',
    role: 'Hybrid Search',
    desc: 'Fuses BAAI/bge dense embeddings, CLIP vision search, and BM25 sparse retrieval with RRF score fusion — top-K across all modalities.',
    tech: ['BAAI/bge', 'CLIP', 'BM25', 'ChromaDB'],
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <line x1="18" y1="18" x2="26" y2="26" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <line x1="12" y1="6"  x2="12" y2="18" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
        <line x1="6"  y1="12" x2="18" y2="12" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
      </svg>
    ),
  },
  {
    num: '03',
    name: 'EXECUTOR',
    role: 'Sandboxed Python',
    desc: 'Runs LLM-generated Python in a restricted AST sandbox — safe eval for data analysis, visualisation, and computation over retrieved docs.',
    tech: ['AST sandbox', 'RestrictedPython', 'NumPy'],
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <rect x="3" y="5" width="22" height="18" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        <path d="M8 10 L12 14 L8 18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <line x1="14" y1="18" x2="20" y2="18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    num: '04',
    name: 'SYNTHESIZER',
    role: 'Streaming Answer',
    desc: 'Streams a final answer token-by-token via Groq + WebSocket, weaving retrieved context, code results, and multi-turn conversation history.',
    tech: ['Llama 3.3 70B', 'Groq', 'WebSocket'],
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M4 14 Q10 6 14 14 Q18 22 24 14" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
        <circle cx="4"  cy="14" r="2" fill="currentColor" opacity="0.5"/>
        <circle cx="24" cy="14" r="2" fill="currentColor" opacity="0.5"/>
        <circle cx="14" cy="14" r="2.5" fill="currentColor"/>
      </svg>
    ),
  },
  {
    num: '05',
    name: 'CRITIC',
    role: 'LLM-as-Judge',
    desc: 'Scores the answer on groundedness, faithfulness, and completeness. If overall < 0.5, triggers one re-synthesis pass with specific improvement cues.',
    tech: ['Llama 3.3 70B', 'Min score 0.5', 'Retry loop'],
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 3 L17 10 H24 L18.5 14.5 L20.5 22 L14 18 L7.5 22 L9.5 14.5 L4 10 H11 Z"
          stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
      </svg>
    ),
  },
]

function AgentCard({ agent, index }: { agent: typeof AGENTS[0]; index: number }) {
  const ref    = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 48, rotateX: 12 }}
      animate={inView ? { opacity: 1, y: 0, rotateX: 0 } : {}}
      transition={{ duration: 0.65, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      className="relative flex flex-col p-6 border border-white/6 bg-white/[0.015] group cursor-default"
      style={{ perspective: '800px', transformStyle: 'preserve-3d' }}
      onMouseEnter={(e: React.MouseEvent<HTMLElement>) => {
        e.currentTarget.style.borderColor = 'rgba(196,30,58,0.4)'
        e.currentTarget.style.background  = 'rgba(196,30,58,0.04)'
        e.currentTarget.style.boxShadow   = '0 0 40px rgba(196,30,58,0.08), inset 0 1px 0 rgba(196,30,58,0.15)'
      }}
      onMouseLeave={(e: React.MouseEvent<HTMLElement>) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'
        e.currentTarget.style.background  = 'rgba(255,255,255,0.015)'
        e.currentTarget.style.boxShadow   = 'none'
      }}
    >
      {/* Number */}
      <div
        className="absolute top-4 right-4 font-mono text-[11px] select-none"
        style={{ color: 'transparent', WebkitTextStroke: '1px rgba(196,30,58,0.2)' }}
      >
        {agent.num}
      </div>

      {/* Icon */}
      <div
        className="text-crimson mb-5 transition-all duration-300 group-hover:drop-shadow-[0_0_12px_rgba(196,30,58,0.8)]"
        style={{ color: '#c41e3a' }}
      >
        {agent.icon}
      </div>

      {/* Name */}
      <div className="font-mono text-xs tracking-widest text-crimson mb-1">{agent.name}</div>
      <div className="font-mono text-[10px] text-white/30 mb-4 tracking-wider">{agent.role}</div>

      {/* Description */}
      <p className="text-white/55 text-[12px] leading-relaxed flex-1 mb-5">{agent.desc}</p>

      {/* Tech pills */}
      <div className="flex flex-wrap gap-1.5">
        {agent.tech.map(t => (
          <span
            key={t}
            className="font-mono text-[9px] px-2 py-0.5 border border-white/8 text-white/30 tracking-wider"
          >
            {t}
          </span>
        ))}
      </div>

      {/* Active glow line at bottom */}
      <motion.div
        className="absolute bottom-0 left-0 h-[1px]"
        style={{ background: 'linear-gradient(90deg, transparent, #c41e3a, transparent)', width: '0%' }}
        whileHover={{ width: '100%' }}
        transition={{ duration: 0.4 }}
      />
    </motion.div>
  )
}

// Animated connection line between cards
function ConnectorLine({ index }: { index: number }) {
  const ref    = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-100px' })

  return (
    <div ref={ref} className="hidden lg:flex items-center justify-center self-center flex-shrink-0 mx-1">
      <div className="relative w-10 h-px overflow-hidden">
        <div className="absolute inset-0 bg-white/8" />
        <motion.div
          className="absolute inset-0"
          style={{ background: 'linear-gradient(90deg, transparent, #c41e3a, transparent)' }}
          initial={{ scaleX: 0, originX: 0 }}
          animate={inView ? { scaleX: 1 } : {}}
          transition={{ duration: 0.5, delay: index * 0.12 + 0.4 }}
        />
      </div>
      <motion.div
        className="w-1 h-1 bg-crimson rotate-45 flex-shrink-0 ml-[-1px]"
        initial={{ opacity: 0, scale: 0 }}
        animate={inView ? { opacity: 1, scale: 1 } : {}}
        transition={{ delay: index * 0.12 + 0.6, duration: 0.3 }}
        style={{ boxShadow: '0 0 6px rgba(196,30,58,0.8)' }}
      />
    </div>
  )
}

export function PipelineSection() {
  const titleRef    = useRef(null)
  const titleInView = useInView(titleRef, { once: true, margin: '-60px' })

  return (
    <section
      id="pipeline-section"
      className="relative min-h-screen bg-black py-24 px-6 overflow-hidden flex flex-col justify-center"
    >
      {/* Background grid (offset from hero) */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(196,30,58,1) 1px, transparent 1px), linear-gradient(90deg, rgba(196,30,58,1) 1px, transparent 1px)',
          backgroundSize:     '80px 80px',
          backgroundPosition: '40px 40px',
        }}
      />

      {/* Subtle radial behind pipeline */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 80% 60% at 50% 50%, rgba(196,30,58,0.05) 0%, transparent 65%)',
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto w-full">

        {/* Section header */}
        <motion.div
          ref={titleRef}
          initial={{ opacity: 0, y: 24 }}
          animate={titleInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="mb-16 text-center lg:text-left"
        >
          <div
            className="font-mono text-[clamp(40px,6vw,72px)] leading-none mb-2 select-none"
            style={{ color: 'transparent', WebkitTextStroke: '1px rgba(196,30,58,0.1)' }}
          >
            02
          </div>
          <h2
            className="font-mono text-white/90 tracking-widest uppercase"
            style={{ fontSize: 'clamp(22px, 4vw, 38px)' }}
          >
            The Pipeline
          </h2>
          <p className="font-mono text-white/30 text-sm mt-3 tracking-wider">
            Five specialised agents — sequential, deterministic, observable
          </p>
          <div className="mt-4 h-px bg-gradient-to-r from-crimson/40 via-crimson/10 to-transparent" />
        </motion.div>

        {/* Cards row */}
        <div className="flex flex-col lg:flex-row items-stretch gap-0 lg:gap-0">
          {AGENTS.map((agent, i) => (
            <div key={agent.name} className="flex flex-col lg:flex-row items-stretch flex-1">
              <div className="flex-1">
                <AgentCard agent={agent} index={i} />
              </div>
              {i < AGENTS.length - 1 && <ConnectorLine index={i} />}
            </div>
          ))}
        </div>

        {/* Bottom tagline */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={titleInView ? { opacity: 1 } : {}}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="mt-12 flex items-center gap-6 justify-center font-mono text-[11px] text-white/20 tracking-widest"
        >
          <span>OPENTELEMETRY TRACED</span>
          <span className="text-crimson/30">·</span>
          <span>PROMETHEUS METRICS</span>
          <span className="text-crimson/30">·</span>
          <span>CELERY ASYNC WORKERS</span>
        </motion.div>
      </div>
    </section>
  )
}
