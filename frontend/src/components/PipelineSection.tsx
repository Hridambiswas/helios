const AGENTS = [
  {
    num: '01',
    name: 'PLANNER',
    role: 'Query Decomposition',
    description: 'Breaks your query into atomic subtasks. Decides whether retrieval, code execution, or direct synthesis is needed.',
    attrs: [
      { label: 'MODEL', value: 'Llama 3.3 70B' },
      { label: 'MAX SUBTASKS', value: '5' },
      { label: 'TEMPERATURE', value: '0.0' },
    ],
    color: '#c41e3a',
  },
  {
    num: '02',
    name: 'RETRIEVER',
    role: 'Hybrid Search',
    description: 'Runs three parallel retrieval paths — HuggingFace dense embeddings via ChromaDB, CLIP multi-modal embeddings, and BM25 sparse search — then score-fuses results.',
    attrs: [
      { label: 'DENSE WEIGHT', value: '0.6' },
      { label: 'BM25 WEIGHT', value: '0.1' },
      { label: 'TOP-K', value: '10' },
    ],
    color: '#8b1a2a',
  },
  {
    num: '03',
    name: 'EXECUTOR',
    role: 'Sandboxed Python',
    description: 'Runs LLM-generated Python in a sandboxed environment with import whitelisting and timeout enforcement. Returns stdout/stderr back into state.',
    attrs: [
      { label: 'TIMEOUT', value: '15s' },
      { label: 'SANDBOX', value: 'AST GUARD' },
      { label: 'BUILTINS', value: 'FILTERED' },
    ],
    color: '#6b0a1a',
  },
  {
    num: '04',
    name: 'SYNTHESIZER',
    role: 'Grounded Answer',
    description: 'Combines retrieved context and execution output into a final cited answer. Injects [doc_id] citations and appends a source block.',
    attrs: [
      { label: 'TEMPERATURE', value: '0.2' },
      { label: 'CITATIONS', value: 'INLINE' },
      { label: 'GROUNDING', value: 'ENFORCED' },
    ],
    color: '#c41e3a',
  },
  {
    num: '05',
    name: 'CRITIC',
    role: 'LLM-as-Judge',
    description: 'Scores the answer on groundedness, faithfulness, and completeness. Flags answers that fall below the minimum threshold for retry.',
    attrs: [
      { label: 'DIMENSIONS', value: '3' },
      { label: 'THRESHOLD', value: '0.7' },
      { label: 'TEMPERATURE', value: '0.0' },
    ],
    color: '#8b0000',
  },
]

export function PipelineSection() {
  return (
    <section className="py-20 px-4 relative">
      {/* Ink brush background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <svg viewBox="0 0 1400 600" className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 w-full opacity-[0.03]" preserveAspectRatio="none">
          <path d="M0,100 Q350,0 700,120 T1400,80 L1400,500 Q1050,600 700,480 T0,520 Z" fill="#c41e3a" />
        </svg>
      </div>

      <div className="max-w-5xl mx-auto">
        {/* Section header */}
        <div className="flex items-baseline gap-6 mb-16">
          <span className="section-number" style={{ fontSize: 'clamp(40px,6vw,72px)' }}>03</span>
          <div>
            <div className="hr-red w-16 mb-2" />
            <h2 className="font-mono text-xs tracking-[0.3em] uppercase text-crimson">Agent Pipeline</h2>
            <p className="text-white text-2xl font-light mt-1">Five-Stage Agentic Graph</p>
          </div>
        </div>

        {/* Agents */}
        <div className="space-y-6">
          {AGENTS.map((agent, i) => (
            <div key={agent.num} className="relative flex items-start gap-6 group">
              {/* Connector line */}
              {i < AGENTS.length - 1 && (
                <div className="absolute left-[19px] top-12 bottom-0 w-px bg-gradient-to-b from-crimson/40 to-transparent" style={{ height: 'calc(100% + 24px)' }} />
              )}

              {/* Number circle */}
              <div className="shrink-0 w-10 h-10 border border-crimson/50 flex items-center justify-center group-hover:border-crimson group-hover:bg-crimson/10 transition-all z-10">
                <span className="font-mono text-xs text-crimson">{agent.num}</span>
              </div>

              {/* Content */}
              <div className="flex-1 card-dark border border-white/5 p-5 group-hover:border-crimson/25 transition-all">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-mono text-crimson tracking-widest text-sm">{agent.name}</h3>
                      <div className="h-px flex-1 bg-crimson/20 max-w-[80px]" />
                      <span className="font-mono text-[10px] text-[#555] uppercase">{agent.role}</span>
                    </div>
                    <p className="text-[#888] text-sm leading-relaxed max-w-xl">{agent.description}</p>
                  </div>

                  {/* Attribute pills */}
                  <div className="flex flex-col gap-2 shrink-0">
                    {agent.attrs.map(({ label, value }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="font-mono text-[9px] text-[#555] tracking-wider uppercase w-24 text-right">{label}</span>
                        <div className="px-2 py-0.5 bg-crimson/10 border border-crimson/20">
                          <span className="font-mono text-[10px] text-crimson">{value}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
