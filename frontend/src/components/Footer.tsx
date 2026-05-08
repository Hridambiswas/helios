export function Footer() {
  return (
    <footer className="border-t border-white/5 py-12 px-4 mt-8">
      <div className="max-w-5xl mx-auto">
        {/* Top divider */}
        <div className="flex items-center gap-4 mb-8">
          <div className="h-px flex-1 bg-gradient-to-r from-crimson/50 to-transparent" />
          <span className="font-display text-white text-xl" style={{ fontFamily: 'Impact, Arial Black, sans-serif' }}>
            HEL<span className="text-crimson">IOS</span>
          </span>
          <div className="h-px flex-1 bg-gradient-to-l from-crimson/50 to-transparent" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase text-crimson mb-3">PLATFORM</h4>
            <ul className="space-y-2">
              {['REST API', 'WebSocket Streaming', 'Celery Workers', 'Prometheus Metrics'].map(item => (
                <li key={item} className="font-mono text-[11px] text-[#555]">{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase text-crimson mb-3">AGENTS</h4>
            <ul className="space-y-2">
              {['Planner', 'Hybrid Retriever', 'Sandboxed Executor', 'Synthesizer', 'Critic'].map(item => (
                <li key={item} className="font-mono text-[11px] text-[#555]">{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase text-crimson mb-3">INFRASTRUCTURE</h4>
            <ul className="space-y-2">
              {['PostgreSQL + CQRS', 'Redis + Rate Limiting', 'MinIO Object Store', 'ChromaDB Vectors', 'OpenTelemetry'].map(item => (
                <li key={item} className="font-mono text-[11px] text-[#555]">{item}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-8 border-t border-white/5">
          <span className="font-mono text-[10px] text-[#333]">
            HELIOS — DISTRIBUTED MULTI-MODAL AGENTIC AI
          </span>
          <span className="font-mono text-[10px] text-[#333]">
            {new Date().getFullYear()} · HRIDAM BISWAS
          </span>
        </div>
      </div>
    </footer>
  )
}
