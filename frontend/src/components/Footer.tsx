export function Footer() {
  return (
    <footer
      className="relative border-t pt-16 pb-10 px-4 mt-8 overflow-hidden"
      style={{ borderColor: 'rgba(139,92,246,0.1)', background: '#000' }}
    >
      {/* Top glow line */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.5), rgba(192,38,211,0.3), rgba(139,92,246,0.5), transparent)' }}
      />

      <div className="relative z-10 max-w-5xl mx-auto">

        {/* Logo */}
        <div className="flex items-center gap-4 mb-10 justify-center">
          <div className="h-px flex-1 max-w-[120px]" style={{ background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.4))' }} />
          <div className="flex flex-col items-center gap-2">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="12" stroke="#8b5cf6" strokeWidth="1.2" strokeOpacity="0.6"/>
              <circle cx="14" cy="14" r="6"  fill="#8b5cf6" fillOpacity="0.15" stroke="#8b5cf6" strokeWidth="0.9"/>
              <circle cx="14" cy="14" r="2.5" fill="#a78bfa"/>
            </svg>
            <span
              style={{
                fontFamily: 'Impact, Arial Black, sans-serif',
                fontSize: '1.5rem',
                color: '#fff',
                textShadow: '0 0 20px rgba(139,92,246,0.6)',
                letterSpacing: '-0.02em',
              }}
            >
              HEL<span style={{ color: '#8b5cf6' }}>IOS</span>
            </span>
          </div>
          <div className="h-px flex-1 max-w-[120px]" style={{ background: 'linear-gradient(90deg, rgba(139,92,246,0.4), transparent)' }} />
        </div>

        {/* Links grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-10">
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase mb-3" style={{ color: '#8b5cf6' }}>
              ⟡ Platform
            </h4>
            <ul className="space-y-2">
              {['REST API', 'WebSocket Streaming', 'Celery Workers', 'Prometheus Metrics'].map(item => (
                <li key={item} className="font-mono text-[11px]" style={{ color: 'rgba(255,255,255,0.2)' }}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase mb-3" style={{ color: '#8b5cf6' }}>
              ⟡ Agents
            </h4>
            <ul className="space-y-2">
              {['Planner', 'Hybrid Retriever', 'Sandboxed Executor', 'Synthesizer', 'Critic'].map(item => (
                <li key={item} className="font-mono text-[11px]" style={{ color: 'rgba(255,255,255,0.2)' }}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase mb-3" style={{ color: '#8b5cf6' }}>
              ⟡ Stack
            </h4>
            <ul className="space-y-2">
              {['PostgreSQL + CQRS', 'Redis + Rate Limiting', 'ChromaDB Vectors', 'OpenTelemetry', 'EC2 + Vercel'].map(item => (
                <li key={item} className="font-mono text-[11px]" style={{ color: 'rgba(255,255,255,0.2)' }}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div
          className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-8"
          style={{ borderTop: '1px solid rgba(139,92,246,0.08)' }}
        >
          <span className="font-mono text-[10px]" style={{ color: 'rgba(255,255,255,0.15)' }}>
            HELIOS — DISTRIBUTED MULTI-MODAL AGENTIC AI
          </span>
          <div className="flex items-center gap-5">
            <a
              href="https://github.com/Hridambiswas/helios"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-[10px] uppercase tracking-wider transition-colors"
              style={{ color: 'rgba(255,255,255,0.2)', textDecoration: 'none' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#8b5cf6')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.2)')}
            >
              GitHub
            </a>
            <a
              href={`${import.meta.env.VITE_API_URL ?? ''}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-[10px] uppercase tracking-wider transition-colors"
              style={{ color: 'rgba(255,255,255,0.2)', textDecoration: 'none' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#8b5cf6')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.2)')}
            >
              API Docs
            </a>
            <span className="font-mono text-[10px]" style={{ color: 'rgba(255,255,255,0.12)' }}>
              {new Date().getFullYear()} · HRIDAM BISWAS
            </span>
          </div>
        </div>

        {/* Bottom rune */}
        <div
          className="text-center mt-6 font-mono text-[10px] tracking-[0.6em]"
          style={{ color: 'rgba(139,92,246,0.12)' }}
        >
          ⟡ ⟡ ⟡ ⟡ ⟡ ⟡ ⟡ ⟡ ⟡
        </div>
      </div>
    </footer>
  )
}
