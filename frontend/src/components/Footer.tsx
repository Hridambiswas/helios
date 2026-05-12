import { WingSilhouette, DragonEye, DragonScales } from './DragonDecor'

export function Footer() {
  return (
    <footer className="relative border-t pt-16 pb-10 px-4 mt-8 overflow-hidden"
      style={{ borderColor: 'rgba(201,162,39,0.12)', background: 'rgba(5,5,5,1)' }}>

      {/* Dragon scale background */}
      <DragonScales opacity={0.06} />

      {/* Dragon wing silhouettes */}
      <WingSilhouette side="left"  opacity={0.07} />
      <WingSilhouette side="right" opacity={0.07} />

      {/* Top fire divider line */}
      <div className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(201,162,39,0.4), rgba(255,107,0,0.3), rgba(201,162,39,0.4), transparent)' }} />

      <div className="relative z-10 max-w-5xl mx-auto">

        {/* Dragon eye + HELIOS wordmark */}
        <div className="flex items-center gap-4 mb-10 justify-center">
          <div className="h-px flex-1 max-w-[120px]"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(201,162,39,0.4))' }} />
          <div className="flex flex-col items-center gap-2">
            <DragonEye size={32} opacity={0.6} />
            <span
              style={{
                fontFamily: 'Impact, Arial Black, sans-serif',
                fontSize: '1.5rem',
                color: 'transparent',
                WebkitTextStroke: '1.5px #C9A227',
                textShadow: '0 0 20px rgba(201,162,39,0.4)',
                letterSpacing: '-0.02em',
              }}>
              HELIOS
            </span>
          </div>
          <div className="h-px flex-1 max-w-[120px]"
            style={{ background: 'linear-gradient(90deg, rgba(201,162,39,0.4), transparent)' }} />
        </div>

        {/* Links grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-10">
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase mb-3" style={{ color: '#C9A227' }}>
              ⟡ Platform
            </h4>
            <ul className="space-y-2">
              {['REST API', 'WebSocket Streaming', 'Celery Workers', 'Prometheus Metrics'].map(item => (
                <li key={item} className="font-mono text-[11px]" style={{ color: 'rgba(201,162,39,0.35)' }}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase mb-3" style={{ color: '#C9A227' }}>
              ⟡ Dragon Agents
            </h4>
            <ul className="space-y-2">
              {['Planner', 'Hybrid Retriever', 'Sandboxed Executor', 'Synthesizer', 'Critic'].map(item => (
                <li key={item} className="font-mono text-[11px]" style={{ color: 'rgba(201,162,39,0.35)' }}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-[10px] tracking-widest uppercase mb-3" style={{ color: '#C9A227' }}>
              ⟡ The Lair
            </h4>
            <ul className="space-y-2">
              {['PostgreSQL + CQRS', 'Redis + Rate Limiting', 'ChromaDB Vectors', 'OpenTelemetry', 'EC2 + Vercel'].map(item => (
                <li key={item} className="font-mono text-[11px]" style={{ color: 'rgba(201,162,39,0.35)' }}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-8"
          style={{ borderTop: '1px solid rgba(201,162,39,0.08)' }}>
          <span className="font-mono text-[10px]" style={{ color: 'rgba(201,162,39,0.22)' }}>
            HELIOS — DISTRIBUTED MULTI-MODAL AGENTIC AI
          </span>
          <div className="flex items-center gap-5">
            <a href="https://github.com/Hridambiswas/helios" target="_blank" rel="noopener noreferrer"
              className="font-mono text-[10px] uppercase tracking-wider transition-colors"
              style={{ color: 'rgba(201,162,39,0.3)', textDecoration: 'none' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#C9A227')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(201,162,39,0.3)')}>
              GitHub
            </a>
            <a href={`${import.meta.env.VITE_API_URL ?? ''}/docs`} target="_blank" rel="noopener noreferrer"
              className="font-mono text-[10px] uppercase tracking-wider transition-colors"
              style={{ color: 'rgba(201,162,39,0.3)', textDecoration: 'none' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#C9A227')}
              onMouseLeave={e => (e.currentTarget.style.color = 'rgba(201,162,39,0.3)')}>
              API Docs
            </a>
            <span className="font-mono text-[10px]" style={{ color: 'rgba(201,162,39,0.2)' }}>
              {new Date().getFullYear()} · HRIDAM BISWAS
            </span>
          </div>
        </div>

        {/* Dragon fire rune bottom */}
        <div className="text-center mt-6 font-mono text-[10px] tracking-[0.6em]"
          style={{ color: 'rgba(201,162,39,0.12)' }}>
          ⟡ ⟡ ⟡ ⟡ ⟡ ⟡ ⟡ ⟡ ⟡
        </div>
      </div>
    </footer>
  )
}
