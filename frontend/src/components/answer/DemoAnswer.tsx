import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import { PipelineState, Score } from '../../pipeline/events'

/**
 * DemoAnswer — the answer view rendered under the arc in demo mode.
 * A single reading column on --frost, markdown answer first, then a
 * quiet source list, then Critic + Verifier gauges, then follow-ups.
 * No grid of cards.
 */

interface Props {
  state: PipelineState
  onReset: () => void
}

export function DemoAnswer({ state, onReset }: Props) {
  const isRunning = state.phase !== 'done' && state.phase !== 'error'
  const isError   = state.phase === 'error'
  const result    = state.result

  return (
    <section
      aria-live="polite"
      style={{
        padding: '40px 24px 120px',
        background: 'var(--polar-night)',
      }}
    >
      <motion.article
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{
          maxWidth: 'var(--answer-column)',
          margin: '0 auto',
          background: 'var(--frost)',
          border: '1px solid var(--frost-hairline)',
          borderRadius: 'var(--radius-lg)',
          padding: '28px 32px',
        }}
      >
        {/* Status line */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 20,
        }}>
          <span className="text--meta" style={{ letterSpacing: 0.2 }}>
            {isRunning ? statusLabel(state.phase) : isError ? 'Something went wrong.' : 'Answer'}
          </span>
          {!isRunning && (
            <button
              onClick={onReset}
              className="text--meta helios-focus"
              style={{
                background: 'transparent',
                border: '1px solid var(--frost-hairline)',
                borderRadius: 'var(--radius-sm)',
                padding: '4px 10px',
                color: 'var(--snow-shadow)',
                cursor: 'pointer',
              }}
            >
              Ask another
            </button>
          )}
        </div>

        {isRunning && (
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <span aria-hidden style={dot(0)} />
            <span aria-hidden style={dot(1)} />
            <span aria-hidden style={dot(2)} />
          </div>
        )}

        {isError && (
          <p className="text" style={{ color: 'var(--corona)' }}>
            {state.errorMessage ?? 'Please try again in a minute.'}
          </p>
        )}

        {!isRunning && result && (
          <>
            <div className="text markdown-answer" style={{
              color: 'var(--snow)',
              fontSize: 'var(--step-0)',
              lineHeight: 1.6,
            }}>
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>

            {result.sources.length > 0 && (
              <>
                <hr style={{
                  border: 'none',
                  borderTop: '1px solid var(--frost-hairline)',
                  margin: '24px 0',
                }} />
                <span className="text--meta" style={{ display: 'block', marginBottom: 8 }}>
                  Sources
                </span>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                  {result.sources.map((s, i) => (
                    <li key={i} style={{ marginBottom: 10 }}>
                      <div className="text" style={{ color: 'var(--snow)' }}>{s.title}</div>
                      <div className="text--meta">
                        {s.domain}{s.snippet ? ` — ${s.snippet}` : ''}
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            )}

            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 16,
              marginTop: 24,
            }}>
              <Gauge label="Critic"   score={result.critic_scores} />
              <Gauge label="Verifier" score={result.verifier_scores} />
            </div>

            {result.latency_ms !== undefined && (
              <div className="text--meta numeric" style={{ marginTop: 16 }}>
                {(result.latency_ms / 1000).toFixed(2)}s end-to-end.
              </div>
            )}

            {result.follow_ups && result.follow_ups.length > 0 && (
              <>
                <hr style={{
                  border: 'none',
                  borderTop: '1px solid var(--frost-hairline)',
                  margin: '20px 0',
                }} />
                <span className="text--meta" style={{ display: 'block', marginBottom: 10 }}>
                  Follow-ups
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {result.follow_ups.map((q, i) => (
                    <span key={i} className="text--meta" style={{
                      border: '1px solid var(--frost-hairline)',
                      borderRadius: 999,
                      padding: '6px 12px',
                      color: 'var(--snow)',
                    }}>{q}</span>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </motion.article>
    </section>
  )
}

function Gauge({ label, score }: { label: string; score?: Score }) {
  const value = score?.overall ?? 0
  const pass  = score?.pass ?? false
  const pct   = Math.round(value * 100)
  const color = pass ? 'var(--sun)' : 'var(--corona)'
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span className="text" style={{ color: 'var(--snow)' }}>{label}</span>
        <span className="text numeric" style={{ color }}>
          {pct}
        </span>
      </div>
      <div style={{
        height: 4,
        background: 'var(--frost-hairline)',
        borderRadius: 2,
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: color,
          transition: 'width 0.5s ease-out',
        }} />
      </div>
    </div>
  )
}

function statusLabel(phase: string) {
  switch (phase) {
    case 'planning':     return 'Planner is decomposing the question…'
    case 'retrieving':   return 'Retriever is pulling relevant chunks…'
    case 'executing':    return 'Executor is running the sandbox…'
    case 'synthesizing': return 'Synthesizer is writing the answer…'
    case 'evaluating':   return 'Critic and Verifier are checking the answer…'
    default:             return 'Working…'
  }
}

function dot(i: number): React.CSSProperties {
  return {
    display: 'inline-block',
    width: 6, height: 6, borderRadius: '50%',
    background: 'var(--snow-shadow)',
    animation: `heliosPulse 1.2s ease-in-out ${i * 0.15}s infinite`,
  }
}
