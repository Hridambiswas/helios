import { motion } from 'framer-motion'
import { usePipeline } from '../../pipeline/PipelineProvider'
import { AgentId } from '../../pipeline/events'

/**
 * The Sun Path — six agents along the day's arc.
 *
 * Idle: static SVG arc + six labelled stops with a one-line description
 *       for each label (no separate grid — H8 minimalist).
 * Live (during a query): the sun moves along the arc as each agent's
 *       phase arrives. The traversal samples the same cubic Bézier
 *       used to draw the path, so the marker is always exactly on the
 *       curve.
 */

export const AGENTS: Array<{
  id: AgentId; name: string; line: string; position: string;
}> = [
  { id: 'planner',     name: 'Planner',     line: 'Breaks the question into subtasks.',           position: 'sunrise' },
  { id: 'retriever',   name: 'Retriever',   line: 'Pulls relevant chunks: dense + BM25 + CLIP.',   position: 'morning' },
  { id: 'executor',    name: 'Executor',    line: 'Runs sandboxed Python if the plan needs code.', position: 'late morning' },
  { id: 'synthesizer', name: 'Synthesizer', line: 'Writes the grounded answer with citations.',    position: 'noon' },
  { id: 'critic',      name: 'Critic',      line: 'Scores groundedness, faithfulness, coverage.',  position: 'afternoon' },
  { id: 'verifier',    name: 'Verifier',    line: 'Cross-checks against a second model (Gemini).', position: 'sunset' },
]

// ── Bézier control points (must match the <path d=…> below) ────────────────
const ARC_HEIGHT = 62
const P0X = 0,  P0Y = 100
const P1X = 100, P1Y = 100
const C1X = 25,  C1Y = 100 - ARC_HEIGHT * 1.3
const C2X = 75,  C2Y = 100 - ARC_HEIGHT * 1.3
const ARC_PATH = `M ${P0X},${P0Y} C ${C1X},${C1Y} ${C2X},${C2Y} ${P1X},${P1Y}`

function bezierAt(t: number): { x: number; y: number } {
  const inv = 1 - t
  const inv2 = inv * inv
  const inv3 = inv2 * inv
  const t2 = t * t
  const t3 = t2 * t
  return {
    x: inv3 * P0X + 3 * inv2 * t * C1X + 3 * inv * t2 * C2X + t3 * P1X,
    y: inv3 * P0Y + 3 * inv2 * t * C1Y + 3 * inv * t2 * C2Y + t3 * P1Y,
  }
}

// t-values for each agent stop: evenly spaced, biased inward so the
// first/last labels sit inside the visible arc rather than at the
// endpoints where the curve is flat.
function stopT(i: number, total: number) {
  return (i + 0.5) / total
}

function stopFor(i: number, total: number) {
  return bezierAt(stopT(i, total))
}

export function ArcSection() {
  const { state } = usePipeline()

  const isActive = (id: AgentId) => state.activeAgents.includes(id)
  const isDone   = (id: AgentId) => state.finishedAt[id] !== undefined
  const isFailed = (id: AgentId) => state.phase === 'error' && (isActive(id) || isDone(id))

  const sunIndex = (() => {
    const activeIdx = AGENTS
      .map((a, i) => (isActive(a.id) ? i : -1))
      .filter(i => i >= 0)
      .pop()
    if (activeIdx !== undefined) return activeIdx
    const doneIdx = AGENTS
      .map((a, i) => (isDone(a.id) ? i : -1))
      .filter(i => i >= 0)
      .pop()
    return doneIdx ?? -1
  })()
  const sunPos = sunIndex >= 0 ? stopFor(sunIndex, AGENTS.length) : null
  const isIdle = state.phase === 'idle'

  return (
    <section
      aria-label="The six agents"
      style={{
        position: 'relative',
        padding: '80px 32px 120px',
        background: 'var(--polar-night)',
      }}
    >
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <motion.h2
          className="display"
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          style={{
            color: 'var(--snow)',
            fontSize: 'var(--step-3)',
            marginBottom: 12,
          }}
        >
          Six agents. One sun path.
        </motion.h2>
        <motion.p
          className="text"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.5, delay: 0.1 }}
          style={{
            color: 'var(--snow-shadow)',
            marginBottom: 48,
            maxWidth: '52ch',
          }}
        >
          Every question walks the arc from sunrise to sunset. When you ask,
          the sun travels the arc as each agent finishes its work.
        </motion.p>

        <div style={{ position: 'relative', height: 320 }}>
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
            }}
            aria-hidden
          >
            <path
              d={ARC_PATH}
              stroke="var(--frost-hairline)"
              strokeWidth="0.4"
              fill="none"
              strokeLinecap="round"
            />
          </svg>

          {sunPos && !isIdle && (
            <motion.div
              animate={{ left: `${sunPos.x}%`, top: `${sunPos.y}%` }}
              transition={{ type: 'spring', stiffness: 140, damping: 24 }}
              style={{
                position: 'absolute',
                width: 22, height: 22,
                borderRadius: '50%',
                background: state.phase === 'error' ? 'var(--corona)' : 'var(--sun)',
                boxShadow: state.phase === 'error'
                  ? '0 0 0 8px var(--corona-soft), 0 0 32px var(--corona-soft)'
                  : '0 0 0 8px var(--sun-soft), 0 0 32px var(--sun-soft)',
                transform: 'translate(-50%, -50%)',
                pointerEvents: 'none',
                zIndex: 2,
              }}
            />
          )}

          {AGENTS.map((a, i) => {
            const { x, y } = stopFor(i, AGENTS.length)
            const active = isActive(a.id)
            const done   = isDone(a.id) && !active
            const failed = isFailed(a.id)
            const dotColor = failed
              ? 'var(--corona)'
              : active
                ? 'var(--sun)'
                : done
                  ? 'var(--snow)'
                  : 'var(--snow-shadow)'
            const labelColor = active || done ? 'var(--snow)' : 'var(--snow-shadow)'
            const ms = state.timings[a.id]
            return (
              <motion.div
                key={a.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.08 * i }}
                style={{
                  position: 'absolute',
                  left: `${x}%`,
                  top: `${y}%`,
                  transform: 'translate(-50%, -50%)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 6,
                  width: 150,
                  textAlign: 'center',
                  zIndex: 1,
                }}
              >
                <span
                  aria-hidden
                  style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: dotColor,
                    boxShadow: active
                      ? '0 0 0 4px var(--sun-soft)'
                      : '0 0 0 3px rgba(238,242,248,0.06)',
                    transition: 'background 0.3s',
                  }}
                />
                <span className="text" style={{
                  color: labelColor,
                  fontSize: 'var(--step-0)',
                  fontWeight: active ? 600 : 400,
                  transition: 'color 0.3s',
                }}>{a.name}</span>
                {/* One-line description under each label — replaces the
                    redundant grid the director flagged (Sev 3.7). Kept
                    quiet so it never fights the label for attention. */}
                <span className="text--meta" style={{
                  color: 'var(--snow-shadow)',
                  fontSize: 11,
                  lineHeight: 1.35,
                }}>
                  {a.line}
                </span>
                {(active || done) && ms !== undefined && (
                  <span className="text--meta numeric" style={{ color: 'var(--snow)', marginTop: 2 }}>
                    {(ms / 1000).toFixed(1)}s
                  </span>
                )}
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
