import { motion } from 'framer-motion'
import { usePipeline } from '../../pipeline/PipelineProvider'
import { AgentId } from '../../pipeline/events'

/**
 * The Sun Path — six agents along the day's arc.
 *
 * Idle: static SVG arc + six labelled stops with a one-line description.
 * Live (during a query): the sun moves along the arc as each agent's
 * phase arrives; the active agent's label brightens and its elapsed
 * time counts up in tabular figures below the label. Done agents keep
 * their final elapsed time; the sun rests at the last active stop.
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

// Approximate positions along a shallow arc y = 100 - sin(x·π)·62,
// x in [0..1]. Six evenly-spaced stops.
function stopFor(i: number, total: number) {
  const t = (i + 0.5) / total
  return {
    x: t * 100,
    y: 100 - Math.sin(t * Math.PI) * 62,
  }
}

export function ArcSection() {
  const { state } = usePipeline()

  // Which agents are ACTIVE (currently working) and which are DONE (already ran).
  const isActive = (id: AgentId) => state.activeAgents.includes(id)
  const isDone   = (id: AgentId) => state.finishedAt[id] !== undefined
  const isFailed = (id: AgentId) => state.phase === 'error' && (isActive(id) || isDone(id))

  // Sun follows the highest-index active agent so it always moves forward.
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

        <div style={{ position: 'relative', height: 260 }}>
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
              d={arcPath(0, 100, 62)}
              stroke="var(--frost-hairline)"
              strokeWidth="0.4"
              fill="none"
              strokeLinecap="round"
            />
          </svg>

          {/* Travelling sun — animates along the arc as sunIndex updates. */}
          {sunPos && !isIdle && (
            <motion.div
              layout
              animate={{ left: `${sunPos.x}%`, top: `${sunPos.y}%` }}
              transition={{ type: 'spring', stiffness: 120, damping: 22 }}
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
                  width: 130,
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
                {(active || done) && ms !== undefined && (
                  <span className="text--meta numeric" style={{ color: 'var(--snow-shadow)' }}>
                    {(ms / 1000).toFixed(1)}s
                  </span>
                )}
              </motion.div>
            )
          })}
        </div>

        {/* One-liners below the arc — always visible so the page reads well while idle. */}
        <ul
          style={{
            listStyle: 'none',
            padding: 0,
            marginTop: 48,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 20,
          }}
        >
          {AGENTS.map(a => (
            <li key={a.id}>
              <span className="text" style={{ color: 'var(--snow)', display: 'block' }}>
                {a.name}
              </span>
              <span className="text--meta" style={{ display: 'block', marginTop: 2 }}>
                {a.line}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

function arcPath(xStart: number, xEnd: number, height: number) {
  const c1x = xStart + (xEnd - xStart) * 0.25
  const c1y = 100 - height * 1.3
  const c2x = xStart + (xEnd - xStart) * 0.75
  const c2y = c1y
  return `M ${xStart},100 C ${c1x},${c1y} ${c2x},${c2y} ${xEnd},100`
}
