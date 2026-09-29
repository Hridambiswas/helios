import { motion } from 'framer-motion'

/**
 * The Sun Path — six agents along the day's arc.
 * Idle state (this file, first pass): a static SVG arc + six labelled
 * stops. The live variant that reacts to pipeline events lands in a
 * follow-up commit alongside the demo mode.
 */

export const AGENTS = [
  { id: 'planner',     name: 'Planner',     line: 'Breaks the question into subtasks.',           position: 'sunrise' },
  { id: 'retriever',   name: 'Retriever',   line: 'Pulls relevant chunks: dense + BM25 + CLIP.',   position: 'morning' },
  { id: 'executor',    name: 'Executor',    line: 'Runs sandboxed Python if the plan needs code.', position: 'late morning' },
  { id: 'synthesizer', name: 'Synthesizer', line: 'Writes the grounded answer with citations.',    position: 'noon' },
  { id: 'critic',      name: 'Critic',      line: 'Scores groundedness, faithfulness, coverage.',  position: 'afternoon' },
  { id: 'verifier',    name: 'Verifier',    line: 'Cross-checks against a second model (Gemini).', position: 'sunset' },
] as const

// Approximate positions along a shallow arc y = -sin(x·π), x in [0,1].
// Six evenly-spaced stops give the sun-path shape without a real curve solver.
function stopFor(i: number, total: number) {
  const t = (i + 0.5) / total
  return {
    x: t * 100,
    y: 100 - Math.sin(t * Math.PI) * 62,
  }
}

export function ArcSection() {
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
          {/* Arc line (SVG for crispness) */}
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

          {/* Agent stops */}
          {AGENTS.map((a, i) => {
            const { x, y } = stopFor(i, AGENTS.length)
            return (
              <motion.div
                key={a.id}
                initial={{ opacity: 0, y: 8 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{ duration: 0.5, delay: 0.08 * i }}
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
                }}
              >
                <span
                  aria-hidden
                  style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: 'var(--snow)',
                    boxShadow: '0 0 0 3px rgba(238,242,248,0.08)',
                  }}
                />
                <span className="text" style={{
                  color: 'var(--snow)',
                  fontSize: 'var(--step-0)',
                }}>{a.name}</span>
              </motion.div>
            )
          })}
        </div>

        {/* One-liners below the arc — meta only, no card grid. */}
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

// Cubic-approximation of y = h·sin(πx / xEnd) over [xStart, xEnd].
// Just precise enough for a decorative curve; not used for hit-testing.
function arcPath(xStart: number, xEnd: number, height: number) {
  const c1x = xStart + (xEnd - xStart) * 0.25
  const c1y = 100 - height * 1.3
  const c2x = xStart + (xEnd - xStart) * 0.75
  const c2y = c1y
  return `M ${xStart},100 C ${c1x},${c1y} ${c2x},${c2y} ${xEnd},100`
}
