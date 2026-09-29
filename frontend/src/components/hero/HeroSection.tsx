import { Suspense, lazy, RefObject } from 'react'
import { motion } from 'framer-motion'

// The R3F canvas is a lazy chunk so the hero renders text-first.
// Poster placeholder (a CSS gradient) covers the same box while the
// chunk downloads, so the page is always usable.
const HeroScene = lazy(() =>
  import('../../three/HeroScene').then(m => ({ default: m.HeroScene })),
)

const CHIPS = [
  'Compare BM25 and dense retrieval',
  'Explain RLHF in 5 lines',
] as const

interface Props {
  query: string
  setQuery: (v: string) => void
  onSubmit: (q?: string) => void
  inputRef: RefObject<HTMLInputElement>
}

export function HeroSection({ query, setQuery, onSubmit, inputRef }: Props) {
  return (
    <section
      aria-label="Ask Helios"
      style={{
        position: 'relative',
        minHeight: '100vh',
        paddingTop: 92,           /* clears the fixed header */
        paddingBottom: 40,
        overflow: 'hidden',
      }}
    >
      {/* Scene layer: sky, low sun, snowfield, Sol.
          Lives behind the text on desktop; sits on top on mobile (see @media). */}
      <div
        aria-hidden
        className="hero-scene-slot"
        style={{
          position: 'absolute',
          inset: 0,
          zIndex: 0,
          background:
            /* Fallback poster: soft polar-night → frost gradient with a
               warm sunset streak, so the hero has depth even without JS. */
            'radial-gradient(120% 60% at 78% 78%, var(--sun-soft) 0%, transparent 55%),' +
            'linear-gradient(180deg, var(--polar-night) 0%, #1B274A 55%, #253566 100%)',
        }}
      >
        <Suspense fallback={null}>
          <HeroScene />
        </Suspense>
      </div>

      {/* Content: left-aligned copy + input, anchored inside a max-width grid.
          Scene reads through the deliberate whitespace on the right. */}
      <div
        style={{
          position: 'relative',
          zIndex: 1,
          maxWidth: 1200,
          margin: '0 auto',
          padding: '0 32px',
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr)',
          gap: 40,
          minHeight: 'calc(100vh - 132px)',
          alignItems: 'center',
        }}
      >
        <div style={{ maxWidth: 580 }}>
          <motion.h1
            className="display--hero"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
            style={{ color: 'var(--snow)', marginBottom: 20 }}
          >
            Ask a hard question.
          </motion.h1>

          <motion.p
            className="text"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
            style={{
              color: 'var(--snow-shadow)',
              fontSize: 'var(--step-1)',
              maxWidth: '32ch',
              marginBottom: 36,
            }}
          >
            Six agents plan, search, compute, write and check each other&rsquo;s work.
          </motion.p>

          <motion.form
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
            onSubmit={e => { e.preventDefault(); onSubmit() }}
            style={{
              display: 'flex',
              gap: 10,
              background: 'var(--frost)',
              padding: 8,
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--frost-hairline)',
              alignItems: 'stretch',
            }}
          >
            <label htmlFor="helios-query" style={{ position: 'absolute', width: 1, height: 1, overflow: 'hidden', clip: 'rect(0 0 0 0)' }}>
              What would you like to know?
            </label>
            <input
              id="helios-query"
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="What would you like to know?"
              autoComplete="off"
              spellCheck={false}
              className="text helios-focus"
              style={{
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: 'var(--snow)',
                padding: '14px 18px',
                fontSize: 'var(--step-0)',
              }}
            />
            <button
              type="submit"
              className="helios-focus"
              style={{
                background: 'var(--sun)',
                color: '#1B1305',
                fontFamily: 'var(--font-text)',
                fontSize: 'var(--step-0)',
                fontWeight: 600,
                border: 'none',
                borderRadius: 'calc(var(--radius-lg) - 4px)',
                padding: '0 22px',
                cursor: 'pointer',
                transition: 'filter 0.15s',
              }}
              onMouseEnter={e => { e.currentTarget.style.filter = 'brightness(1.06)' }}
              onMouseLeave={e => { e.currentTarget.style.filter = 'none' }}
            >
              Ask
            </button>
          </motion.form>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.75 }}
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 10,
              marginTop: 20,
            }}
          >
            <span className="text--meta" style={{ alignSelf: 'center' }}>
              Try:
            </span>
            {CHIPS.map(chip => (
              <button
                key={chip}
                type="button"
                onClick={() => { setQuery(chip); onSubmit(chip) }}
                className="text--meta helios-focus"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--frost-hairline)',
                  borderRadius: 999,
                  padding: '8px 14px',
                  color: 'var(--snow)',
                  fontSize: 'var(--step--1)',
                  cursor: 'pointer',
                  transition: 'transform 0.15s, border-color 0.15s, background 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = 'var(--snow-shadow)'
                  e.currentTarget.style.background = 'var(--frost)'
                  e.currentTarget.style.transform = 'translateY(-1px)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = 'var(--frost-hairline)'
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.transform = 'translateY(0)'
                }}
              >
                {chip}
              </button>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  )
}
