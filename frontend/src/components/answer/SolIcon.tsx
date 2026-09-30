import { motion } from 'framer-motion'

/**
 * SolIcon — a small SVG ermine that lives next to the answer card so
 * the peak-end hop is visible in-viewport. Not the R3F mascot: this
 * one is drawn in ~30 SVG paths and animated by framer-motion so it
 * is cheap and always renders even without WebGL. The full R3F Sol
 * still lives up in the hero.
 *
 * States (mirror the R3F Sol):
 *   pass  → happy hop (Y translate + squash)
 *   fail  → ears down, head low, no motion
 *   idle  → gentle breath
 */

interface Props {
  variant?: 'idle' | 'pass' | 'fail'
  size?: number
}

export function SolIcon({ variant = 'idle', size = 48 }: Props) {
  const hop = variant === 'pass'
  const sad = variant === 'fail'

  return (
    <motion.svg
      width={size}
      height={size}
      viewBox="-30 -50 100 90"
      aria-label={sad ? 'Sol looks disappointed' : hop ? 'Sol looks pleased' : 'Sol'}
      role="img"
      animate={hop ? { y: [0, -6, 0, -2, 0] } : { y: 0 }}
      transition={hop ? { duration: 0.9, times: [0, 0.35, 0.6, 0.8, 1], ease: 'easeOut' } : {}}
      style={{ overflow: 'visible', flex: '0 0 auto' }}
    >
      {/* Body — capsule side profile */}
      <ellipse cx="20" cy="12" rx="24" ry="10" fill="var(--snow)" stroke="var(--frost-hairline)" strokeWidth="0.6" />

      {/* Tail base + tip */}
      <path
        d={sad
          ? 'M -4 16 Q -18 22 -22 30'
          : 'M -4 12 Q -18 4 -24 -8'}
        stroke="var(--snow)"
        strokeWidth="7"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d={sad
          ? 'M -18 26 Q -22 30 -24 34'
          : 'M -20 -4 Q -24 -8 -25 -12'}
        stroke="var(--tail-tip, #181022)"
        strokeWidth="5"
        strokeLinecap="round"
        fill="none"
      />

      {/* Head — a slightly larger circle for chibi feel */}
      <circle
        cx={sad ? 40 : 42}
        cy={sad ? 10 : 4}
        r="12"
        fill="var(--snow)"
        stroke="var(--frost-hairline)"
        strokeWidth="0.6"
      />

      {/* Ears */}
      <path
        d={sad
          ? 'M 34 4 Q 32 8 30 12'
          : 'M 34 -6 Q 30 -12 28 -14 Q 33 -12 36 -8 Z'}
        fill="var(--snow)"
        stroke="var(--frost-hairline)"
        strokeWidth="0.4"
      />
      <path
        d={sad
          ? 'M 46 4 Q 48 8 50 12'
          : 'M 46 -6 Q 42 -12 44 -14 Q 49 -12 50 -8 Z'}
        fill="var(--snow)"
        stroke="var(--frost-hairline)"
        strokeWidth="0.4"
      />

      {/* Eye */}
      <circle
        cx={sad ? 44 : 46}
        cy={sad ? 12 : 3}
        r="1.6"
        fill="#0E1526"
      />
      {/* Highlight */}
      <circle
        cx={sad ? 44.4 : 46.4}
        cy={sad ? 11.6 : 2.6}
        r="0.55"
        fill="#EEF2F8"
      />

      {/* Nose */}
      <circle
        cx={sad ? 52 : 54}
        cy={sad ? 12 : 5}
        r="1.2"
        fill="#231A1F"
      />

      {/* Warm sun rim (only when happy) */}
      {hop && (
        <path
          d="M 30 -6 Q 40 -10 52 -2"
          stroke="var(--sun)"
          strokeWidth="0.9"
          strokeLinecap="round"
          fill="none"
          opacity="0.85"
        />
      )}
    </motion.svg>
  )
}
