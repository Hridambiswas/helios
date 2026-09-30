/**
 * Sol — parametric config for the procedural ermine mascot.
 *
 * All colours, proportions, and animation strengths live here so the
 * mascot can be tuned without touching the geometry code.
 */

export const solConfig = {
  // Colours — Sol reads "snow-white" but is actually a hair cooler
  // than the snowfield (--snow) so he doesn't disappear into the
  // crest. Roughly matches a real ermine's slightly-bluer winter coat
  // under a warm-lit horizon.
  fur:          '#EEF2F8',   // --snow (pure white per DESIGN_BRIEF)
  furShadow:    '#3E4E7A',   // deep cool shadow (between --frost and --polar-night)
  furRim:       '#F4B942',   // --sun; warm amber rim on the sun-facing side
  earInner:     '#F2C4CC',   // soft pink
  eye:          '#0E1526',   // near-black, glossy
  nose:         '#231A1F',   // dark
  tailTip:      '#181022',   // near-black
  tailTipGlow:  '#B04A50',   // very subtle ember when active

  // Anatomy (units are R3F world-space; tuned to fit inside the hero
  // scene's ortho camera at zoom=1)
  bodyLength:   0.62,        // half-length of the capsule body
  bodyRadius:   0.14,
  headRadius:   0.18,
  headOffset:   [0.62, 0.12, 0], // relative to body center
  earSize:      0.06,
  earSpread:    0.12,        // side-to-side ear separation
  earHeight:    0.16,        // above head center
  eyeSize:      0.028,
  eyeSpread:    0.08,
  eyeOffset:    [0.14, 0.02, 0.10], // forward, up, side (mirrored per eye)
  noseSize:     0.020,
  tailLength:   0.52,
  tailTipRatio: 0.22,        // fraction of the tail that is black

  // Animation strengths (multiplied by --motion-scale, see tokens.css)
  breathAmp:    0.02,
  breathHz:     0.9,
  blinkEvery:   4.2,         // seconds between blinks (with jitter)
  blinkDur:     0.14,
  tailSwishAmp: 0.15,        // radians
  tailSwishHz:  0.4,
  cursorEyeMax: 0.45,        // how far the eyes rotate toward the cursor
} as const

export type SolConfig = typeof solConfig
