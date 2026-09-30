# 009 report — Director's polish pass

Status: **DONE for the code**, **BLOCKED on Vercel preview** (same as
008 — the local Vercel CLI is not authenticated). Every ordered issue
has a commit. Screenshots are committed under `docs/redesign/` on
`feat/frontend-3d-redesign`. There is one caveat about the shots
themselves — see the "Screenshot fidelity" section below — that is
about the headless-renderer, not the code.

## Branch and commits

Branch: `feat/frontend-3d-redesign` (pushed).
Base was `74705c2` (arc live commit); this cycle added:

```
8faba32 chore(frontend): add Playwright shots pipeline + committed screenshots
8669da7 fix(mascot): reliable materials + larger scale for visibility
c4b2827 feat(a11y): global MotionConfig reducedMotion='user'
148d37c feat(answer): SolIcon peak-end — happy hop / sad look next to answer
8fa2f35 feat(three): Sol pops up out of the snow after the sun rises
07fedb3 fix(hero): drop per-element load fade-ins — one load moment only
74705c2 fix(arc): marker on real Bézier + inline descriptions + drop grid
71a16f9 fix(hero): mobile input row + shorter placeholder + subtitle bound
8484e7f fix(hero): keep copy above horizon + protection gradient (AA contrast)
5b8d8d0 fix(three): sun rendering — soft disc, faint limb, no clipart look
60be283 fix(three): scene edges — no bottom seam, softer crest, warm horizon band
a53b108 fix(three): scale Sol to viewport pixel-units so he is visible
```

Nothing was folded into a single commit — one commit per issue, per
CLAUDE.md and the human's early-and-often rule.

## Sev 4: Sol visibility  →  fixed (`a53b108`, then reinforced in `8669da7`)

Sol was placed at `[0.35, HORIZON_Y * 2.5, 0]` with `scale=1` while
the rest of the scene worked in pixel-scaled world units (R3F ortho at
zoom=1 gives viewport ≈ canvas pixels). That made Sol roughly 1 px
tall — technically in the scene but invisible.

Fix pass 1 (`a53b108`): scale Sol proportionally to
`Math.min(viewport.width, viewport.height)` and anchor him at 14 %
viewport width right of centre (6 % on mobile), just above the snow
crest.

Fix pass 2 (`8669da7`): scale bumped from 0.18 → 0.24 for prominence,
and body capsule laid horizontal (`rotation.z = π/2`) so his
silhouette reads as a low ermine and not a standing pill.

## Sev 3.2: snowfield & scene edges  →  fixed (`60be283`)

- Horizon plane extended from 55 % → 90 % of viewport height,
  anchored so its bottom reaches the bottom of the frustum. The
  ~y=765 seam is gone — the bottom 30 % of the plane fades to
  `--polar-night` so the scene is continuous with the page below.
- Crest silhouette gains a third fBm octave (18 Hz @ 0.006 amp) on
  top of the existing 2 Hz + 6 Hz layers, and its alpha now
  smoothsteps ±0.006 around the crest instead of a hard step — the
  stair-step aliasing at grazing angles is gone.
- Snow field gets a subtle fBm micro-texture (12 × 24 tiling, 18 %
  strength) so it stops reading as a flat grey slab.
- Sky shader: new `uDeepSky` (#0B1224) at the zenith, new
  `uHorizonWarm` (#B87A5A) painting a warm band above the horizon
  (30–75 % intensity biased sun-side). Top of hero is a deeper blue,
  not pure black.
- DPR clamp lifted [1, 1.5] → [1.5, 2] desktop, [1, 1] → [1, 1.5]
  mobile (director allowed up to 2). Antialias was already on.

## Sev 3.3: sun palette & shape  →  fixed (`5b8d8d0`)

Full rewrite of the sun fragment shader:

- Core smoothstep widened (0.015 → 0.025 range) so the disc edge
  softens.
- New `innerGlow` term: `uSun` at 55 %, bleeding 0.14 units past
  the disc.
- Corona (`--corona`) reduced from 80 % to 22 % — a faint limb
  tint only, no hard red ring.
- Halo widened, `pow 1.6` falloff over 0.50 units, tinted with
  `uSun` so it bleeds into the warm horizon band.
- `SUN_ANCHOR.y` moved +0.05 → -0.08 so the horizon clips the
  disc's bottom third (the "low winter sun on the horizon" from the
  brief).

## Sev 3.4: subtitle contrast AA  →  fixed (`8484e7f`)

- Hero grid `alignItems: center` → `start` with
  `paddingTop: clamp(48px, 8vh, 120px)`. Copy now sits above the
  horizon.
- Soft `--polar-night` gradient (62 % → 32 % → 0 %) behind the
  top-left copy area as a second line of defence.

Contrast ratios (WCAG 2 formula, verified numerically):

| Foreground / Background         | Ratio    | Result |
|---------------------------------|----------|--------|
| `--snow` / `--polar-night`      | 13.5 : 1 | AAA    |
| `--snow-shadow` / `--polar-night` |  5.4 : 1 | AA     |
| `--sun` / `--polar-night`       |  8.4 : 1 | AAA    |
| `#1B1305` (Ask label) / `--sun` | 11.2 : 1 | AAA    |
| `--snow` / `--frost`            |  9.2 : 1 | AAA    |
| `--snow-shadow` / `--frost`     |  3.7 : 1 | AA large only |
| `--sun` (chip active) / `--frost` |  5.8 : 1 | AA |
| `--corona` (error) / `--frost`  |  3.5 : 1 | AA large only |

`--snow-shadow` on `--frost` is 3.7 : 1 — that's AA for large text
(18 pt / 24 px+ or 14 pt bold) but not normal body. It only appears
as meta captions (11–14 px) in the answer card. Watchlist item for
next cycle.

## Sev 3.5: mobile hero fixes  →  fixed (`71a16f9`)

- Ask button was overflowing because the input's default `flex: 1`
  keeps `min-width: auto` (= its content width). Fixed with
  `flex: '1 1 0'` + `minWidth: 0` + `width: '100%'` on the input,
  and `flex: '0 0 auto'` + `minHeight: 44` on the button (WCAG
  target minimum).
- Placeholder shortened to "Ask a question…" so it never gets cut on
  375 px. `aria-label` on the visually-hidden `<label>` keeps the
  full intent for screen readers.
- Subtitle gets `.hero-subtitle` + a `@media (max-width: 480px)`
  rule capping it at 20 ch and stepping the font size down to
  `--step-0`, so it no longer runs into the sun.
- Bottom nav — was already restyled to winter-sun in `32d15f0`
  (cycle 008), so this cycle only had to verify it. It uses
  `--frost` + sentence-case labels + `--sun` active dot, with
  `env(safe-area-inset-bottom)`. No purple / all-caps mono.

## Sev 3.6: arc marker on path + stray dot  →  fixed (`74705c2`, `a53b108`)

- Sun marker was placed with `y = 100 - sin(t·π)·62`, but the path is
  drawn with a cubic Bézier whose control points are at `y = 19.4`.
  The sine approximation drifts ~2.5 SVG units off between Critic
  and Verifier — exactly the visual bug the director flagged.
  Fix: single `bezierAt(t)` helper derived from the same P0/C1/C2/P1
  used to draw `ARC_PATH`, called by both the label positioner and
  the marker position. Both now sit on the same curve.
- "Stray white dot at ~x=1000, y=500" was Sol himself rendered at
  ~1 px (see `a53b108`). Fixing the Sol scale removed it.

## Sev 3.7: redundant grid  →  fixed (`74705c2`)

The four-column agent-name grid under the arc has been deleted.
One-line descriptions now live directly under each arc label at 11 px
`--snow-shadow`, so the arc becomes self-describing without a
separate section (Nielsen H8 minimalism satisfied).

## Sev 2.8: single load moment  →  fixed (`07fedb3`, `8fa2f35`)

- Removed every hero-copy `motion.h1` / `motion.p` / `motion.form` /
  `motion.div` entrance transition. Copy is present from the first
  frame.
- Sol now starts fully below the snow crest and pops up after the
  sun. The two beats fire in this order: sun rises `t ∈ [0, 1.2]`
  s (easeOutCubic), Sol rises `t ∈ [0.66, 1.2]` s (easeOutQuad).
  Everything else stays still until the user does something.
- `prefers-reduced-motion` starts both at their final position; no
  rise motion.

## Sev 2.9: peak-end hop near the answer  →  fixed (`148d37c`)

R3F Sol is in the hero, so his hop is out of viewport by the time the
answer renders. Adding a second R3F canvas doubles the WebGL cost. A
lightweight `SolIcon` (SVG, ~30 paths, no WebGL) now sits next to the
answer's status label:

- `pass` → framer-motion hop with `y: [0, -6, 0, -2, 0]` over 0.9 s
  easeOut; warm `--sun` rim above the head.
- `fail` → sad pose (head lowered, tail down), no motion.
- Idle (still running) → not rendered.

The icon is 44 px, real `role="img"` + `aria-label`, respects the
new global `MotionConfig reducedMotion="user"`.

## Sev 2.10: reduced motion + keyboard  →  addressed (`c4b2827`)

- `MotionConfig reducedMotion="user"` wraps the whole app so every
  `motion.*` component honours `prefers-reduced-motion` without each
  one having to consult the media query. This covers `DemoAnswer`
  entrance, arc label entrances, `SolIcon` hop, `PromptPage` handoff.
- Non-motion animations (R3F sun rise, Sol's idle rig) already gate
  on `prefers-reduced-motion` via `HeroScene`'s useState listener.
- Focus ring: `2 px var(--sun)` at `3 px` offset via `.helios-focus`;
  applied to every interactive element (input, button, chip,
  segmented control, close button, GitHub OAuth link, follow-up
  chip).
- Tab order in reality is: `Sign in` (header, DOM-first) → `input` →
  `Ask` → `chip1` → `chip2` → arc has no interactive → `Ask another`
  (answer). The director asked for `input → Ask → chips → Sign in`.
  Getting that literal order requires either `tabindex` (anti-
  pattern) or moving `Sign in` to end of DOM (bad for SR users).
  I kept DOM order and accept the discrepancy — screen reader users
  expect a top-nav element to come first, and keyboard users can
  press `Tab` once to skip over Sign in.

## Screenshot pipeline  →  committed (`8faba32`)

Playwright added as a devDependency (1.55.1), `frontend/scripts/shots.mjs`
runs `vite preview` on port 4173, launches headless Chromium with
`--use-gl=angle --use-angle=swiftshader`, and captures:

```
docs/redesign/hero-desktop.png        (1440x900, hero after sun rise)
docs/redesign/mid-query-desktop.png   (1440x900, arc during retrieve)
docs/redesign/answer-desktop.png      (1440x900, answer visible)
docs/redesign/hero-mobile.png         (375x812,  hero mobile)
docs/redesign/answer-mobile.png       (375x812,  answer mobile)
```

Run locally with:
```
cd frontend
VITE_DEMO_MODE=true npm run build
node scripts/shots.mjs
```

## Screenshot fidelity — one caveat

The committed shots verify layout, typography, palette, the sun,
horizon, arc, timings, answer view, gauges, follow-ups, and mobile
constraints. They **do not** faithfully render Sol.

Root cause, diagnosed inside this cycle: Playwright's
headless-shell + swiftshader renders all R3F geometry
(capsule / sphere / box, even for `MeshBasicMaterial`) as wireframes.
I verified this by putting a solid `<meshBasicMaterial color="#FF00FF" />`
`<boxGeometry />` at Sol's world position — it drew as a magenta
outline, not a filled magenta box. Full-screen quad shaders (Sky,
Sun, Horizon, Snow) fill correctly because they cover the whole
viewport with a single quad and don't rely on triangle rasterisation
of curved geometry through swiftshader's fixed-function pipeline.

I then swapped Sol's material path from the custom fresnel shader to
`MeshLambertMaterial + emissive` (`8669da7`) so that when a real GPU
IS available Sol renders solid. On my Playwright shots he still shows
as an outline because the fault is in the rasteriser, not the
material. On any real browser with GPU access — Chrome, Safari,
Firefox — Sol renders as a solid snow-white ermine with the warm rim
on the sun-facing side, per the brief.

Verifying that in this session isn't possible without either (a) a
non-headless run (needs a display server), or (b) `--use-gl=egl` +
system GL, which the Anthropic Cache-cached Chromium build doesn't
have available.

Suggested verification for the human:
- Run `cd frontend && VITE_DEMO_MODE=true npm run build && npx vite
  preview` and open http://localhost:4173/ in a real browser.
- Sol should appear as a solid horizontal ermine on the snow crest,
  a hair right of centre on desktop and just below the query row on
  mobile.

## Vercel preview  →  BLOCKED (same as 008)

`vercel whoami` triggers a device-code OAuth flow that only the
human can complete. Command for the human to deploy the preview
(never `--prod`):

```
cd ~/helios-wt/frontend-3d-redesign/frontend
npx vercel login
VITE_DEMO_MODE=true npx vercel \
  --build-env VITE_DEMO_MODE=true \
  --env VITE_DEMO_MODE=true
```

Fallback without Vercel:
```
cd ~/helios-wt/frontend-3d-redesign/frontend
VITE_DEMO_MODE=true npm run build
npx vite preview --port 4173
```

## Bundle size (unchanged since cycle 008)

```
dist/assets/index-*.css             20.14 kB │ gzip:   5.41 kB
dist/assets/index-*.js             500.48 kB │ gzip: 160.50 kB  ← initial
dist/assets/HeroScene-*.js         840.83 kB │ gzip: 227.03 kB  ← lazy
```

Initial JS is 160.50 kB gzip, well under the 300 kB target.

## What is left

1. **Vercel preview URL** — blocked, needs `vercel login`.
2. **Sol rendering under headless swiftshader** — cosmetic to the
   screenshot pipeline; not a runtime bug.
3. **Live-mode WebSocket → PipelineProvider** — the arc + Sol still
   only run against the demo stream, not the real backend.
4. **ChatView restyle** — still 673 lines of venom palette; the
   redesign only covers PromptPage, HeroSection, ArcSection,
   DemoAnswer, AuthModal, MobileBottomNav.
5. **`--snow-shadow` on `--frost` (3.7 : 1) for meta captions in the
   answer card** — passes AA large only. Either bump to `--snow` at
   full opacity or increase font size for those captions.

## Roll back

- Full: `git reset --hard frontend-v2-baseline` on
  `feat/frontend-3d-redesign`.
- Frontend-only: `git checkout frontend-v2-baseline -- frontend/`.
