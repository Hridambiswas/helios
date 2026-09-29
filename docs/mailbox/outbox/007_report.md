# 007 report — Frontend redesign with a 3D mascot

Status: **PARTIAL**. First block landed (baseline tag, working branch,
design tokens, fonts). The bulk of the redesign (mascot, orbit, agent
live view, demo mode, screenshots, preview) is queued for subsequent
cycles.

## What shipped this cycle

### Rollback plan
- Tag `frontend-v2-baseline` created on `feat/hf-space-deploy` HEAD
  (`4ca0a56`) and pushed to `origin`.
- Rollback in one command:
  `git checkout frontend-v2-baseline -- frontend/`
  (or reset the working branch with
  `git reset --hard frontend-v2-baseline`).

### Working branch
- `feat/frontend-3d-redesign` created from `chore/repo-hygiene`
  (which is ahead of `feat/hf-space-deploy` with the repo-hygiene
  commits — safer base since the same CI runs will apply to this
  branch).
- Pushed to `origin/feat/frontend-3d-redesign`.
- Work is done in a worktree at
  `~/helios-wt/frontend-3d-redesign` so the mailbox on
  `fix/request-failed` stays visible.

### Commits (feat/frontend-3d-redesign, on top of chore/repo-hygiene)

```
9d0be96 feat(frontend): winter-sun design tokens and fonts
```

Details:
- `frontend/src/styles/tokens.css` (new) — winter-sun palette
  (`--polar-night` #141D33, `--frost` #2A365A, `--snow-shadow`
  #6F86B3, `--snow` #EEF2F8, `--sun` #F4B942, `--corona` #E86A8A), the
  1.25 type scale (14/17/21/26/33/41/64 + `.display--hero` clamp
  48px→96px), font stacks, radii, focus ring, and a
  `--motion-scale` variable gated on `prefers-reduced-motion` so
  every component can honour it via one variable.
- `frontend/src/main.tsx` — imports `tokens.css` **after**
  `globals.css` so the new palette overrides the legacy `--violet`
  variables. Existing components keep working via a semantic bridge
  (`--text-primary`, `--card-bg`, etc. now resolve to snow/frost).
- `frontend/index.html` — swaps Google Fonts for Unbounded (display)
  and Atkinson Hyperlegible Next (text); keeps DM Sans + IBM Plex
  Mono temporarily because unmigrated venom components still
  reference them. Both are dropped once the venom components are
  gone. `theme-color` updated to `--polar-night` so mobile
  chrome/Safari match.

Build passes:
```
$ cd frontend && npm run build
✓ 2115 modules transformed.
dist/assets/index-*.js   1,328.07 kB │ gzip: 382.70 kB
dist/assets/index-*.css     27.17 kB │ gzip:   6.48 kB
✓ built in 2.96s
```
Bundle is currently over the 300KB gzip target from the brief; that
is expected because the venom scenes (`DragonScene`, `Venom*`,
`FluidBackground`) are still imported. Reducing bundle size is part
of the mascot-and-scene work in the next cycles (lazy loading the
`HeroScene`, dropping venom code, splitting into chunks).

## What is left (roadmap for next cycles)

Ordered roughly by dependency:

1. **Scene teardown** — remove/gate venom & dragon components
   (`VenomHero`, `VenomOverlay`, `PurpleExplosion`, `DragonDecor`,
   `FluidBackground`, `LiquidCursor`, `three/DragonScene`) and the
   custom purple cursor. Keep `CustomCursor` for keyboard users;
   restore native cursor for pointer devices.
2. **Layout skeleton** — new `HeroSection` and `ArcSection` in
   `components/hero/` and `components/arc/`. Wire routing so the
   prompt page shows Hero+Arc, chat page shows the streamed answer
   view. Use `--frost` for surfaces, `--polar-night` for page bg.
3. **Sol (procedural mascot)** — new module under
   `src/three/mascot/`:
   - `config.ts` — parametric params (fur color, shadow tint, ear
     shape, inner colour, tail length + tip colour, eye size + colour,
     body length, glow strength).
   - `SolMesh.tsx` — geometry (capsule body + head, spheres for eyes
     + ears, curved tail).
   - `SolMaterial.tsx` — fresnel + fur shader, warm amber rim on the
     sun-facing side, cool blue shadow tint, subtle ember on the
     tail tip.
   - `SolRig.tsx` — animation state machine (idle, curious, playful,
     loaf, planning, retrieving, executing, synthesising,
     critic/verifier pass/fail, error). Idle = breathing, blinking,
     whiskers, ear flicks, tail swish. Curious = alert pose,
     cursor-follow eyes.
   - Reduced motion: only blink.
4. **Sun corona shader** — one continuous R3F canvas behind the hero
   with the horizon, the low sun, drifting fine snow, Sol on the
   snow crest. Lazy-loaded chunk with a poster placeholder so the
   page is usable before the canvas mounts.
5. **The arc (six agents)** — planner → retriever → executor →
   synthesizer → critic → verifier, positioned along the sun's daily
   arc. Idle state = static arc with one line per agent. Live state
   = sun traverses the arc, elapsed time in tabular figures per
   agent, Sol trots along the snow following the sun. Uses the
   existing WebSocket event stream from `websocket.py`. Fixes the
   stale "re-rank" copy (it is six agents now).
6. **Answer view** — single 68ch reading column on `--frost`,
   markdown-rendered answer first, then a quiet source list (title,
   domain, one line), then Critic + Verifier gauges (thin arc,
   tabular numbers), then follow-up chips.
7. **Chat page, history, auth modal, upload panel** — restyle
   consistently on the new palette. Nothing removed.
8. **Demo mode** — `VITE_DEMO_MODE=true` gate. Scripted event stream
   for all six agents with realistic delays, a canned markdown
   answer with citations and scores, no backend. Small "Demo" badge.
9. **Perf, a11y, mobile** — bundle split so initial JS <300KB gzip
   excluding the lazy 3D chunk; 60fps target on M-series; mobile
   (375px) uses lower DPR and fewer particles or a 2D fallback;
   keyboard nav, focus ring = 2px `--sun` at 3px offset; contrast AA
   on `--polar-night`.
10. **Screenshots + preview** — Playwright (add as devDep) shots at
    1440x900 (hero, mid-query, answer) and 375x812 (mobile) into
    `docs/redesign/`. If `frontend/.vercel/project.json` exists,
    `npx vercel` for a preview URL (never `--prod`); otherwise
    document `npm run preview`.
11. **Skills audit** — list every relevant skill used
    (frontend-design, react-best-practices, shadcn, next-forge,
    turbopack, ai-sdk, etc. — most Vercel skills are Next.js-centric
    and will not apply to this Vite + React app, but frontend-design
    is directly on-brief). Justify each new npm dep with bundle
    impact.

## Design skills survey (initial)

Relevant skills seen in the environment for this prompt (matched
against 007's "use every relevant skill" ask):

- `frontend-design:frontend-design` — direct match, will drive
  the visual design pass (color, type, rhythm, self-critique
  checklist).
- `vercel:shadcn` — some primitives (button, input, dialog) might
  be worth cherry-picking; the app is Vite, not Next.js, so
  registry install patterns need adapting.
- `vercel:react-best-practices` — trigger candidate once multiple
  TSX components are edited.
- `vercel:turbopack`, `vercel:nextjs`, `vercel:next-forge`,
  `vercel:next-cache-components`, `vercel:next-upgrade`,
  `vercel:vercel-functions`, `vercel:workflow`, `vercel:chat-sdk`
  — Next.js-specific, not applicable to this Vite app.
- `vercel:deploy`, `vercel:deployments-cicd`, `vercel:vercel-cli`,
  `vercel:env-vars` — relevant for the preview deployment step
  when we get there.
- `vercel:ai-sdk`, `vercel:ai-architect`, `vercel:ai-gateway` —
  the AI plumbing is on the backend; not needed here.
- `keybindings-help`, `update-config`, `fewer-permission-prompts`,
  `verify`, `code-review`, `security-review`, `init`, `review`,
  `claude-api`, `run`, `loop`, `schedule` — meta/tooling, not
  design.

Full report in the final 007 cycle will name each skill actually
invoked and what it changed.

## New dependencies

None yet. Planned:
- `@react-three/drei` and `@react-three/postprocessing` are already
  in `package.json` — Sol and the corona can be built with them.
- Playwright as devDep (screenshots only; not in prod bundle).
- Possibly a fresnel/fur helper (`three-fresnel-shader`), but only
  if a plain custom shader turns out to be materially harder.

## Bundle size baseline

Before the redesign work: `dist/assets/index-*.js` = 1328KB / 382.70KB
gzip; the target is <300KB gzip excluding the lazy 3D chunk. The
biggest wins will come from lazy-loading the R3F canvas and removing
the venom-era code.

## How to roll back

- Reset to baseline: `git reset --hard frontend-v2-baseline` on the
  redesign branch.
- Cherry-pick baseline into another branch:
  `git checkout frontend-v2-baseline -- frontend/`.

## Next cycle should

1. Start with the scene teardown (item 1 above) to shrink the
   bundle before adding new 3D code.
2. Land the layout skeleton and the Hero with the tokens applied so
   there is a visible before/after for the human to react to, even
   before Sol is fully built.
3. Commit each component individually (per CLAUDE.md's commit-early
   rule and the 80–216-commit target for 007) so partial progress
   is always in `origin/feat/frontend-3d-redesign`.
