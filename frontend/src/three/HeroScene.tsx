import { Suspense, useMemo, useRef, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Sol } from './mascot/Sol'
import { usePipeline } from '../pipeline/PipelineProvider'

/**
 * HeroScene — winter-sun R3F canvas.
 *
 * One continuous scene behind the hero:
 *   • Sky gradient plane (polar-night → frost → warm horizon)
 *   • Snowfield horizon (a soft crest across the lower third)
 *   • Low sun disc with a corona (radial glow + rim)
 *   • Drifting snow particles
 *
 * Mount animation (respects prefers-reduced-motion):
 *   The sun rises from behind the horizon over ~1.2s, then holds.
 *
 * Performance:
 *   • DPR clamped to [1, 1.5]
 *   • On viewports narrower than 768px, DPR is capped at 1 and the
 *     particle count is halved (still visually meaningful, cheaper).
 */

// ── Tokens shared between scene pieces ──────────────────────────────────────
const COLORS = {
  polarNight: '#141D33',
  frost:      '#2A365A',
  snow:       '#EEF2F8',
  snowShadow: '#6F86B3',
  sun:        '#F4B942',
  corona:     '#E86A8A',
}

// Screen-space anchor for the sun. x,y in [-1,+1] view coords (right/up).
// The sun sits low so the horizon clips its bottom half — the winter-sun rig.
const SUN_ANCHOR = { x: 0.62, y: -0.08 }
const SUN_RADIUS = 0.36
const HORIZON_Y  = -0.15

// ────────────────────────────────────────────────────────────────────────────
// Sky — full-screen plane, custom gradient shader.
function Sky() {
  const mat = useRef<THREE.ShaderMaterial>(null!)
  const { viewport } = useThree()
  useFrame((_, dt) => {
    // Trace a very slow time so the "warm" horizon breathes 0.5% —
    // subtle enough that it reads as a still image on first glance.
    mat.current.uniforms.uTime.value += dt
  })

  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uDeepSky:    { value: new THREE.Color('#0B1224') },      // deeper blue at zenith (not pure black)
    uPolarNight: { value: new THREE.Color(COLORS.polarNight) },
    uFrost:      { value: new THREE.Color(COLORS.frost) },
    uWarm:       { value: new THREE.Color('#7A5A6E') },      // pre-sunset blush at the horizon
    uHorizonWarm:{ value: new THREE.Color('#B87A5A') },      // warm glow band right above horizon
    uHorizon:    { value: HORIZON_Y },
  }), [])

  return (
    <mesh scale={[viewport.width, viewport.height, 1]} position={[0, 0, -3]}>
      <planeGeometry args={[1, 1, 1, 1]} />
      <shaderMaterial
        ref={mat}
        uniforms={uniforms}
        vertexShader={/* glsl */`
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `}
        fragmentShader={/* glsl */`
          precision highp float;
          uniform float uTime;
          uniform vec3  uDeepSky;
          uniform vec3  uPolarNight;
          uniform vec3  uFrost;
          uniform vec3  uWarm;
          uniform vec3  uHorizonWarm;
          uniform float uHorizon;
          varying vec2 vUv;
          void main() {
            float y = vUv.y;
            float horizonUv = 0.5 + uHorizon * 0.5; // convert [-1,+1] to [0,1]

            // Sky (above horizon): deeperSky at zenith → polarNight mid → frost near horizon
            // → warm blush right above the sun.
            float toZenith = smoothstep(horizonUv, 1.0, y);
            vec3 sky = mix(uPolarNight, uDeepSky, toZenith);
            sky = mix(uFrost, sky, smoothstep(horizonUv - 0.02, horizonUv + 0.30, y));
            // Warm band just above the horizon, brighter on the sun-facing right.
            float warmBand = smoothstep(horizonUv + 0.14, horizonUv, y);
            float warmSide = smoothstep(0.4, 1.0, vUv.x);
            sky = mix(sky, uHorizonWarm, warmBand * (0.30 + 0.45 * warmSide));
            sky = mix(uWarm, sky, smoothstep(horizonUv, horizonUv + 0.22, y));

            // Below horizon: gently drifting polar-night — snow/horizon plane will cover this,
            // but a faint reflected warmth helps sell continuity.
            float pulse = 0.5 + 0.5 * sin(uTime * 0.35 + vUv.x * 3.0);
            vec3 belowHorizon = mix(uPolarNight, uFrost * 0.55, y / max(horizonUv, 0.001));
            belowHorizon += 0.015 * pulse * vec3(0.6, 0.7, 1.0);

            vec3 color = mix(belowHorizon, sky, step(horizonUv, y));
            gl_FragColor = vec4(color, 1.0);
          }
        `}
      />
    </mesh>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// Sun — disc + corona. Positioned in screen space via a plane with a shader
// that draws a soft circle. Cheaper than lots of concentric meshes.
function Sun({ rise }: { rise: number }) {
  const { viewport } = useThree()
  const mat = useRef<THREE.ShaderMaterial>(null!)
  useFrame((_, dt) => { mat.current.uniforms.uTime.value += dt })

  const uniforms = useMemo(() => ({
    uTime:    { value: 0 },
    uSun:     { value: new THREE.Color(COLORS.sun) },
    uCorona:  { value: new THREE.Color(COLORS.corona) },
    uRadius:  { value: SUN_RADIUS * 0.5 },     // core radius in normalized quad
  }), [])

  // World-space anchor derived from viewport size so it stays on screen.
  const x = SUN_ANCHOR.x * viewport.width * 0.5
  const yBase = SUN_ANCHOR.y * viewport.height * 0.5
  // rise ∈ [0..1] pushes the sun up from below the horizon.
  const y = yBase - (1.0 - rise) * viewport.height * 0.35

  const size = Math.min(viewport.width, viewport.height) * 0.55

  return (
    <mesh position={[x, y, -2]}>
      <planeGeometry args={[size, size, 1, 1]} />
      <shaderMaterial
        ref={mat}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        vertexShader={/* glsl */`
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `}
        fragmentShader={/* glsl */`
          precision highp float;
          uniform float uTime;
          uniform vec3  uSun;
          uniform vec3  uCorona;
          uniform float uRadius;
          varying vec2 vUv;
          void main() {
            vec2 c = vUv - 0.5;
            float d = length(c);

            // Soft core: wide smoothstep so the edge isn't a hard clipart circle.
            // Falls off from full sun colour to zero over ~0.02 units.
            float core = smoothstep(uRadius + 0.020, uRadius - 0.005, d);

            // Warm inner glow that bleeds outward from the disc (uSun tint).
            float innerGlow = smoothstep(uRadius + 0.14, uRadius + 0.002, d);

            // Corona is a *very* faint limb tint just outside the disc.
            // Was a hard red ring — now a whisper.
            float limb = smoothstep(uRadius + 0.045, uRadius + 0.008, d)
                        - smoothstep(uRadius + 0.008, uRadius - 0.005, d);
            limb = max(limb, 0.0);

            // Wide halo that fades into the sky (long tail so no hard edge).
            float halo = pow(smoothstep(0.50, uRadius + 0.010, d), 1.6);

            // Slow breathing so it feels alive without shimmering.
            float breathe = 0.015 * sin(uTime * 0.8);

            // Colour build:
            //   core       full uSun
            //   innerGlow  uSun at 55%, added on top of the halo
            //   limb       uCorona at 22% only (a whisper)
            //   halo       uSun at 22% + breathe
            vec3 color = uSun    * core
                       + uSun    * innerGlow * 0.55
                       + uCorona * limb      * 0.22
                       + uSun    * halo      * (0.22 + breathe);

            float alpha = core
                        + innerGlow * 0.55
                        + limb      * 0.22
                        + halo      * (0.40 + breathe);
            if (alpha < 0.004) discard;
            gl_FragColor = vec4(color, alpha);
          }
        `}
      />
    </mesh>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// Horizon — a soft snow crest across the lower half of the screen.
// The plane runs from just above the crest down to the bottom of the
// canvas so there's no hard mid-page seam. The bottom of the plane
// fades into --polar-night so the scene reads as continuous with the
// page background below.
function Horizon() {
  const { viewport } = useThree()
  const mat = useRef<THREE.ShaderMaterial>(null!)
  useFrame((_, dt) => { mat.current.uniforms.uTime.value += dt })

  const uniforms = useMemo(() => ({
    uTime:       { value: 0 },
    uSnow:       { value: new THREE.Color(COLORS.snow) },
    uSnowShadow: { value: new THREE.Color(COLORS.snowShadow) },
    uPolarNight: { value: new THREE.Color(COLORS.polarNight) },
  }), [])

  // Position centred so the crest lands slightly above the mid-line
  // and the plane runs to the bottom of the frustum. Height covers
  // 90% of the viewport so there's no gap at the bottom.
  const planeHeight = viewport.height * 0.9
  const yCentre = HORIZON_Y * viewport.height * 0.5 - planeHeight * 0.30

  return (
    <mesh position={[0, yCentre, -1]}>
      <planeGeometry args={[viewport.width * 1.1, planeHeight, 1, 1]} />
      <shaderMaterial
        ref={mat}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        vertexShader={/* glsl */`
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `}
        fragmentShader={/* glsl */`
          precision highp float;
          uniform float uTime;
          uniform vec3 uSnow;
          uniform vec3 uSnowShadow;
          uniform vec3 uPolarNight;
          varying vec2 vUv;

          // hash + fBm for the crest silhouette and micro-texture
          float h(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
          float n(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f*f*(3.0-2.0*f);
            return mix(
              mix(h(i), h(i + vec2(1,0)), f.x),
              mix(h(i + vec2(0,1)), h(i + vec2(1,1)), f.x),
              f.y
            );
          }
          float fbm(vec2 p) {
            float v = 0.0;
            float a = 0.5;
            for (int i = 0; i < 4; i++) {
              v += a * n(p);
              p *= 2.1;
              a *= 0.55;
            }
            return v;
          }

          void main() {
            // Crest baseline at vUv.y ≈ 0.86 (near top of the plane),
            // gently undulating at multiple frequencies to soften the
            // stair-step aliasing that a single-frequency crest gave.
            float crest = 0.86
                        + 0.030 * n(vec2(vUv.x * 2.0, 0.0))
                        + 0.015 * n(vec2(vUv.x * 6.0 + 12.0, 0.0))
                        + 0.006 * n(vec2(vUv.x * 18.0 + 41.0, 0.0));
            // Soft AA around the crest: fade the top edge over ~2 pixels
            // instead of a hard alpha step.
            float crestAA = smoothstep(crest + 0.006, crest - 0.006, vUv.y);
            if (crestAA < 0.001) discard;

            // Snow field colour: brighter near the crest (sun-hit), cooler shadow below.
            float depth = smoothstep(crest, crest - 0.5, vUv.y);
            vec3 base = mix(uSnow, uSnowShadow, depth * 0.85);

            // Micro-texture so the field doesn't read as a flat grey slab.
            float micro = fbm(vec2(vUv.x * 12.0, vUv.y * 24.0));
            base = mix(base, base * vec3(0.94, 0.96, 1.02), micro * 0.18);

            // Sun-side warmth: gentle warm tint on the right where the sun sits.
            float warmSide = smoothstep(0.4, 1.0, vUv.x);
            base = mix(base, base * vec3(1.12, 1.02, 0.90), 0.20 * warmSide);

            // Fade the bottom of the plane into --polar-night so the
            // scene has no hard bottom seam against the page. The fade
            // spans the lower ~25% of the plane.
            float bottomFade = smoothstep(0.05, 0.35, vUv.y);
            vec3 color = mix(uPolarNight, base, bottomFade);

            gl_FragColor = vec4(color, crestAA);
          }
        `}
      />
    </mesh>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// Snow — instanced points drifting down. Alpha-blended, additive off.
function Snow({ count }: { count: number }) {
  const { viewport } = useThree()
  const positions = useMemo(() => {
    const a = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      a[i * 3 + 0] = (Math.random() - 0.5) * viewport.width * 1.2
      a[i * 3 + 1] = (Math.random() - 0.5) * viewport.height * 1.2
      a[i * 3 + 2] = (Math.random() - 0.5) * 0.4
    }
    return a
  }, [count, viewport.width, viewport.height])

  // Per-flake horizontal drift phase.
  const phases = useMemo(() => {
    const a = new Float32Array(count)
    for (let i = 0; i < count; i++) a[i] = Math.random() * Math.PI * 2
    return a
  }, [count])

  const geom = useRef<THREE.BufferGeometry>(null!)
  const mat  = useRef<THREE.ShaderMaterial>(null!)

  useFrame((_, dt) => {
    const arr = geom.current.attributes.position.array as Float32Array
    const halfH = viewport.height * 0.6
    const halfW = viewport.width * 0.6
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 1] -= dt * 0.15 * (0.6 + (i % 5) * 0.15)
      arr[i * 3 + 0] += Math.sin(mat.current.uniforms.uTime.value * 0.4 + phases[i]) * dt * 0.02
      if (arr[i * 3 + 1] < -halfH) {
        arr[i * 3 + 1] = halfH
        arr[i * 3 + 0] = (Math.random() - 0.5) * viewport.width * 1.2
      }
      if (arr[i * 3 + 0] < -halfW) arr[i * 3 + 0] = halfW
      if (arr[i * 3 + 0] >  halfW) arr[i * 3 + 0] = -halfW
    }
    geom.current.attributes.position.needsUpdate = true
    mat.current.uniforms.uTime.value += dt
  })

  return (
    <points>
      <bufferGeometry ref={geom}>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <shaderMaterial
        ref={mat}
        transparent
        depthWrite={false}
        uniforms={{ uTime: { value: 0 } }}
        vertexShader={/* glsl */`
          void main() {
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = clamp(3.0 + (position.z + 0.4) * 6.0, 1.5, 6.0);
            gl_Position  = projectionMatrix * mv;
          }
        `}
        fragmentShader={/* glsl */`
          precision highp float;
          void main() {
            vec2 c = gl_PointCoord - 0.5;
            float d = length(c);
            float a = smoothstep(0.5, 0.0, d) * 0.6;
            gl_FragColor = vec4(0.93, 0.96, 1.0, a);
          }
        `}
      />
    </points>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// Rise-in orchestrator — animates a normalized "rise" value 0→1 on mount.
function SceneContents({ prefersReducedMotion }: { prefersReducedMotion: boolean }) {
  const { viewport } = useThree()
  const { state } = usePipeline()
  const critic = state.result?.critic_scores
  const verifier = state.result?.verifier_scores
  const passed = critic ? (critic.pass ?? undefined) : undefined
  const verifierPassed = verifier ? (verifier.pass ?? undefined) : undefined
  const finalPass =
    passed === undefined && verifierPassed === undefined
      ? undefined
      : (passed !== false && verifierPassed !== false)
  const [rise, setRise] = useState(prefersReducedMotion ? 1 : 0)

  useEffect(() => {
    if (prefersReducedMotion) return
    const start = performance.now()
    const dur   = 1200
    let frame = 0
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur)
      const eased = 1 - Math.pow(1 - t, 3) // easeOutCubic
      setRise(eased)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [prefersReducedMotion])

  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768
  const snowCount = isMobile ? 60 : 140

  // Sol placement — everything else in this scene works in
  // pixel-scaled world units (ortho zoom=1), so Sol must scale with
  // the viewport too or he ends up ~1px tall.
  //   Desktop: to the right of centre, on the crest, sun-side.
  //   Mobile:  a hair right of centre, slightly smaller.
  const solScale = Math.min(viewport.width, viewport.height) * (isMobile ? 0.20 : 0.24)
  const crestY = -0.01 * viewport.height
  // Sequential load moment: sun rises first, then Sol pops up out of
  // the snow. Sol's rise starts at rise=0.55 (roughly 0.55·1.2s ≈ 660ms
  // into the sun rise) so the two beats feel intentional.
  const solRise = Math.max(0, (rise - 0.55) / 0.45)
  const solRiseEased = 1 - Math.pow(1 - solRise, 2)
  const solY = crestY + solScale * 0.10 - (1 - solRiseEased) * solScale * 0.9
  const solPos: [number, number, number] = [
    isMobile ? viewport.width * 0.06 : viewport.width * 0.14,
    solY,
    0.4,   // sits just in front of horizon (z=-1) and sun (z=-2)
  ]

  return (
    <>
      <Sky />
      <Horizon />
      <Sun rise={rise} />
      <Snow count={snowCount} />
      {/* IMPORTANT: scale x and y only. Uniform scale on Sol makes his
          Z-extent = ±solScale·bodyRadius (roughly ±30 world units at
          the 216-scale desktop viewport), which pushes his front faces
          past the camera's near plane and hollows him out to a
          cross-section ring. Z is left at 1 — Sol's model already
          has appropriate depth in local units. */}
      <group position={solPos} scale={[solScale, solScale, 1]}>
        <Sol
          prefersReducedMotion={prefersReducedMotion}
          phase={state.phase}
          passed={finalPass}
        />
      </group>
    </>
  )
}

// ────────────────────────────────────────────────────────────────────────────
export function HeroScene() {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    const m = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(m.matches)
    const listener = (e: MediaQueryListEvent) => setReduced(e.matches)
    m.addEventListener('change', listener)
    return () => m.removeEventListener('change', listener)
  }, [])

  const [ready, setReady] = useState(false)
  useEffect(() => {
    // Defer canvas mount by one frame so the text hero paints first.
    const id = requestAnimationFrame(() => setReady(true))
    return () => cancelAnimationFrame(id)
  }, [])

  if (!ready) return null

  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <Canvas
        orthographic
        camera={{ position: [0, 0, 500], zoom: 1, near: 0.1, far: 2000 }}
        dpr={isMobile ? [1, 1.5] : [1.5, 2]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ display: 'block', width: '100%', height: '100%' }}
      >
        <Suspense fallback={null}>
          <SceneContents prefersReducedMotion={reduced} />
        </Suspense>
      </Canvas>
    </div>
  )
}
