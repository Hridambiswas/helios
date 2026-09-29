import { Suspense, useMemo, useRef, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Sol } from './mascot/Sol'

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
const SUN_ANCHOR = { x: 0.62, y: 0.05 }
const SUN_RADIUS = 0.36
const HORIZON_Y  = -0.15 // just below the sun so it "rises" into view

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
    uPolarNight: { value: new THREE.Color(COLORS.polarNight) },
    uFrost:      { value: new THREE.Color(COLORS.frost) },
    uWarm:       { value: new THREE.Color('#7A5A6E') },      // pre-sunset blush at the horizon
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
          uniform vec3  uPolarNight;
          uniform vec3  uFrost;
          uniform vec3  uWarm;
          uniform float uHorizon;
          varying vec2 vUv;
          void main() {
            // vUv.y = 0 at bottom, 1 at top; horizon sits around vUv.y ≈ 0.42
            float y = vUv.y;
            float horizonUv = 0.5 + uHorizon * 0.5; // convert [-1,+1] to [0,1]
            // Sky (above horizon): polarNight at top, frost mid, warm blush right above horizon.
            vec3 sky = mix(uFrost, uPolarNight, smoothstep(horizonUv, 1.0, y));
            sky = mix(uWarm, sky, smoothstep(horizonUv, horizonUv + 0.20, y));
            // Below horizon: darken with a slow drift (aurora-ish, very faint).
            float pulse = 0.5 + 0.5 * sin(uTime * 0.35 + vUv.x * 3.0);
            vec3 belowHorizon = mix(uPolarNight, uFrost * 0.6, y / horizonUv);
            belowHorizon += 0.02 * pulse * vec3(0.6, 0.7, 1.0);
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
            float core   = smoothstep(uRadius + 0.010, uRadius - 0.005, d);
            float rim    = smoothstep(uRadius + 0.045, uRadius + 0.005, d);
            float halo   = smoothstep(0.50, uRadius + 0.020, d);
            // Very slow shimmer on the rim (breathing).
            float breathe = 0.02 * sin(uTime * 1.2);
            vec3 color = uSun * core
                       + uCorona * (rim - core) * 0.8
                       + uCorona * halo * (0.12 + breathe);
            float alpha = core + (rim - core) * 0.8 + halo * (0.35 + breathe);
            if (alpha < 0.005) discard;
            gl_FragColor = vec4(color, alpha);
          }
        `}
      />
    </mesh>
  )
}

// ────────────────────────────────────────────────────────────────────────────
// Horizon — a soft snow crest across the lower third of the screen.
function Horizon() {
  const { viewport } = useThree()
  const mat = useRef<THREE.ShaderMaterial>(null!)
  useFrame((_, dt) => { mat.current.uniforms.uTime.value += dt })

  const uniforms = useMemo(() => ({
    uTime:       { value: 0 },
    uSnow:       { value: new THREE.Color(COLORS.snow) },
    uSnowShadow: { value: new THREE.Color(COLORS.snowShadow) },
  }), [])

  const y = HORIZON_Y * viewport.height * 0.5

  return (
    <mesh position={[0, y, -1]}>
      <planeGeometry args={[viewport.width * 1.1, viewport.height * 0.55, 1, 1]} />
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
          varying vec2 vUv;

          // hash + fBm for the crest silhouette
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

          void main() {
            // Crest baseline at vUv.y ≈ 0.62 (top of the plane), gently undulating.
            float crest = 0.62
                        + 0.06  * n(vec2(vUv.x * 2.0, 0.0))
                        + 0.025 * n(vec2(vUv.x * 6.0 + 12.0, 0.0));
            float aboveCrest = step(crest, vUv.y);
            if (aboveCrest > 0.5) discard;  // sky reads through

            // Snow field colour: brighter near the crest (sun-hit), cooler shadow below.
            float depth = smoothstep(crest, 0.0, vUv.y);
            vec3 color = mix(uSnow, uSnowShadow, depth);

            // Sun-side warmth: sun sits at x≈0.81, so blend a subtle warm tint on the right.
            float warmSide = smoothstep(0.4, 1.0, vUv.x);
            color = mix(color, color * vec3(1.15, 1.02, 0.90), 0.25 * warmSide);

            gl_FragColor = vec4(color, 0.92);
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
  const [rise, setRise] = useState(prefersReducedMotion ? 1 : 0)

  useEffect(() => {
    if (prefersReducedMotion) return
    const start = performance.now()
    const dur   = 1200
    let frame = 0
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur)
      // easeOutCubic
      const eased = 1 - Math.pow(1 - t, 3)
      setRise(eased)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [prefersReducedMotion])

  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768
  const snowCount = isMobile ? 60 : 140

  return (
    <>
      <Sky />
      <Horizon />
      <Sun rise={rise} />
      <Snow count={snowCount} />
      {/* Sol sits on the snow crest, sun-facing (positive X → warm rim). */}
      <group position={[
        (isMobile ? 0.0 : 0.35),
        HORIZON_Y * 2.5,
        0,
      ]} scale={isMobile ? 0.85 : 1}>
        <Sol prefersReducedMotion={prefersReducedMotion} />
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
        camera={{ position: [0, 0, 5], zoom: 1, near: 0.1, far: 100 }}
        dpr={isMobile ? [1, 1] : [1, 1.5]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ display: 'block' }}
      >
        <Suspense fallback={null}>
          <SceneContents prefersReducedMotion={reduced} />
        </Suspense>
      </Canvas>
    </div>
  )
}
