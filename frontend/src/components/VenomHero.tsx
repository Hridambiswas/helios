import { useRef, useMemo, useEffect, useCallback, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { motion, useMotionValue, useSpring, AnimatePresence } from 'framer-motion'

// ─── GLSL ────────────────────────────────────────────────────────────────────

const VERT = /* glsl */ `
precision highp float;
attribute vec3 position;
attribute vec2 uv;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;

uniform float uTime;
uniform vec2  uMouse;
uniform float uScroll;
uniform vec2  uResolution;

// ── Simplex noise (Ian McEwan) ──────────────────────────────────────────────
vec3 mod289(vec3 x) { return x - floor(x*(1./289.))*289.; }
vec2 mod289(vec2 x) { return x - floor(x*(1./289.))*289.; }
vec3 permute(vec3 x) { return mod289(((x*34.)+1.)*x); }

float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.,0.) : vec2(0.,1.);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute(permute(i.y+vec3(0.,i1.y,1.)) + i.x+vec3(0.,i1.x,1.));
  vec3 m = max(0.5-vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);
  m = m*m; m = m*m;
  vec3 x = 2.*fract(p*C.www)-1.;
  vec3 h = abs(x)-0.5;
  vec3 ox = floor(x+0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314*(a0*a0+h*h);
  vec3 g;
  g.x  = a0.x*x0.x  + h.x*x0.y;
  g.yz = a0.yz*x12.xz + h.yz*x12.yw;
  return 130.*dot(m,g);
}

// ── Rotation matrix ──────────────────────────────────────────────────────────
mat2 rot(float a) { float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

// ── FBM ─────────────────────────────────────────────────────────────────────
float fbm(vec2 p) {
  float v=0., a=0.5;
  mat2 R = rot(0.37);
  for(int i=0;i<5;i++){
    v += a * snoise(p);
    p  = R * p * 2.03;
    a *= 0.49;
  }
  return v;
}

void main() {
  vec2 uv = vUv;
  float ar = uResolution.x / uResolution.y;
  vec2 p = (uv - 0.5) * vec2(ar, 1.0);

  float t = uTime * 0.18;

  // ── Mouse repel field ─────────────────────────────────────────────────────
  vec2 mouse = (uMouse - 0.5) * vec2(ar, 1.0);
  vec2 toMouse = p - mouse;
  float md = length(toMouse);
  float repel = 0.22 * exp(-md * 3.8);
  vec2 repelDir = normalize(toMouse + vec2(0.001)) * repel;

  // ── Domain warp (2-pass) ──────────────────────────────────────────────────
  vec2 q = vec2(fbm(p + t * vec2(0.9,-0.5)),
                fbm(p + t * vec2(-0.7, 0.8)));

  vec2 r = vec2(fbm(p + 2.1*q + t*vec2(1.2,-0.6) + repelDir),
                fbm(p + 2.1*q + t*vec2(-0.8, 1.1) + repelDir));

  float n = fbm(p + 2.7*r + repelDir);

  // ── Scroll consumption threshold ─────────────────────────────────────────
  float threshold = mix(-0.52, 1.05, uScroll);
  float edgeW = 0.038 + 0.018 * snoise(p * 4.2 + t * 1.6);
  float sdf = n - threshold + p.y * 0.25;
  float fluidMask = 1.0 - smoothstep(-edgeW, edgeW, sdf);

  // ── Metaball cursor ───────────────────────────────────────────────────────
  float ball = clamp(0.14 / (md*md + 0.012), 0., 0.40);
  fluidMask = clamp(fluidMask + ball, 0., 1.);

  // ── Specular via finite difference normals ────────────────────────────────
  float eps = 0.003;
  float nx = fbm(p + vec2(eps,0.) + 2.7*r) - fbm(p - vec2(eps,0.) + 2.7*r);
  float ny = fbm(p + vec2(0.,eps) + 2.7*r) - fbm(p - vec2(0.,eps) + 2.7*r);
  vec3  norm = normalize(vec3(nx, ny, eps*18.));
  vec3  light = normalize(vec3(-0.6, 0.8, 1.0));
  float spec = pow(max(dot(norm, light), 0.), 32.) * 0.35;

  // ── Oil-slick iridescence ─────────────────────────────────────────────────
  float irid = snoise(p * 6.0 + t * 2.4) * 0.5 + 0.5;
  vec3  iriCol = 0.5 + 0.5 * cos(vec3(0.,2.094,4.189) + irid * 3.8 + 1.2);
  iriCol *= vec3(0.5, 0.22, 0.9); // violet tint

  // ── Base fluid colour ─────────────────────────────────────────────────────
  vec3 fluidCol = mix(vec3(0.024, 0.008, 0.055), vec3(0.08, 0.04, 0.18), n * 0.5 + 0.5);
  fluidCol += iriCol * 0.22 * fluidMask;
  fluidCol += spec * vec3(0.7, 0.5, 1.0);

  // Edge shimmer line
  float edgeGlow = smoothstep(0.0, edgeW*2., fluidMask) * (1. - smoothstep(edgeW*2., edgeW*5., fluidMask));
  fluidCol += edgeGlow * vec3(0.55, 0.28, 1.0) * 0.9;

  // ── Background ────────────────────────────────────────────────────────────
  vec3 bgCol = vec3(0.0, 0.0, 0.0);

  vec3 col = mix(bgCol, fluidCol, fluidMask);

  // ── Vignette ──────────────────────────────────────────────────────────────
  float vign = 1. - 0.52 * dot(uv - 0.5, uv - 0.5) * 3.4;
  col *= vign;

  gl_FragColor = vec4(col, 1.0);
}
`

// ─── R3F fluid mesh ──────────────────────────────────────────────────────────

interface FluidMeshProps {
  mouseRef: React.MutableRefObject<{ x: number; y: number }>
  scrollRef: React.MutableRefObject<number>
}

function FluidMesh({ mouseRef, scrollRef }: FluidMeshProps) {
  const uniforms = useMemo(() => ({
    uTime:       { value: 0.0 },
    uMouse:      { value: new THREE.Vector2(0.5, 0.5) },
    uScroll:     { value: 0.0 },
    uResolution: { value: new THREE.Vector2(1, 1) },
  }), [])

  useFrame(({ clock, size }) => {
    uniforms.uTime.value       = clock.getElapsedTime()
    uniforms.uMouse.value.set(mouseRef.current.x, mouseRef.current.y)
    uniforms.uScroll.value     = scrollRef.current
    uniforms.uResolution.value.set(size.width, size.height)
  })

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <rawShaderMaterial
        vertexShader={VERT}
        fragmentShader={FRAG}
        uniforms={uniforms}
        depthTest={false}
        depthWrite={false}
      />
    </mesh>
  )
}

// ─── Film grain canvas ───────────────────────────────────────────────────────

function FilmGrain() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef  = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    let raf = 0
    let tick = 0

    const resize = () => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize, { passive: true })

    const draw = () => {
      raf = requestAnimationFrame(draw)
      tick++
      if (tick % 2 !== 0) return  // 30 fps
      const { width, height } = canvas
      const img = ctx.createImageData(width, height)
      const d   = img.data
      for (let i = 0; i < d.length; i += 4) {
        const v = (Math.random() * 255) | 0
        d[i] = d[i+1] = d[i+2] = v
        d[i+3] = 14
      }
      ctx.putImageData(img, 0, 0)
    }
    draw()
    frameRef.current = raf

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute', inset: 0,
        width: '100%', height: '100%',
        pointerEvents: 'none',
        mixBlendMode: 'overlay',
        opacity: 0.55,
        zIndex: 2,
      }}
    />
  )
}

// ─── Venom cursor ────────────────────────────────────────────────────────────

function VenomCursor() {
  const mx = useMotionValue(-300)
  const my = useMotionValue(-300)

  const blobX  = useSpring(mx, { damping: 15, stiffness: 90 })
  const blobY  = useSpring(my, { damping: 15, stiffness: 90 })
  const ringX  = useSpring(mx, { damping: 30, stiffness: 600 })
  const ringY  = useSpring(my, { damping: 30, stiffness: 600 })
  const dotX   = useSpring(mx, { damping: 40, stiffness: 900 })
  const dotY   = useSpring(my, { damping: 40, stiffness: 900 })

  useEffect(() => {
    const move = (e: MouseEvent) => { mx.set(e.clientX); my.set(e.clientY) }
    window.addEventListener('mousemove', move)
    return () => window.removeEventListener('mousemove', move)
  }, [mx, my])

  return (
    <>
      <motion.div style={{
        position: 'fixed', x: blobX, y: blobY,
        translateX: '-50%', translateY: '-50%',
        width: 52, height: 52, borderRadius: '50%',
        background: 'rgba(255,255,255,0.07)',
        pointerEvents: 'none', zIndex: 9998,
        mixBlendMode: 'difference',
      }} />
      <motion.div style={{
        position: 'fixed', x: ringX, y: ringY,
        translateX: '-50%', translateY: '-50%',
        width: 28, height: 28, borderRadius: '50%',
        border: '1px solid rgba(255,255,255,0.55)',
        pointerEvents: 'none', zIndex: 9999,
        mixBlendMode: 'difference',
      }} />
      <motion.div style={{
        position: 'fixed', x: dotX, y: dotY,
        translateX: '-50%', translateY: '-50%',
        width: 4, height: 4, borderRadius: '50%',
        background: '#fff',
        pointerEvents: 'none', zIndex: 9999,
        mixBlendMode: 'difference',
      }} />
    </>
  )
}

// ─── Magnetic button wrapper ─────────────────────────────────────────────────

function Magnetic({ children, strength = 0.3 }: { children: React.ReactNode; strength?: number }) {
  const ref  = useRef<HTMLDivElement>(null)
  const rawX = useMotionValue(0)
  const rawY = useMotionValue(0)
  const x    = useSpring(rawX, { damping: 12, stiffness: 160 })
  const y    = useSpring(rawY, { damping: 12, stiffness: 160 })

  const onMove = useCallback((e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect()
    rawX.set((e.clientX - (r.left + r.width  / 2)) * strength)
    rawY.set((e.clientY - (r.top  + r.height / 2)) * strength)
  }, [rawX, rawY, strength])

  const onLeave = useCallback(() => { rawX.set(0); rawY.set(0) }, [rawX, rawY])

  return (
    <motion.div ref={ref} style={{ x, y, display: 'inline-block' }}
      onMouseMove={onMove} onMouseLeave={onLeave}>
      {children}
    </motion.div>
  )
}

// ─── Main VenomHero ──────────────────────────────────────────────────────────

interface VenomHeroProps {
  onQuerySubmit: (q: string) => void
  onAuthClick:   () => void
  isLoggedIn:    boolean
}

const CHIPS = ['What is RAG?', 'Summarise this PDF', 'Multi-agent pipeline', 'Semantic search']

const PLACEHOLDER_PHRASES = [
  'Ask anything about your data...',
  'Enter what you want to know...',
  'Query your documents...',
  'What insights are you looking for...',
  'Explore your knowledge base...',
  'Summarise any file or topic...',
]

function useTypewriterPlaceholder() {
  const [text, setText]       = useState('')
  const [phraseIdx, setPhraseIdx] = useState(0)
  const [phase, setPhase]     = useState<'typing' | 'deleting'>('typing')

  useEffect(() => {
    const phrase = PLACEHOLDER_PHRASES[phraseIdx]
    if (phase === 'typing') {
      if (text.length < phrase.length) {
        const t = setTimeout(() => setText(phrase.slice(0, text.length + 1)), 52)
        return () => clearTimeout(t)
      }
      const t = setTimeout(() => setPhase('deleting'), 3400)
      return () => clearTimeout(t)
    }
    if (text.length > 0) {
      const t = setTimeout(() => setText(text.slice(0, -1)), 28)
      return () => clearTimeout(t)
    }
    setPhraseIdx(i => (i + 1) % PLACEHOLDER_PHRASES.length)
    setPhase('typing')
  }, [text, phase, phraseIdx])

  return text
}

export function VenomHero({ onQuerySubmit, onAuthClick, isLoggedIn }: VenomHeroProps) {
  const [query, setQuery]       = useState('')
  const [focused, setFocused]   = useState(false)
  const [hovered, setHovered]   = useState(false)
  const animatedPlaceholder = useTypewriterPlaceholder()
  const mouseRef  = useRef({ x: 0.5, y: 0.5 })
  const scrollRef = useRef(0)
  const sectionRef = useRef<HTMLDivElement>(null)

  // Track mouse in normalized [0,1] coords for shader
  useEffect(() => {
    const move = (e: MouseEvent) => {
      mouseRef.current = {
        x: e.clientX / window.innerWidth,
        y: 1 - e.clientY / window.innerHeight,
      }
    }
    window.addEventListener('mousemove', move, { passive: true })
    return () => window.removeEventListener('mousemove', move)
  }, [])

  // Track scroll progress for consumption transition
  useEffect(() => {
    const el = sectionRef.current
    if (!el) return
    const update = () => {
      const rect = el.getBoundingClientRect()
      const total = el.offsetHeight - window.innerHeight
      if (total <= 0) { scrollRef.current = 0; return }
      scrollRef.current = Math.max(0, Math.min(1, -rect.top / total))
    }
    window.addEventListener('scroll', update, { passive: true })
    update()
    return () => window.removeEventListener('scroll', update)
  }, [])

  const submit = (q: string) => {
    const trimmed = q.trim()
    if (!trimmed) return
    if (!isLoggedIn) { onAuthClick(); return }
    onQuerySubmit(trimmed)
    setQuery('')
  }

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') submit(query)
  }

  return (
    <>
      <VenomCursor />

      <div ref={sectionRef} style={{ position: 'relative', minHeight: '200vh' }}>
        {/* Sticky WebGL viewport */}
        <div style={{ position: 'sticky', top: 0, width: '100%', height: '100vh', zIndex: 0 }}>

          {/* WebGL fluid background */}
          <div style={{ position: 'absolute', inset: 0 }}>
            <Canvas
              camera={{ position: [0, 0, 1] }}
              gl={{ antialias: false, alpha: false }}
              style={{ display: 'block', width: '100%', height: '100%' }}
            >
              <FluidMesh mouseRef={mouseRef} scrollRef={scrollRef} />
            </Canvas>
            <FilmGrain />
          </div>

          {/* UI overlay */}
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            zIndex: 10, padding: '0 1.5rem',
          }}>

            {/* Eyebrow */}
            <motion.p
              initial={{ opacity: 0, y: -16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
              style={{
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: 10,
                letterSpacing: '0.35em',
                textTransform: 'uppercase',
                color: 'rgba(139,92,246,0.75)',
                marginBottom: 28,
              }}
            >
              Distributed Multi-Agent GenAI
            </motion.p>

            {/* Hero heading */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.1, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
              style={{
                fontFamily: '"Montserrat", sans-serif',
                fontWeight: 900,
                fontSize: 'clamp(72px, 16vw, 200px)',
                lineHeight: 0.88,
                letterSpacing: '-0.04em',
                textAlign: 'center',
                color: '#fff',
                textShadow: '0 0 80px rgba(139,92,246,0.35), 0 0 160px rgba(139,92,246,0.15)',
                marginBottom: 48,
                userSelect: 'none',
              }}
            >
              HELIOS
            </motion.h1>

            {/* Query input */}
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.0, delay: 0.55, ease: [0.16, 1, 0.3, 1] }}
              style={{ width: '100%', maxWidth: 640, position: 'relative' }}
            >
              <div style={{
                display: 'flex',
                background: 'rgba(0,0,0,0.55)',
                backdropFilter: 'blur(24px)',
                border: `1px solid ${focused ? 'rgba(139,92,246,0.55)' : 'rgba(255,255,255,0.10)'}`,
                transition: 'border-color 0.3s',
                boxShadow: focused ? '0 0 0 3px rgba(139,92,246,0.08), 0 0 40px rgba(139,92,246,0.12)' : 'none',
              }}>
                <div style={{ position: 'relative', flex: 1 }}>
                  <input
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={handleKey}
                    onFocus={() => setFocused(true)}
                    onBlur={() => setFocused(false)}
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="off"
                    spellCheck={false}
                    data-1p-ignore=""
                    data-lpignore="true"
                    data-form-type="other"
                    placeholder=""
                    style={{
                      width: '100%',
                      background: 'transparent',
                      border: 'none',
                      outline: 'none',
                      padding: '16px 20px',
                      fontFamily: '"IBM Plex Mono", monospace',
                      fontSize: 13,
                      color: '#fff',
                      cursor: 'none',
                    }}
                  />
                  {query === '' && (
                    <span
                      aria-hidden
                      style={{
                        position: 'absolute',
                        left: 20,
                        top: '50%',
                        transform: 'translateY(-50%)',
                        fontFamily: '"IBM Plex Mono", monospace',
                        fontSize: 13,
                        color: 'rgba(255,255,255,0.28)',
                        pointerEvents: 'none',
                        userSelect: 'none',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                      }}
                    >
                      {animatedPlaceholder}
                      <span style={{
                        display: 'inline-block',
                        width: 1,
                        height: '1em',
                        background: 'rgba(255,255,255,0.5)',
                        marginLeft: 2,
                        verticalAlign: 'text-bottom',
                        animation: 'blink 1s step-end infinite',
                      }} />
                    </span>
                  )}
                </div>
                <Magnetic strength={0.25}>
                  <button
                    onClick={() => submit(query)}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                    style={{
                      padding: '0 24px',
                      background: hovered ? 'rgba(139,92,246,0.9)' : 'rgba(139,92,246,0.7)',
                      border: 'none',
                      cursor: 'none',
                      fontFamily: '"IBM Plex Mono", monospace',
                      fontSize: 10,
                      letterSpacing: '0.25em',
                      textTransform: 'uppercase',
                      color: '#fff',
                      transition: 'background 0.2s',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    RUN
                  </button>
                </Magnetic>
              </div>
            </motion.div>

            {/* Suggestion chips */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.85 }}
              style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 18, justifyContent: 'center' }}
            >
              {CHIPS.map(chip => (
                <button
                  key={chip}
                  onClick={() => submit(chip)}
                  style={{
                    fontFamily: '"IBM Plex Mono", monospace',
                    fontSize: 10,
                    letterSpacing: '0.12em',
                    padding: '6px 14px',
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.10)',
                    color: 'rgba(255,255,255,0.38)',
                    cursor: 'none',
                    transition: 'color 0.2s, border-color 0.2s',
                  }}
                  onMouseEnter={e => {
                    const b = e.currentTarget as HTMLButtonElement
                    b.style.color = 'rgba(255,255,255,0.80)'
                    b.style.borderColor = 'rgba(139,92,246,0.45)'
                  }}
                  onMouseLeave={e => {
                    const b = e.currentTarget as HTMLButtonElement
                    b.style.color = 'rgba(255,255,255,0.38)'
                    b.style.borderColor = 'rgba(255,255,255,0.10)'
                  }}
                >
                  {chip}
                </button>
              ))}
            </motion.div>

            {/* Scroll indicator */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 1.3 }}
              style={{
                position: 'absolute', bottom: 36,
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
              }}
            >
              <span style={{
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: 9, letterSpacing: '0.35em',
                color: 'rgba(255,255,255,0.18)',
                textTransform: 'uppercase',
              }}>Scroll</span>
              <motion.div
                animate={{ y: [0, 7, 0] }}
                transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
                style={{ width: 1, height: 36, background: 'linear-gradient(to bottom, rgba(139,92,246,0.5), transparent)' }}
              />
            </motion.div>
          </div>
        </div>

        {/* Scroll spacer (makes sticky section scrollable for consumption effect) */}
        <div style={{ height: '100vh' }} />
      </div>
    </>
  )
}
