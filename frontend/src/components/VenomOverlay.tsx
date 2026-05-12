import { Component, Suspense, useEffect, useRef, useMemo } from 'react'
import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// ─── Shaders ─────────────────────────────────────────────────────────────────

const VERT = /* glsl */`
precision highp float;
attribute vec3 position;
attribute vec3 normal;
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat3 normalMatrix;
uniform float uTime;
varying vec3 vN;
varying vec3 vPV;

float h(vec3 p){
  p = fract(p * vec3(0.1031,0.1030,0.0973));
  p += dot(p,p.yzx+33.33);
  return fract((p.x+p.y)*p.z);
}
float n3(vec3 p){
  vec3 i=floor(p), f=fract(p);
  f=f*f*(3.0-2.0*f);
  return mix(
    mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),
    mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z
  );
}
float fbm(vec3 p){
  float v=0.0,a=0.5;
  for(int i=0;i<5;i++){v+=a*n3(p);p*=2.1;a*=0.5;}
  return v;
}

void main(){
  float pulse = 0.08 * sin(uTime * 1.3);
  float noise  = fbm(position * 2.2 + uTime * 0.28);
  vec3  disp   = position + normal * (noise * 0.42 + pulse);
  vN  = normalize(normalMatrix * normal);
  vPV = (modelViewMatrix * vec4(disp, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(disp, 1.0);
}
`

const FRAG = /* glsl */`
precision highp float;
varying vec3 vN;
varying vec3 vPV;
uniform float uTime;

void main(){
  vec3 n = normalize(vN);
  vec3 vdir = normalize(-vPV);

  vec3 ldir = normalize(vec3(cos(uTime*0.4)*2.0, sin(uTime*0.3)*1.5+1.0, 3.0));
  vec3 hdir = normalize(ldir + vdir);

  float diff = max(dot(n, ldir), 0.0) * 0.15;
  float spec = pow(max(dot(n, hdir), 0.0), 90.0) * 2.2;
  float fr   = pow(1.0 - max(dot(n, vdir), 0.0), 3.8);

  vec3 col = vec3(0.012, 0.006, 0.025) * (diff + 0.08);
  col += vec3(0.75, 0.80, 0.85) * spec;
  col += fr * vec3(0.12, 0.08, 0.22) * 0.9;

  gl_FragColor = vec4(col, 1.0);
}
`

// ─── R3F mesh ─────────────────────────────────────────────────────────────────

function VenomBlob() {
  const meshRef = useRef<THREE.Mesh>(null)
  const uniforms = useMemo(() => ({ uTime: { value: 0 } }), [])

  useFrame(({ clock }) => {
    uniforms.uTime.value = clock.getElapsedTime()
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.getElapsedTime() * 0.11
      meshRef.current.rotation.x = Math.sin(clock.getElapsedTime() * 0.07) * 0.12
    }
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1.6, 96, 96]} />
      <rawShaderMaterial
        vertexShader={VERT}
        fragmentShader={FRAG}
        uniforms={uniforms}
        side={THREE.FrontSide}
      />
    </mesh>
  )
}

// ─── Error boundary so a WebGL crash never blacks out the whole app ───────────

type BoundaryState = { failed: boolean }

class WebGLBoundary extends Component<{ children: ReactNode }, BoundaryState> {
  state: BoundaryState = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    if (this.state.failed) return null   // silently fall back; CSS text still shows
    return this.props.children
  }
}

// ─── Framer Motion variants ───────────────────────────────────────────────────

const variants = {
  initial: {
    opacity: 0,
    clipPath: 'circle(0% at 50% 50%)',
  },
  animate: {
    opacity: 1,
    clipPath: 'circle(150% at 50% 50%)',
    transition: { duration: 0.9, ease: [0.16, 1, 0.3, 1] },
  },
  exit: {
    clipPath: 'circle(0% at 50% 50%)',
    opacity: 0,
    transition: { duration: 0.85, ease: [0.76, 0, 0.24, 1] },
  },
}

// ─── Main export ──────────────────────────────────────────────────────────────

export function VenomOverlay() {
  const barRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = barRef.current
    if (!el) return
    el.style.transition = 'transform 3.2s linear'
    requestAnimationFrame(() => { el.style.transform = 'scaleX(1)' })
  }, [])

  return (
    <motion.div
      key="venom-overlay"
      variants={variants}
      initial="initial"
      animate="animate"
      exit="exit"
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: '#000',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}
    >
      {/* 3-D organic blob — wrapped in boundaries so any crash degrades gracefully */}
      <div style={{ position: 'absolute', inset: 0 }}>
        <WebGLBoundary>
          <Suspense fallback={null}>
            <Canvas
              camera={{ position: [0, 0, 4.2], fov: 44 }}
              gl={{ antialias: true, alpha: false }}
              style={{ width: '100%', height: '100%', display: 'block' }}
            >
              <VenomBlob />
            </Canvas>
          </Suspense>
        </WebGLBoundary>
      </div>

      {/* Text + progress bar always on top of the 3-D canvas */}
      <div
        style={{
          position: 'relative', zIndex: 2,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', gap: 28,
          pointerEvents: 'none',
        }}
      >
        <h1
          style={{
            fontFamily: '"Inter Tight", "Montserrat", sans-serif',
            fontWeight: 900,
            fontSize: 'clamp(72px, 16vw, 200px)',
            letterSpacing: '-0.045em',
            color: '#fff',
            lineHeight: 1,
            userSelect: 'none',
          }}
        >
          HELIOS
        </h1>

        {/* Thin progress bar */}
        <div
          style={{
            width: 80, height: 1,
            background: 'rgba(255,255,255,0.12)',
            overflow: 'hidden',
          }}
        >
          <div
            ref={barRef}
            style={{
              width: '100%', height: '100%',
              background: 'rgba(255,255,255,0.7)',
              transform: 'scaleX(0)',
              transformOrigin: 'left',
            }}
          />
        </div>

        <p
          style={{
            fontFamily: '"IBM Plex Mono", monospace',
            fontSize: 9, letterSpacing: '0.5em',
            textTransform: 'uppercase',
            color: 'rgba(255,255,255,0.35)',
          }}
        >
          Initializing
        </p>
      </div>
    </motion.div>
  )
}
