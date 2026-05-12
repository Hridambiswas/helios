import { Suspense, useRef, useMemo, Component, type ReactNode } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Environment } from '@react-three/drei'
import * as THREE from 'three'

class CanvasErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }> {
  state = { error: false }
  static getDerivedStateFromError() { return { error: true } }
  render() { return this.state.error ? this.props.fallback : this.props.children }
}

// Easing helpers
const easeOutCubic  = (t: number) => 1 - Math.pow(1 - t, 3)
const easeOutElastic = (t: number) => {
  if (t === 0 || t === 1) return t
  return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * (2 * Math.PI) / 3) + 1
}

const DARK_MAT_PROPS = {
  color:            new THREE.Color('#08080e'),
  metalness:        0.99,
  roughness:        0.02,
  envMapIntensity:  2.2,
}

// Each shard: [x, y, z, scale, rx, ry, rz]
const SHARDS: [number, number, number, number, number, number, number][] = [
  [ 1.55,  0.35,  0.10, 0.70,  0.8,  0.5,  0.2],
  [-1.40,  0.75,  0.20, 0.55,  0.3,  1.2,  0.4],
  [ 0.30,  1.80, -0.30, 0.60,  1.1,  0.2,  0.7],
  [-0.20, -1.65,  0.50, 0.50,  0.6,  0.9,  0.1],
  [ 1.10, -1.15,  0.65, 0.45,  0.4,  1.5,  0.3],
  [-1.50, -0.60, -0.25, 0.60,  0.7,  0.3,  1.1],
  [ 0.80,  0.70,  1.60, 0.50,  1.3,  0.6,  0.5],
  [-0.60,  1.10, -1.45, 0.55,  0.2,  1.1,  0.8],
  [ 0.55, -0.80, -1.70, 0.40,  0.9,  0.4,  1.2],
  [-1.20,  1.40,  0.80, 0.45,  1.6,  0.2,  0.6],
  [ 1.70, -0.20,  0.90, 0.35,  0.5,  1.8,  0.3],
  [-0.40, -1.20,  1.50, 0.40,  1.0,  0.7,  1.4],
]

function ShatteredCrystal() {
  const groupRef   = useRef<THREE.Group>(null)
  const innerRef   = useRef<THREE.Mesh>(null)
  const coreRef    = useRef<THREE.Mesh>(null)
  const shardRefs  = useRef<(THREE.Mesh | null)[]>(Array(SHARDS.length).fill(null))
  const intro      = useRef(0)   // 0→1 over ~2s — the "venomous eruption" phase
  const idle       = useRef(0)

  const darkMat = useMemo(() => new THREE.MeshStandardMaterial(DARK_MAT_PROPS), [])

  useFrame((_, delta) => {
    idle.current += delta

    // ── INTRO: shard eruption animation ──────────────────────────────────────
    if (intro.current < 1) {
      intro.current = Math.min(1, intro.current + delta / 2.1)

      shardRefs.current.forEach((mesh, i) => {
        if (!mesh) return
        const delay  = (i / SHARDS.length) * 0.55   // stagger 0–0.55s
        const p      = Math.max(0, (intro.current - delay) / (1 - delay * 0.7))
        const eased  = easeOutElastic(Math.min(1, p))

        // Start at origin, move to final position
        mesh.position.x = SHARDS[i][0] * eased
        mesh.position.y = SHARDS[i][1] * eased
        mesh.position.z = SHARDS[i][2] * eased
        mesh.scale.setScalar(SHARDS[i][3] * easeOutCubic(Math.min(1, p * 1.3)))
      })

      // Core pulses during intro
      if (coreRef.current) {
        const coreP = easeOutCubic(Math.min(1, intro.current * 1.8))
        coreRef.current.scale.setScalar(coreP * 1.2)
      }
      if (innerRef.current) {
        const innerP = easeOutCubic(Math.min(1, intro.current * 1.4))
        innerRef.current.scale.setScalar(innerP)
      }
    }

    // ── IDLE: slow floating rotation ─────────────────────────────────────────
    if (groupRef.current) {
      const t = idle.current
      groupRef.current.rotation.y = t * 0.07
      groupRef.current.rotation.x = Math.sin(t * 0.04) * 0.09
      groupRef.current.position.y = Math.sin(t * 0.38) * 0.07
    }

    // Inner core pulses
    if (innerRef.current && intro.current >= 1) {
      innerRef.current.rotation.z = idle.current * 0.28
      const pulse = 1 + Math.sin(idle.current * 1.6) * 0.08
      innerRef.current.scale.setScalar(pulse)
    }
  })

  return (
    <group ref={groupRef}>
      {/* Central crystal body */}
      <mesh material={darkMat}>
        <icosahedronGeometry args={[1.1, 0]} />
      </mesh>

      {/* Outer shards — erupted outward on load */}
      {SHARDS.map(([, , , , rx, ry, rz], i) => (
        <mesh
          key={i}
          ref={el => { shardRefs.current[i] = el }}
          rotation={[rx, ry, rz]}
          position={[0, 0, 0]}
          scale={0}
          material={darkMat}
        >
          <icosahedronGeometry args={[0.6, 0]} />
        </mesh>
      ))}

      {/* Purple inner core — grows first */}
      <mesh ref={innerRef} scale={0}>
        <sphereGeometry args={[0.42, 20, 20]} />
        <meshStandardMaterial
          color="#9333ea"
          emissive="#8b5cf6"
          emissiveIntensity={5.0}
          transparent
          opacity={0.95}
        />
      </mesh>

      {/* Tiny bright spark */}
      <mesh ref={coreRef} scale={0}>
        <sphereGeometry args={[0.13, 10, 10]} />
        <meshStandardMaterial
          color="#fff"
          emissive="#e9d5ff"
          emissiveIntensity={10}
          transparent
          opacity={0.9}
        />
      </mesh>

      <pointLight color="#8b5cf6" intensity={12} distance={7} decay={2} />
      <pointLight color="#c026d3" intensity={6}  distance={5} decay={2} position={[0.8, 0.8, 0.5]} />
      <pointLight color="#6366f1" intensity={5}  distance={4} decay={2} position={[-0.8, -0.5, 0.5]} />
    </group>
  )
}

export function HeroScene() {
  return (
    <CanvasErrorBoundary fallback={<div />}>
      <Canvas
        camera={{ position: [0, 0, 5.5], fov: 48 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
        onCreated={({ gl }) => {
          gl.toneMapping         = THREE.ACESFilmicToneMapping
          gl.toneMappingExposure = 1.8
        }}
      >
        <Environment preset="night" />
        <ambientLight intensity={0.03} />
        <Suspense fallback={null}>
          <ShatteredCrystal />
        </Suspense>
      </Canvas>
    </CanvasErrorBoundary>
  )
}
