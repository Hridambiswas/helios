import { Suspense, useRef, useMemo, Component, type ReactNode } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Environment } from '@react-three/drei'
import * as THREE from 'three'

class CanvasErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }> {
  state = { error: false }
  static getDerivedStateFromError() { return { error: true } }
  render() { return this.state.error ? this.props.fallback : this.props.children }
}

// Dark glossy material — looks like black obsidian shards
const DARK_MAT_PROPS = {
  color: new THREE.Color('#08080f'),
  metalness: 0.99,
  roughness: 0.02,
  envMapIntensity: 2.0,
}

function ShatteredCrystal() {
  const groupRef = useRef<THREE.Group>(null)
  const innerRef = useRef<THREE.Mesh>(null)
  const t = useRef(0)

  const darkMat = useMemo(() => new THREE.MeshStandardMaterial(DARK_MAT_PROPS), [])

  // Outer shard positions: [x, y, z, scale, rx, ry, rz]
  const shards = useMemo<[number, number, number, number, number, number, number][]>(() => [
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
  ], [])

  useFrame((_, delta) => {
    t.current += delta
    if (groupRef.current) {
      groupRef.current.rotation.y  = t.current * 0.08
      groupRef.current.rotation.x  = Math.sin(t.current * 0.05) * 0.10
      groupRef.current.position.y  = Math.sin(t.current * 0.40) * 0.08
    }
    if (innerRef.current) {
      innerRef.current.rotation.z = t.current * 0.25
    }
  })

  return (
    <group ref={groupRef}>
      {/* Central crystal body */}
      <mesh material={darkMat}>
        <icosahedronGeometry args={[1.1, 0]} />
      </mesh>

      {/* Outer shards */}
      {shards.map(([x, y, z, s, rx, ry, rz], i) => (
        <mesh key={i} position={[x, y, z]} rotation={[rx, ry, rz]} scale={s} material={darkMat}>
          <icosahedronGeometry args={[0.6, 0]} />
        </mesh>
      ))}

      {/* Purple inner core glow */}
      <mesh ref={innerRef}>
        <sphereGeometry args={[0.42, 20, 20]} />
        <meshStandardMaterial
          color="#9333ea"
          emissive="#8b5cf6"
          emissiveIntensity={4.5}
          transparent
          opacity={0.95}
        />
      </mesh>

      {/* Tiny bright spark at center */}
      <mesh>
        <sphereGeometry args={[0.14, 10, 10]} />
        <meshStandardMaterial
          color="#fff"
          emissive="#e9d5ff"
          emissiveIntensity={8}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Lighting */}
      <pointLight color="#8b5cf6" intensity={10} distance={7}  decay={2} />
      <pointLight color="#c026d3" intensity={5}  distance={5}  decay={2} position={[0.8, 0.8, 0.5]} />
      <pointLight color="#6366f1" intensity={4}  distance={4}  decay={2} position={[-0.8, -0.5, 0.5]} />
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
          gl.toneMapping        = THREE.ACESFilmicToneMapping
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
