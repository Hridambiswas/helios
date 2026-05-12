import { Suspense, useRef, Component, type ReactNode } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Sparkles, Environment } from '@react-three/drei'
import * as THREE from 'three'

class CanvasErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }> {
  state = { error: false }
  static getDerivedStateFromError() { return { error: true } }
  render() { return this.state.error ? this.props.fallback : this.props.children }
}

function GlowOrb() {
  const outerRef = useRef<THREE.Mesh>(null)
  const midRef   = useRef<THREE.Mesh>(null)
  const innerRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    if (outerRef.current) {
      outerRef.current.rotation.y = t * 0.10
      outerRef.current.rotation.x = Math.sin(t * 0.07) * 0.18
    }
    if (midRef.current) {
      midRef.current.rotation.y = -t * 0.22
      midRef.current.rotation.z =  Math.cos(t * 0.09) * 0.14
    }
    if (innerRef.current) {
      innerRef.current.rotation.y = t * 0.40
    }
  })

  return (
    <group>
      {/* Outer faceted crystalline shell */}
      <mesh ref={outerRef}>
        <icosahedronGeometry args={[1.6, 1]} />
        <meshStandardMaterial
          color="#3b0764"
          emissive="#7c3aed"
          emissiveIntensity={0.9}
          metalness={0.98}
          roughness={0.04}
          transparent
          opacity={0.82}
          side={THREE.FrontSide}
        />
      </mesh>

      {/* Mid wireframe layer */}
      <mesh ref={midRef}>
        <icosahedronGeometry args={[1.25, 0]} />
        <meshStandardMaterial
          color="#8b5cf6"
          emissive="#a78bfa"
          emissiveIntensity={1.8}
          metalness={0.7}
          roughness={0.1}
          wireframe
          transparent
          opacity={0.7}
        />
      </mesh>

      {/* Bright inner glowing core */}
      <mesh ref={innerRef}>
        <sphereGeometry args={[0.55, 32, 32]} />
        <meshStandardMaterial
          color="#c4b5fd"
          emissive="#ffffff"
          emissiveIntensity={4.0}
          transparent
          opacity={0.95}
        />
      </mesh>

      {/* Outer glow halo (large low-opacity sphere) */}
      <mesh>
        <sphereGeometry args={[2.2, 16, 16]} />
        <meshStandardMaterial
          color="#7c3aed"
          emissive="#8b5cf6"
          emissiveIntensity={0.15}
          transparent
          opacity={0.04}
          side={THREE.BackSide}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Dramatic purple point lights */}
      <pointLight color="#8b5cf6" intensity={25} distance={14} decay={2} />
      <pointLight color="#a855f7" intensity={12} distance={9} position={[2.5, 1.5, 1]} decay={2} />
      <pointLight color="#6366f1" intensity={10} distance={8} position={[-2, -1.2, 0.5]} decay={2} />
      <pointLight color="#c026d3" intensity={8}  distance={7} position={[0, -2, 2]} decay={2} />

      {/* Floating sparkle particles */}
      <Sparkles
        count={180}
        scale={7}
        size={1.4}
        speed={0.25}
        noise={0.5}
        color="#c4b5fd"
        opacity={0.65}
      />
    </group>
  )
}

export function HeroScene() {
  return (
    <CanvasErrorBoundary fallback={<div />}>
      <Canvas
        camera={{ position: [0, 0, 5.5], fov: 52 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
        onCreated={({ gl }) => {
          gl.toneMapping        = THREE.ACESFilmicToneMapping
          gl.toneMappingExposure = 1.6
        }}
      >
        <Environment preset="night" />
        <ambientLight intensity={0.04} />

        <Suspense fallback={null}>
          <GlowOrb />
        </Suspense>
      </Canvas>
    </CanvasErrorBoundary>
  )
}
