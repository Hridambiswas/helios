import { Suspense, useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'

const FIRE_COUNT = 600

// ── Fire breath particles ─────────────────────────────────────────────────────
function FireBreath() {
  const t = useRef(0)

  const { positions, velocities, lifetimes, ages, colors } = useMemo(() => {
    const positions  = new Float32Array(FIRE_COUNT * 3)
    const velocities = new Float32Array(FIRE_COUNT * 3)
    const lifetimes  = new Float32Array(FIRE_COUNT)
    const ages       = new Float32Array(FIRE_COUNT)
    const colors     = new Float32Array(FIRE_COUNT * 3)
    for (let i = 0; i < FIRE_COUNT; i++) {
      positions[i * 3]     = 0
      positions[i * 3 + 1] = 0
      positions[i * 3 + 2] = 0
      const angle = (Math.random() - 0.5) * 0.55
      const spread = (Math.random() - 0.5) * 0.55
      velocities[i * 3]     = Math.random() * 3.5 + 1.5  // shoot forward
      velocities[i * 3 + 1] = Math.sin(angle) * 0.8
      velocities[i * 3 + 2] = Math.sin(spread) * 0.8
      lifetimes[i] = Math.random() * 1.1 + 0.35
      ages[i]      = Math.random() * 2
      colors[i * 3]     = 1
      colors[i * 3 + 1] = 0.7
      colors[i * 3 + 2] = 0
    }
    return { positions, velocities, lifetimes, ages, colors }
  }, [])

  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    g.setAttribute('color',    new THREE.BufferAttribute(colors,    3))
    return g
  }, [positions, colors])

  useFrame((_, delta) => {
    t.current += delta
    const pos = geo.attributes.position.array as Float32Array
    const col = geo.attributes.color.array    as Float32Array
    for (let i = 0; i < FIRE_COUNT; i++) {
      ages[i] += delta
      if (ages[i] > lifetimes[i]) {
        ages[i]        = 0
        pos[i * 3]     = (Math.random() - 0.5) * 0.12
        pos[i * 3 + 1] = (Math.random() - 0.5) * 0.12
        pos[i * 3 + 2] = (Math.random() - 0.5) * 0.12
      } else {
        pos[i * 3]     += velocities[i * 3]     * delta
        pos[i * 3 + 1] += velocities[i * 3 + 1] * delta
        pos[i * 3 + 2] += velocities[i * 3 + 2] * delta
        const na = ages[i] / lifetimes[i]
        if (na < 0.2) {
          col[i * 3] = 1; col[i * 3 + 1] = 1 - na * 1.5; col[i * 3 + 2] = na < 0.1 ? 0.5 : 0
        } else if (na < 0.55) {
          col[i * 3] = 1; col[i * 3 + 1] = Math.max(0, 0.5 - (na - 0.2) * 1.4); col[i * 3 + 2] = 0
        } else {
          col[i * 3] = Math.max(0, 1 - (na - 0.55) * 2.2); col[i * 3 + 1] = 0; col[i * 3 + 2] = 0
        }
      }
    }
    geo.attributes.position.needsUpdate = true
    geo.attributes.color.needsUpdate    = true
  })

  return (
    <points geometry={geo}>
      <pointsMaterial
        size={0.09} vertexColors transparent opacity={0.92}
        blending={THREE.AdditiveBlending} depthWrite={false} sizeAttenuation
      />
    </points>
  )
}

// ── Real Hyper3D dragon model ─────────────────────────────────────────────────
function DragonModel() {
  const { scene } = useGLTF('/dragon.glb')
  const groupRef  = useRef<THREE.Group>(null)
  const t = useRef(0)

  useFrame((_, delta) => {
    t.current += delta
    if (!groupRef.current) return
    groupRef.current.position.y = Math.sin(t.current * 0.68) * 0.45
    groupRef.current.rotation.y = Math.sin(t.current * 0.25) * 0.22
    groupRef.current.rotation.z = Math.sin(t.current * 0.48) * 0.055
  })

  return (
    <group ref={groupRef}>
      {/* Dragon mesh */}
      <primitive object={scene} />

      {/* Fire breath at mouth (positive X end of model is typically the head) */}
      <group position={[2.6, 0.15, 0]}>
        <FireBreath />
        <pointLight color="#FF5500" intensity={5} distance={4.5} decay={2} />
      </group>

      {/* Eye glows */}
      <pointLight position={[2.3, 0.4, 0.2]}  color="#FF7700" intensity={1.2} distance={1.5} />
      <pointLight position={[2.3, 0.4, -0.2]} color="#FF7700" intensity={1.2} distance={1.5} />
    </group>
  )
}

// ── Floating embers ───────────────────────────────────────────────────────────
function Embers() {
  const ref = useRef<THREE.Points>(null)
  const count = 200

  const geo = useMemo(() => {
    const pos    = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 26
      pos[i * 3 + 1] = (Math.random() - 0.5) * 14
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10 - 3
      colors[i * 3]     = 0.8 + Math.random() * 0.2
      colors[i * 3 + 1] = Math.random() * 0.2
      colors[i * 3 + 2] = 0
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos,    3))
    g.setAttribute('color',    new THREE.BufferAttribute(colors, 3))
    return g
  }, [])

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.04) * 0.08
      ref.current.position.y = Math.sin(clock.getElapsedTime() * 0.16) * 0.18
    }
  })

  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial
        size={0.04} vertexColors transparent opacity={0.5}
        blending={THREE.AdditiveBlending} depthWrite={false}
      />
    </points>
  )
}

// ── Loading placeholder ───────────────────────────────────────────────────────
function Placeholder() {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = clock.getElapsedTime() * 0.8
      ref.current.rotation.x = Math.sin(clock.getElapsedTime() * 0.5) * 0.3
    }
  })
  return (
    <mesh ref={ref}>
      <octahedronGeometry args={[1.2, 1]} />
      <meshStandardMaterial color="#8B0020" wireframe emissive="#5B0010" emissiveIntensity={0.6} />
    </mesh>
  )
}

// ── Cinematic camera drift ────────────────────────────────────────────────────
function CameraRig() {
  useFrame(({ camera, clock }) => {
    const t = clock.getElapsedTime()
    camera.position.x = Math.sin(t * 0.16) * 1.1
    camera.position.y = Math.cos(t * 0.11) * 0.5 + 1.0
    camera.position.z = 10 + Math.sin(t * 0.08) * 0.8
    camera.lookAt(0, 0.3, 0)
  })
  return null
}

// ── Exported Canvas ───────────────────────────────────────────────────────────
export function DragonScene() {
  return (
    <Canvas
      camera={{ position: [0, 1.0, 10], fov: 48 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'transparent' }}
    >
      <color attach="background" args={['#000000']} />

      <ambientLight intensity={0.15} color="#200010" />
      <directionalLight position={[5, 8, 4]}  intensity={0.7} color="#FF3300" />
      <directionalLight position={[-4, 3, 2]} intensity={0.3} color="#FFD000" />
      <pointLight position={[0, 5, 5]} color="#FF2200" intensity={2} distance={16} />

      <Suspense fallback={<Placeholder />}>
        <DragonModel />
      </Suspense>

      <Embers />
      <CameraRig />

      <EffectComposer>
        <Bloom luminanceThreshold={0.25} luminanceSmoothing={0.9} intensity={2.5} mipmapBlur />
      </EffectComposer>
    </Canvas>
  )
}

useGLTF.preload('/dragon.glb')
