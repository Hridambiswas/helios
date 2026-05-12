import { Suspense, useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'

const FIRE_COUNT = 900   // more fire

// ── Fire breath ───────────────────────────────────────────────────────────────
function FireBreath() {
  const t = useRef(0)

  const { positions, velocities, lifetimes, ages, colors } = useMemo(() => {
    const positions  = new Float32Array(FIRE_COUNT * 3)
    const velocities = new Float32Array(FIRE_COUNT * 3)
    const lifetimes  = new Float32Array(FIRE_COUNT)
    const ages       = new Float32Array(FIRE_COUNT)
    const colors     = new Float32Array(FIRE_COUNT * 3)
    for (let i = 0; i < FIRE_COUNT; i++) {
      const angle  = (Math.random() - 0.5) * 0.7
      const spread = (Math.random() - 0.5) * 0.7
      positions[i * 3]     = (Math.random() - 0.5) * 0.2
      positions[i * 3 + 1] = (Math.random() - 0.5) * 0.2
      positions[i * 3 + 2] = (Math.random() - 0.5) * 0.2
      velocities[i * 3]     = Math.random() * 4.5 + 1.5
      velocities[i * 3 + 1] = Math.sin(angle) * 1.2
      velocities[i * 3 + 2] = Math.sin(spread) * 1.2
      lifetimes[i] = Math.random() * 1.4 + 0.4
      ages[i]      = Math.random() * 2
      colors[i * 3] = 1; colors[i * 3 + 1] = 0.75; colors[i * 3 + 2] = 0
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
        pos[i * 3]     = (Math.random() - 0.5) * 0.2
        pos[i * 3 + 1] = (Math.random() - 0.5) * 0.2
        pos[i * 3 + 2] = (Math.random() - 0.5) * 0.2
      } else {
        pos[i * 3]     += velocities[i * 3]     * delta
        pos[i * 3 + 1] += velocities[i * 3 + 1] * delta
        pos[i * 3 + 2] += velocities[i * 3 + 2] * delta
        const na = ages[i] / lifetimes[i]
        if (na < 0.15) {
          // Hot white-yellow core
          col[i * 3] = 1; col[i * 3 + 1] = 1; col[i * 3 + 2] = 0.6 - na * 3
        } else if (na < 0.45) {
          // Gold/orange
          col[i * 3] = 1; col[i * 3 + 1] = Math.max(0, 0.78 - (na - 0.15) * 2); col[i * 3 + 2] = 0
        } else if (na < 0.75) {
          // Deep orange to red
          col[i * 3] = 1; col[i * 3 + 1] = Math.max(0, 0.2 - (na - 0.45)); col[i * 3 + 2] = 0
        } else {
          // Fade to dark red
          col[i * 3] = Math.max(0, 1 - (na - 0.75) * 3.5); col[i * 3 + 1] = 0; col[i * 3 + 2] = 0
        }
      }
    }
    geo.attributes.position.needsUpdate = true
    geo.attributes.color.needsUpdate    = true
  })

  return (
    <points geometry={geo}>
      <pointsMaterial size={0.12} vertexColors transparent opacity={0.9}
        blending={THREE.AdditiveBlending} depthWrite={false} sizeAttenuation />
    </points>
  )
}

// ── Hyper3D dragon model ──────────────────────────────────────────────────────
function DragonModel() {
  const { scene } = useGLTF('/dragon.glb')
  const groupRef  = useRef<THREE.Group>(null)
  const t = useRef(0)

  useFrame((_, delta) => {
    t.current += delta
    if (!groupRef.current) return
    groupRef.current.position.y = Math.sin(t.current * 0.65) * 0.6   // bigger soar
    groupRef.current.rotation.y = Math.sin(t.current * 0.22) * 0.28
    groupRef.current.rotation.z = Math.sin(t.current * 0.44) * 0.06
  })

  return (
    <group ref={groupRef}>
      {/* Scale up 2x so it fills the screen */}
      <primitive object={scene} scale={[2.2, 2.2, 2.2]} />

      {/* Fire breath at mouth — positioned for 2.2x scaled model */}
      <group position={[5.8, 0.3, 0]}>
        <FireBreath />
        {/* Fire point lights */}
        <pointLight color="#FF6B00" intensity={8}  distance={8}  decay={2} />
        <pointLight color="#FFD700" intensity={4}  distance={5}  decay={2} />
      </group>

      {/* Eye glows */}
      <pointLight position={[5.0, 0.9, 0.4]}  color="#FFD700" intensity={2} distance={2.5} />
      <pointLight position={[5.0, 0.9, -0.4]} color="#FFD700" intensity={2} distance={2.5} />
    </group>
  )
}

// ── Floating ember field ──────────────────────────────────────────────────────
function Embers() {
  const ref = useRef<THREE.Points>(null)
  const count = 280

  const geo = useMemo(() => {
    const pos    = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 30
      pos[i * 3 + 1] = (Math.random() - 0.5) * 18
      pos[i * 3 + 2] = (Math.random() - 0.5) * 12 - 3
      const isGold = Math.random() > 0.5
      colors[i * 3]     = isGold ? 0.8 : 1.0
      colors[i * 3 + 1] = isGold ? 0.64 : 0.42
      colors[i * 3 + 2] = 0
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos,    3))
    g.setAttribute('color',    new THREE.BufferAttribute(colors, 3))
    return g
  }, [])

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.03) * 0.1
      ref.current.position.y = Math.sin(clock.getElapsedTime() * 0.14) * 0.22
    }
  })

  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial size={0.045} vertexColors transparent opacity={0.6}
        blending={THREE.AdditiveBlending} depthWrite={false} />
    </points>
  )
}

// ── Ground fog / atmosphere plane ──────────────────────────────────────────────
function AtmosphereFog() {
  const t = useRef(0)
  const mat = useMemo(() => new THREE.MeshBasicMaterial({
    color: new THREE.Color('#C9A227'),
    transparent: true,
    opacity: 0.03,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  }), [])
  const ref = useRef<THREE.Mesh>(null)
  useFrame((_, delta) => {
    t.current += delta
    if (ref.current) {
      ref.current.position.y = -2.5 + Math.sin(t.current * 0.3) * 0.15
      ;(ref.current.material as THREE.MeshBasicMaterial).opacity = 0.03 + Math.sin(t.current * 0.5) * 0.01
    }
  })
  return (
    <mesh ref={ref} position={[0, -2.5, 0]} rotation={[-Math.PI / 2, 0, 0]} material={mat}>
      <planeGeometry args={[30, 20, 1, 1]} />
    </mesh>
  )
}

// ── Loading placeholder ────────────────────────────────────────────────────────
function Placeholder() {
  const ref = useRef<THREE.Mesh>(null)
  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = clock.getElapsedTime() * 0.9
      ref.current.rotation.x = Math.sin(clock.getElapsedTime() * 0.5) * 0.35
    }
  })
  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[1.5, 1]} />
      <meshStandardMaterial color="#C9A227" wireframe emissive="#8B6914" emissiveIntensity={0.5} />
    </mesh>
  )
}

// ── Cinematic camera ──────────────────────────────────────────────────────────
function CameraRig() {
  useFrame(({ camera, clock }) => {
    const t = clock.getElapsedTime()
    // Closer, wider, more dramatic
    camera.position.x = Math.sin(t * 0.14) * 1.5
    camera.position.y = Math.cos(t * 0.10) * 0.7 + 1.2
    camera.position.z = 6.5 + Math.sin(t * 0.07) * 1.2
    camera.lookAt(0, 0.5, 0)
  })
  return null
}

// ── Exported Canvas ───────────────────────────────────────────────────────────
export function DragonScene() {
  return (
    <Canvas
      camera={{ position: [0, 1.2, 6.5], fov: 62 }}   // wider, closer
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'transparent' }}
    >
      <color attach="background" args={['#000000']} />

      {/* Dramatic gold/fire lighting */}
      <ambientLight intensity={0.12} color="#1A0E00" />
      <directionalLight position={[5, 10, 4]}   intensity={0.8}  color="#FFD700" />
      <directionalLight position={[-4, 3, 2]}   intensity={0.35} color="#FF6B00" />
      <pointLight      position={[0, 6, 6]}     intensity={2.5}  color="#FF6B00" distance={18} />
      <pointLight      position={[-8, 2, 0]}    intensity={1.0}  color="#C9A227" distance={14} />

      <Suspense fallback={<Placeholder />}>
        <DragonModel />
      </Suspense>

      <Embers />
      <AtmosphereFog />
      <CameraRig />

      <EffectComposer>
        <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.88} intensity={3.0} mipmapBlur />
      </EffectComposer>
    </Canvas>
  )
}

useGLTF.preload('/dragon.glb')
