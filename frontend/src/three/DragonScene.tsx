import { Suspense, useRef, useMemo, Component, type ReactNode } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF, Environment } from '@react-three/drei'
import * as THREE from 'three'

// ── Error boundary so a WebGL crash doesn't blank the page ────────────────────
class CanvasErrorBoundary extends Component<{ children: ReactNode; fallback: ReactNode }> {
  state = { error: false }
  static getDerivedStateFromError() { return { error: true } }
  render() { return this.state.error ? this.props.fallback : this.props.children }
}

// ── Fire breath particles ─────────────────────────────────────────────────────
const FIRE_COUNT = 600

function FireBreath() {
  const t = useRef(0)

  const { positions, velocities, lifetimes, ages } = useMemo(() => {
    const positions  = new Float32Array(FIRE_COUNT * 3)
    const velocities = new Float32Array(FIRE_COUNT * 3)
    const lifetimes  = new Float32Array(FIRE_COUNT)
    const ages       = new Float32Array(FIRE_COUNT)
    for (let i = 0; i < FIRE_COUNT; i++) {
      positions[i * 3] = positions[i * 3 + 1] = positions[i * 3 + 2] = 0
      velocities[i * 3]     = Math.random() * 3.5 + 1
      velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.8
      velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.8
      lifetimes[i] = Math.random() * 1.2 + 0.3
      ages[i]      = Math.random() * 2
    }
    return { positions, velocities, lifetimes, ages }
  }, [])

  const colors = useMemo(() => new Float32Array(FIRE_COUNT * 3).fill(1), [])

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
        ages[i] = 0
        pos[i * 3] = pos[i * 3 + 1] = pos[i * 3 + 2] = 0
      } else {
        pos[i * 3]     += velocities[i * 3]     * delta
        pos[i * 3 + 1] += velocities[i * 3 + 1] * delta
        pos[i * 3 + 2] += velocities[i * 3 + 2] * delta
        const p = ages[i] / lifetimes[i]
        col[i * 3]     = 1
        col[i * 3 + 1] = Math.max(0, 1 - p * 1.5)
        col[i * 3 + 2] = 0
      }
    }
    geo.attributes.position.needsUpdate = true
    geo.attributes.color.needsUpdate    = true
  })

  return (
    <points geometry={geo}>
      <pointsMaterial size={0.1} vertexColors transparent opacity={0.85}
        blending={THREE.AdditiveBlending} depthWrite={false} sizeAttenuation />
    </points>
  )
}

// ── Embers field ──────────────────────────────────────────────────────────────
function Embers() {
  const ref = useRef<THREE.Points>(null)
  const geo = useMemo(() => {
    const count = 200
    const pos   = new Float32Array(count * 3)
    const col   = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 24
      pos[i * 3 + 1] = (Math.random() - 0.5) * 14
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10 - 2
      const gold = Math.random() > 0.4
      col[i * 3]     = gold ? 0.79 : 1.0
      col[i * 3 + 1] = gold ? 0.64 : 0.42
      col[i * 3 + 2] = 0
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('color',    new THREE.BufferAttribute(col, 3))
    return g
  }, [])

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.04) * 0.08
      ref.current.position.y = Math.sin(clock.getElapsedTime() * 0.18) * 0.18
    }
  })

  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial size={0.04} vertexColors transparent opacity={0.55}
        blending={THREE.AdditiveBlending} depthWrite={false} />
    </points>
  )
}

// ── Loaded dragon model ───────────────────────────────────────────────────────
function DragonModel() {
  const { scene } = useGLTF('/dragon.glb')
  const groupRef  = useRef<THREE.Group>(null)
  const t = useRef(0)

  // Force gold/metallic materials so the dragon is always visible
  useMemo(() => {
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh
        const mat = new THREE.MeshStandardMaterial({
          color:            new THREE.Color('#C9A227'),
          emissive:         new THREE.Color('#3D2800'),
          emissiveIntensity: 0.4,
          metalness:        0.7,
          roughness:        0.35,
        })
        // Try to keep the original map if it exists
        if (Array.isArray(mesh.material)) {
          mesh.material = mesh.material.map(m => {
            const src = m as THREE.MeshStandardMaterial
            if (src.map) { mat.map = src.map }
            return mat.clone()
          })
        } else {
          const src = mesh.material as THREE.MeshStandardMaterial
          if (src?.map) mat.map = src.map
          mesh.material = mat
        }
        mesh.castShadow    = true
        mesh.receiveShadow = false
      }
    })
  }, [scene])

  useFrame((_, delta) => {
    t.current += delta
    if (!groupRef.current) return
    groupRef.current.position.y  = Math.sin(t.current * 0.55) * 0.5
    groupRef.current.rotation.y  = Math.sin(t.current * 0.20) * 0.25
    groupRef.current.rotation.z  = Math.sin(t.current * 0.38) * 0.05
  })

  return (
    <group ref={groupRef}>
      <primitive object={scene} scale={[1.8, 1.8, 1.8]} />

      {/* Fire breath emitter at the dragon's head */}
      <group position={[4.5, 0.3, 0]}>
        <FireBreath />
        <pointLight color="#FF6B00" intensity={12} distance={7}  decay={2} />
        <pointLight color="#FFD700" intensity={5}  distance={4}  decay={2} />
      </group>

      {/* Eye glows */}
      <pointLight position={[4.0, 0.8, 0.35]}  color="#FFD700" intensity={3} distance={2.5} />
      <pointLight position={[4.0, 0.8, -0.35]} color="#FFD700" intensity={3} distance={2.5} />
    </group>
  )
}

// ── Spinning gold placeholder while GLB loads ─────────────────────────────────
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
      <icosahedronGeometry args={[1.4, 1]} />
      <meshStandardMaterial color="#C9A227" wireframe emissive="#C9A227" emissiveIntensity={0.8} />
    </mesh>
  )
}

// ── Slow cinematic camera drift ───────────────────────────────────────────────
function CameraRig() {
  useFrame(({ camera, clock }) => {
    const t = clock.getElapsedTime()
    camera.position.x = Math.sin(t * 0.12) * 1.2
    camera.position.y = Math.cos(t * 0.09) * 0.6 + 1.0
    camera.position.z = 5.5 + Math.sin(t * 0.06) * 0.8
    camera.lookAt(0, 0.3, 0)
  })
  return null
}

// ── CSS fallback if WebGL fails ───────────────────────────────────────────────
function DragonFallback() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <svg viewBox="0 0 400 300" width="80%" height="80%" style={{ opacity: 0.35 }}>
        {/* Simple stylised dragon silhouette */}
        <ellipse cx="200" cy="150" rx="120" ry="40" fill="none" stroke="#C9A227" strokeWidth="1.5"/>
        <path d="M80,150 Q120,80 200,100 Q280,120 320,150 Q280,200 200,180 Q120,160 80,150Z"
          fill="rgba(201,162,39,0.08)" stroke="#C9A227" strokeWidth="1"/>
        <circle cx="280" cy="130" r="8" fill="none" stroke="#C9A227" strokeWidth="1.5"/>
        <circle cx="285" cy="130" r="3" fill="#FF6B00"/>
        <path d="M300,125 Q330,100 350,115" stroke="#C9A227" strokeWidth="1" fill="none"/>
        <path d="M300,135 Q320,130 345,120" stroke="#C9A227" strokeWidth="1" fill="none"/>
      </svg>
    </div>
  )
}

// ── Exported scene ────────────────────────────────────────────────────────────
export function DragonScene() {
  return (
    <CanvasErrorBoundary fallback={<DragonFallback />}>
      <Canvas
        camera={{ position: [0, 1.0, 5.5], fov: 60 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
        onCreated={({ gl }) => {
          gl.shadowMap.enabled = true
          gl.shadowMap.type    = THREE.PCFSoftShadowMap
          gl.toneMapping       = THREE.ACESFilmicToneMapping
          gl.toneMappingExposure = 1.4
        }}
      >
        {/* Rich HDR-style environment for realistic PBR lighting */}
        <Environment preset="night" />

        {/* Additional dramatic fire/gold lights */}
        <ambientLight intensity={0.08} color="#0D0800" />
        <directionalLight position={[6, 10, 5]}   intensity={1.2}  color="#FFD700" />
        <directionalLight position={[-5, 4, 2]}   intensity={0.5}  color="#FF6B00" />
        <pointLight      position={[0, 8, 6]}     intensity={3.0}  color="#FF8C00" distance={20} />
        <pointLight      position={[-6, 2, 0]}    intensity={1.5}  color="#C9A227" distance={15} />

        <Suspense fallback={<Placeholder />}>
          <DragonModel />
        </Suspense>

        <Embers />
        <CameraRig />
      </Canvas>
    </CanvasErrorBoundary>
  )
}

useGLTF.preload('/dragon.glb')
