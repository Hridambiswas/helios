import { useMemo, useRef, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { solConfig } from './config'
import type { PipelinePhase } from '../../pipeline/events'

/**
 * Sol — procedural ermine mascot.
 *
 * All geometry is composed from primitive shapes (capsule, sphere,
 * curved cylinder). No external models. Materials use a fresnel-rim
 * shader that lights the sun-facing side warm and the opposite side
 * cool, matching the hero's low-sun rig.
 *
 * Idle behaviours (respect prefers-reduced-motion):
 *   • breathing         — subtle Y-scale on the body
 *   • blinking          — timed eye pinch
 *   • tail swish        — sinusoidal rotation of the tail rig
 *   • cursor-follow eyes — pupils rotate to track the pointer
 *
 * Pipeline-driven states (mapped to phase):
 *   planning     → alert pose: stands up slightly, head tilt
 *   retrieving   → nose down: head tilts forward, small nod
 *   executing    → focused: extra still (breath amp × 0.4)
 *   synthesizing → glow: rim colour intensifies (uRim brighter)
 *   evaluating   → watching: head tilts up toward the arc
 *   done         → happy hop: quick vertical hop then rest
 *   error        → sad loaf: sinks slightly, ears down (tail hangs low)
 */

interface Props {
  position?: [number, number, number]
  scale?: number
  prefersReducedMotion?: boolean
  phase?: PipelinePhase
  passed?: boolean
}

// Fur material — MeshStandardMaterial reacts to the scene's directional
// lights so shading is correct under any renderer (native or SwiftShader).
// The old custom fresnel shader produced near-white output under an
// orthographic camera where most visible normals faced the camera, making
// Sol invisible against the snow crest.
function useFurMaterial({
  base, shadow: _shadow, rim,
}: { base: string; shadow: string; rim: string }) {
  return useMemo(() => {
    // MeshLambertMaterial + high emissive: renders correctly on real
    // GPUs (the ambient + directional lights sculpt Sol nicely) and
    // stays visible under low-quality fallbacks (the emissive term
    // paints the surface even when normals barely respond).
    return new THREE.MeshLambertMaterial({
      color:             new THREE.Color(base),
      emissive:          new THREE.Color(rim),
      emissiveIntensity: 0.20,
    })
  }, [base, _shadow, rim])
}

export function Sol({
  position = [0, 0, 0],
  scale = 1,
  prefersReducedMotion = false,
  phase = 'idle',
  passed,
}: Props) {
  const cfg = solConfig
  const group     = useRef<THREE.Group>(null!)
  const body      = useRef<THREE.Mesh>(null!)
  const head      = useRef<THREE.Group>(null!)
  const tailPivot = useRef<THREE.Group>(null!)
  const leftEyeInner  = useRef<THREE.Group>(null!)
  const rightEyeInner = useRef<THREE.Group>(null!)
  const leftEyeLid    = useRef<THREE.Mesh>(null!)
  const rightEyeLid   = useRef<THREE.Mesh>(null!)

  // Persistent per-phase timers.
  const phaseStart = useRef<number>(0)
  const prevPhase  = useRef<PipelinePhase>('idle')
  useEffect(() => {
    if (prevPhase.current !== phase) {
      prevPhase.current = phase
      phaseStart.current = performance.now()
    }
  }, [phase])

  // Cursor tracking — updated from a window-level pointermove listener so
  // the pupils follow the real page cursor rather than the canvas.
  const cursor = useRef({ nx: 0, ny: 0 })
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      cursor.current.nx = (e.clientX / window.innerWidth) * 2 - 1
      cursor.current.ny = (e.clientY / window.innerHeight) * 2 - 1
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  // Blink schedule with jitter so it feels alive.
  const nextBlink = useRef<number>(cfg.blinkEvery)
  const blinkPhase = useRef<number>(0) // 0..1 where 0.5 = fully shut

  const furBody  = useFurMaterial({ base: cfg.fur,      shadow: cfg.furShadow, rim: cfg.furRim })
  const furEar   = useFurMaterial({ base: cfg.earInner, shadow: cfg.furShadow, rim: cfg.furRim })
  const eyeMat   = useMemo(() => new THREE.MeshStandardMaterial({
    color: cfg.eye, roughness: 0.2, metalness: 0.1,
  }), [cfg.eye])
  const lidMat   = useMemo(() => new THREE.MeshBasicMaterial({
    color: cfg.fur, transparent: true, opacity: 0,
  }), [cfg.fur])
  const noseMat  = useMemo(() => new THREE.MeshStandardMaterial({
    color: cfg.nose, roughness: 0.6,
  }), [cfg.nose])
  const tailTip  = useFurMaterial({ base: cfg.tailTip, shadow: cfg.tailTip, rim: cfg.tailTipGlow })

  const { camera } = useThree()

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime
    const motion = prefersReducedMotion ? 0 : 1
    const sincePhase = (performance.now() - phaseStart.current) / 1000

    // Per-phase mask — some behaviours attenuate or intensify under certain phases.
    const breathScale =
      phase === 'executing' ? 0.4 :
      phase === 'error'     ? 0.3 :
      1
    const tailScale =
      phase === 'planning' ? 1.4 :
      phase === 'done' && passed !== false ? 1.6 :
      phase === 'error' ? 0.2 :
      1

    // ── Breathing ─────────────────────────────────────────────────────────
    if (body.current) {
      const s = 1 + Math.sin(t * cfg.breathHz * Math.PI * 2) * cfg.breathAmp * motion * breathScale
      body.current.scale.set(1, s, 1)
    }

    // ── Tail swish ────────────────────────────────────────────────────────
    if (tailPivot.current) {
      tailPivot.current.rotation.z =
        Math.sin(t * cfg.tailSwishHz * Math.PI * 2) * cfg.tailSwishAmp * motion * tailScale
        + (phase === 'error' ? -0.35 : 0)
    }

    // ── Phase-driven pose (group Y + head tilt + happy hop) ────────────────
    if (group.current) {
      let poseY = 0
      let headTilt = 0
      // hop timing on 'done' + pass
      if (phase === 'done' && passed !== false && motion === 1 && sincePhase < 0.9) {
        // easeOutQuad hop up over 0.3s, back down by 0.6s
        const hop = sincePhase < 0.3
          ? Math.sin((sincePhase / 0.3) * Math.PI) * 0.18
          : sincePhase < 0.6
            ? Math.sin(((sincePhase - 0.3) / 0.3 + 1) * Math.PI) * 0.09
            : 0
        poseY += hop
      }
      if (phase === 'error') poseY -= 0.05  // sad slump
      if (phase === 'planning')     headTilt = 0.20
      if (phase === 'retrieving')   headTilt = -0.25   // nose down
      if (phase === 'synthesizing') headTilt = 0.05
      if (phase === 'evaluating')   headTilt = 0.30    // looking up at arc
      if (phase === 'error')        headTilt = -0.35   // eyes down
      // Smoothly ease group + head toward targets.
      const gy = position[1] + poseY
      group.current.position.y += (gy - group.current.position.y) * 0.15
      if (head.current) {
        head.current.rotation.x += (headTilt - head.current.rotation.x) * 0.10
      }
    }

    // ── Blink ─────────────────────────────────────────────────────────────
    nextBlink.current -= dt
    if (nextBlink.current <= 0) {
      blinkPhase.current = 1
      nextBlink.current = cfg.blinkEvery + Math.random() * 2.5
    }
    if (blinkPhase.current > 0) {
      blinkPhase.current = Math.max(0, blinkPhase.current - dt / cfg.blinkDur)
      const shut = 1 - Math.abs(blinkPhase.current - 0.5) * 2 // 0→1→0
      const lidY = motion === 0 ? 0 : shut
      if (leftEyeLid.current)  (leftEyeLid.current.material as THREE.Material & { opacity: number }).opacity  = lidY
      if (rightEyeLid.current) (rightEyeLid.current.material as THREE.Material & { opacity: number }).opacity = lidY
    }

    // ── Cursor tracking eyes ──────────────────────────────────────────────
    if (leftEyeInner.current && rightEyeInner.current) {
      const { nx, ny } = cursor.current
      const maxR = cfg.cursorEyeMax
      const rx = ny * maxR
      const ry = nx * maxR
      leftEyeInner.current.rotation.set(rx, ry, 0)
      rightEyeInner.current.rotation.set(rx, ry, 0)
    }

    // Very subtle head bob toward the cursor (mostly turns head to follow).
    if (group.current) {
      const targetY = cursor.current.nx * 0.06 * motion
      group.current.rotation.y += (targetY - group.current.rotation.y) * 0.08
    }

    // Synthesizing warms Sol's fur toward a cream tint.
    const rimBoost = phase === 'synthesizing' ? 0.30 + 0.15 * Math.sin(t * 3.0) : 0
    const target = new THREE.Color(cfg.fur).lerp(new THREE.Color(cfg.furRim), rimBoost)
    furBody.color.lerp(target, 0.10)
    void camera
  })

  const eyeOffX = cfg.eyeOffset[0]
  const eyeOffY = cfg.eyeOffset[1]
  const eyeOffZ = cfg.eyeOffset[2]

  return (
    <group ref={group} position={position} scale={scale}>
      {/* ── Body (capsule, laid horizontal — ermines are long and low). ── */}
      <mesh ref={body} material={furBody} rotation={[0, 0, Math.PI / 2]}>
        <capsuleGeometry args={[cfg.bodyRadius, cfg.bodyLength * 2, 12, 24]} />
      </mesh>

      {/* ── Head ──────────────────────────────────────────────────────── */}
      <group ref={head} position={cfg.headOffset as [number, number, number]}>
        <mesh material={furBody}>
          <sphereGeometry args={[cfg.headRadius, 32, 24]} />
        </mesh>

        {/* Ears — two cone-ish shapes on top */}
        {[-1, 1].map(side => (
          <group key={side} position={[0, cfg.earHeight, side * cfg.earSpread]}
                 rotation={[0, 0, side * 0.15]}>
            <mesh material={furBody}>
              <coneGeometry args={[cfg.earSize, cfg.earSize * 2.2, 12]} />
            </mesh>
            {/* Inner ear */}
            <mesh material={furEar} position={[0.005, 0, 0]} scale={[0.55, 0.6, 0.55]}>
              <coneGeometry args={[cfg.earSize, cfg.earSize * 2.2, 12]} />
            </mesh>
          </group>
        ))}

        {/* Eyes — outer sockets + inner rotating groups for cursor tracking */}
        {[-1, 1].map((side, i) => {
          const ref = i === 0 ? leftEyeInner : rightEyeInner
          const lidRef = i === 0 ? leftEyeLid : rightEyeLid
          return (
            <group key={side} position={[eyeOffX, eyeOffY, side * cfg.eyeSpread * (eyeOffZ > 0 ? 1 : 1)]}>
              <group ref={ref}>
                <mesh material={eyeMat} position={[0, 0, 0]}>
                  <sphereGeometry args={[cfg.eyeSize, 20, 16]} />
                </mesh>
                {/* Highlight — small offset white bump */}
                <mesh position={[cfg.eyeSize * 0.35, cfg.eyeSize * 0.35, 0]}>
                  <sphereGeometry args={[cfg.eyeSize * 0.28, 12, 10]} />
                  <meshBasicMaterial color="#EEF2F8" />
                </mesh>
              </group>
              {/* Lid — a fur-coloured cap that fades in during blinks */}
              <mesh ref={lidRef} material={lidMat} position={[cfg.eyeSize * 0.1, 0, 0]} scale={[1.05, 1.05, 1.05]}>
                <sphereGeometry args={[cfg.eyeSize * 1.05, 16, 12]} />
              </mesh>
            </group>
          )
        })}

        {/* Nose — small dark bump */}
        <mesh material={noseMat} position={[cfg.headRadius * 0.98, -cfg.headRadius * 0.15, 0]}>
          <sphereGeometry args={[cfg.noseSize, 12, 10]} />
        </mesh>
      </group>

      {/* ── Tail (curved cylinder, pivoted at the base) ──────────────── */}
      <group ref={tailPivot} position={[-cfg.bodyLength * 0.9, -0.02, 0]}>
        <mesh material={furBody} rotation={[0, 0, Math.PI * 0.55]}>
          {/* Long cylinder body of the tail */}
          <cylinderGeometry
            args={[cfg.bodyRadius * 0.6, cfg.bodyRadius * 0.35, cfg.tailLength, 12, 4, false]}
          />
        </mesh>
        {/* Black tip */}
        <mesh material={tailTip}
              rotation={[0, 0, Math.PI * 0.55]}
              position={[
                -Math.cos(Math.PI * 0.55) * cfg.tailLength * (1 - cfg.tailTipRatio / 2),
                -Math.sin(Math.PI * 0.55) * cfg.tailLength * (1 - cfg.tailTipRatio / 2),
                0,
              ]}>
          <cylinderGeometry
            args={[cfg.bodyRadius * 0.35, cfg.bodyRadius * 0.22, cfg.tailLength * cfg.tailTipRatio, 12, 2, false]}
          />
        </mesh>
      </group>

      {/* Warm key light on the sun-facing side, cool fill from the
          shadow side, and a lot of ambient — ortho cameras only give
          the shader a small number of visible normals so we make sure
          Sol reads solid regardless of which face is showing. */}
      <ambientLight intensity={1.0} />
      <directionalLight position={[1, 0.4, 1.6]} intensity={0.8} color="#FFF6E5" />
      <directionalLight position={[-1.2, 0.6, 0.8]} intensity={0.6} color="#B8C6E2" />
      <directionalLight position={[0, 1, 0.2]}  intensity={0.35} color="#FFFFFF" />
    </group>
  )
}
