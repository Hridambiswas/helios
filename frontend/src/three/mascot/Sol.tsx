import { useMemo, useRef, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { solConfig } from './config'

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
 */

interface Props {
  position?: [number, number, number]
  scale?: number
  prefersReducedMotion?: boolean
}

// Fresnel + rim-light material. The uniform `uSunDir` points at the
// sun (in world space) so the rim always lights the correct side.
function useFurMaterial({
  base, shadow, rim,
}: { base: string; shadow: string; rim: string }) {
  return useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        uBase:   { value: new THREE.Color(base) },
        uShadow: { value: new THREE.Color(shadow) },
        uRim:    { value: new THREE.Color(rim) },
        uSunDir: { value: new THREE.Vector3(1.0, 0.2, 0.4).normalize() },
      },
      vertexShader: /* glsl */`
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vNormalW = normalize(mat3(modelMatrix) * normal);
          vViewDir = normalize(-mvPosition.xyz);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: /* glsl */`
        precision highp float;
        uniform vec3 uBase;
        uniform vec3 uShadow;
        uniform vec3 uRim;
        uniform vec3 uSunDir;
        varying vec3 vNormalW;
        varying vec3 vViewDir;
        void main() {
          // Sun-side amount and shadow-side amount
          float sunDot   = clamp(dot(vNormalW, uSunDir), -1.0, 1.0);
          float sunMix   = smoothstep(-0.1, 0.9, sunDot);
          float shadeMix = smoothstep(0.4, -0.6, sunDot);

          // Fresnel term: falls off at grazing angles → creamy edge glow.
          float fres = pow(1.0 - clamp(dot(vNormalW, vViewDir), 0.0, 1.0), 2.2);

          // Base fur → mixed with cool shadow, then warm rim added on the sun side.
          vec3 body = mix(uBase, uShadow, shadeMix * 0.65);
          vec3 rim  = uRim * fres * (0.35 + 0.65 * sunMix);
          gl_FragColor = vec4(body + rim, 1.0);
        }
      `,
    })
  }, [base, shadow, rim])
}

export function Sol({
  position = [0, 0, 0],
  scale = 1,
  prefersReducedMotion = false,
}: Props) {
  const cfg = solConfig
  const group     = useRef<THREE.Group>(null!)
  const body      = useRef<THREE.Mesh>(null!)
  const tailPivot = useRef<THREE.Group>(null!)
  const leftEyeInner  = useRef<THREE.Group>(null!)
  const rightEyeInner = useRef<THREE.Group>(null!)
  const leftEyeLid    = useRef<THREE.Mesh>(null!)
  const rightEyeLid   = useRef<THREE.Mesh>(null!)

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

    // ── Breathing ─────────────────────────────────────────────────────────
    if (body.current) {
      const s = 1 + Math.sin(t * cfg.breathHz * Math.PI * 2) * cfg.breathAmp * motion
      body.current.scale.set(1, s, 1)
    }

    // ── Tail swish ────────────────────────────────────────────────────────
    if (tailPivot.current) {
      tailPivot.current.rotation.z =
        Math.sin(t * cfg.tailSwishHz * Math.PI * 2) * cfg.tailSwishAmp * motion
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
    // Camera hint used so the material's viewDir stays sensible even under
    // an orthographic camera (no perspective divide in shader inputs).
    void camera
  })

  const eyeOffX = cfg.eyeOffset[0]
  const eyeOffY = cfg.eyeOffset[1]
  const eyeOffZ = cfg.eyeOffset[2]

  return (
    <group ref={group} position={position} scale={scale}>
      {/* ── Body (capsule) ────────────────────────────────────────────── */}
      <mesh ref={body} material={furBody}>
        <capsuleGeometry args={[cfg.bodyRadius, cfg.bodyLength * 2, 8, 20]} />
      </mesh>

      {/* ── Head ──────────────────────────────────────────────────────── */}
      <group position={cfg.headOffset as [number, number, number]}>
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

      {/* A soft key light so the fresnel material has something to react
          to under the orthographic camera. */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[1, 0.4, 1.6]} intensity={0.9} color="#FFF6E5" />
      <directionalLight position={[-1.2, -0.1, -0.8]} intensity={0.35} color="#B8C6E2" />
    </group>
  )
}
