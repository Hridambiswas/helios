import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// ─── Shaders ─────────────────────────────────────────────────────────────────

const VERT = /* glsl */`
precision highp float;
attribute vec3 position;
attribute vec2 uv;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const FRAG = /* glsl */`
precision highp float;
varying vec2 vUv;

uniform float uTime;
uniform vec2  uMouse;
uniform vec2  uResolution;

// ── Simplex noise 2D (Ian McEwan) ────────────────────────────────────────────
vec3 smod289(vec3 x){ return x - floor(x*(1./289.))*289.; }
vec2 smod289v2(vec2 x){ return x - floor(x*(1./289.))*289.; }
vec3 sperm(vec3 x){ return smod289(((x*34.)+1.)*x); }

float snoise(vec2 v){
  const vec4 C=vec4(.211324865405187,.366025403784439,-.577350269189626,.024390243902439);
  vec2 i=floor(v+dot(v,C.yy));
  vec2 x0=v-i+dot(i,C.xx);
  vec2 i1=(x0.x>x0.y)?vec2(1.,0.):vec2(0.,1.);
  vec4 x12=x0.xyxy+C.xxzz; x12.xy-=i1;
  i=smod289v2(i);
  vec3 p=sperm(sperm(i.y+vec3(0.,i1.y,1.))+i.x+vec3(0.,i1.x,1.));
  vec3 m=max(.5-vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);
  m=m*m; m=m*m;
  vec3 x=2.*fract(p*C.www)-1.;
  vec3 h=abs(x)-.5;
  vec3 a0=x-floor(x+.5);
  m*=1.79284291400159-.85373472095314*(a0*a0+h*h);
  vec3 g;
  g.x  = a0.x*x0.x  + h.x*x0.y;
  g.yz = a0.yz*x12.xz + h.yz*x12.yw;
  return 130.*dot(m,g);
}

mat2 rot2(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

float fbm(vec2 p){
  float v=0., a=.5;
  mat2 R=rot2(.37);
  for(int i=0;i<5;i++){ v+=a*snoise(p); p=R*p*2.03; a*=.49; }
  return v;
}

void main(){
  vec2 uv = vUv;
  float ar = uResolution.x / uResolution.y;
  vec2 p   = (uv - .5) * vec2(ar, 1.);
  float t  = uTime * .10;

  // Mouse ripple
  vec2 mouse = (uMouse - .5) * vec2(ar, 1.);
  float md   = length(p - mouse);
  float ripple  = .22 * exp(-md * 4.5) * sin(md * 12. - t * 9.);
  vec2 rDir  = normalize(p - mouse + .001) * ripple;

  // 2-pass domain warp
  vec2 q = vec2(fbm(p + t*vec2(.8,-.4)),
                fbm(p + t*vec2(-.5,.7)));
  vec2 r = vec2(fbm(p + 1.9*q + t*vec2(1.,-.5) + rDir),
                fbm(p + 1.9*q + t*vec2(-.6,.9) + rDir));
  float n = fbm(p + 2.4*r + rDir);

  // Surface normals → specular
  float eps = .003;
  float dnx = fbm(p+vec2(eps,0.)+2.4*r) - fbm(p-vec2(eps,0.)+2.4*r);
  float dny = fbm(p+vec2(0.,eps)+2.4*r) - fbm(p-vec2(0.,eps)+2.4*r);
  vec3 norm  = normalize(vec3(dnx*5., dny*5., eps*14.));

  vec3 l1 = normalize(vec3(-.6, .8, 1.0));
  vec3 l2 = normalize(vec3( .5,-.3, .7));
  float s1 = pow(max(dot(norm,l1),0.), 36.) * .35;
  float s2 = pow(max(dot(norm,l2),0.), 22.) * .20;

  // Oil-slick iridescence — clearly visible
  float ii   = snoise(p*4.5 + t*2.2) * .5 + .5;
  vec3  irid = .5 + .5*cos(vec3(0., 2.094, 4.189) + ii*4.8 + 1.8);
  irid *= vec3(.35, .45, 1.0); // blue-violet tint

  // Large-scale undulation (visible surface topology)
  float bigN = fbm(p*.45 + t*.07) * .5 + .5;
  float depth = bigN;

  // ── Compose (dark liquid with clear surface detail) ──────────────────────
  vec3 col = vec3(0.);

  // Base dark purple tint
  col += vec3(.025, .010, .060) * depth;

  // Iridescence — 50% strength, clearly visible
  col += irid * .50 * (n*.4 + .6);

  // Specular highlights — 2× brighter
  col += vec3(s1 + s2) * vec3(.65,.75,1.) * 2.0;

  // Micro gloss — adds 'wet' look
  float gloss = snoise(p*8. + t*1.4) * .5 + .5;
  col += vec3(.01,.005,.025) * gloss * bigN;

  // Vignette (slightly lighter at center to show the fluid mass)
  float vig = 1. - dot(uv-.5, uv-.5) * 1.6;
  col *= max(vig, 0.);

  gl_FragColor = vec4(col, 1.);
}
`

// ─── R3F mesh ─────────────────────────────────────────────────────────────────

function FluidMesh({ mouseRef }: { mouseRef: React.MutableRefObject<{ x: number; y: number }> }) {
  const uniforms = useMemo(() => ({
    uTime:       { value: 0.0 },
    uMouse:      { value: new THREE.Vector2(0.5, 0.5) },
    uResolution: { value: new THREE.Vector2(1, 1) },
  }), [])

  useFrame(({ clock, size }) => {
    uniforms.uTime.value = clock.getElapsedTime()
    uniforms.uMouse.value.set(mouseRef.current.x, mouseRef.current.y)
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

// ─── Export ───────────────────────────────────────────────────────────────────

export function FluidBackground({ mouseRef }: { mouseRef: React.MutableRefObject<{ x: number; y: number }> }) {
  return (
    <Canvas
      camera={{ position: [0, 0, 1] }}
      gl={{ antialias: false, alpha: false }}
      style={{ display: 'block', width: '100%', height: '100%' }}
    >
      <FluidMesh mouseRef={mouseRef} />
    </Canvas>
  )
}
