import type React from 'react'

const V = '#8b5cf6'

export function FireDivider({ flip = false }: { flip?: boolean }) {
  return (
    <div className={`relative h-16 overflow-hidden ${flip ? 'scale-y-[-1]' : ''}`}>
      <svg viewBox="0 0 1400 64" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        <path d="M0,0 C200,64 400,0 700,32 C1000,64 1200,0 1400,32 L1400,64 L0,64 Z" fill="rgba(139,92,246,0.04)" />
        <path d="M0,20 C150,64 350,8 600,36 C850,64 1100,8 1400,40 L1400,64 L0,64 Z" fill="rgba(139,92,246,0.02)" />
      </svg>
      <div className="absolute top-1/2 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.25), transparent)' }} />
    </div>
  )
}

export function DragonScales({ opacity = 0.06 }: { opacity?: number }) {
  return (
    <div className="absolute inset-0 pointer-events-none" style={{ opacity }}>
      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="scales" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse">
            <path d="M0,26 Q15,0 30,26 Q45,0 60,26" fill="none" stroke={V} strokeWidth="0.7"/>
            <path d="M-30,52 Q-15,26 0,52 Q15,26 30,52 Q45,26 60,52 Q75,26 90,52" fill="none" stroke={V} strokeWidth="0.7"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#scales)"/>
      </svg>
    </div>
  )
}

export function ClawMark({ corner, size = 120, opacity = 0.08 }: { corner: 'tl'|'tr'|'bl'|'br'; size?: number; opacity?: number }) {
  const style: React.CSSProperties = {
    position: 'absolute',
    width: size,
    height: size,
    opacity,
    pointerEvents: 'none',
    top:    corner.startsWith('t') ? 0 : undefined,
    bottom: corner.startsWith('b') ? 0 : undefined,
    left:   corner.endsWith('l')   ? 0 : undefined,
    right:  corner.endsWith('r')   ? 0 : undefined,
    transform: corner === 'tr' ? 'scaleX(-1)' : corner === 'br' ? 'scale(-1,-1)' : corner === 'bl' ? 'scaleY(-1)' : undefined,
  }
  return (
    <div style={style}>
      <svg viewBox="0 0 100 100" fill="none" width="100%" height="100%">
        <path d="M5,5 Q30,20 20,55"  stroke={V} strokeWidth="2.5" strokeLinecap="round"/>
        <path d="M22,5 Q45,18 38,52" stroke={V} strokeWidth="2"   strokeLinecap="round"/>
        <path d="M40,5 Q60,16 58,48" stroke={V} strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    </div>
  )
}

export function DragonEye({ size = 28, opacity = 0.7 }: { size?: number; opacity?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" style={{ opacity }}>
      <ellipse cx="16" cy="16" rx="14" ry="9" stroke={V} strokeWidth="1.5"/>
      <ellipse cx="16" cy="16" rx="5"  ry="9" fill={V} fillOpacity="0.2" stroke={V} strokeWidth="1.2"/>
      <circle  cx="16" cy="16" r="3.5" fill="#c026d3"/>
      <circle  cx="16" cy="16" r="1.8" fill="#000"/>
    </svg>
  )
}

export function WingSilhouette({ side, opacity = 0.07 }: { side: 'left'|'right'; opacity?: number }) {
  const flip = side === 'right'
  return (
    <div
      className="absolute top-0 bottom-0 pointer-events-none"
      style={{ [side]: 0, opacity, transform: flip ? 'scaleX(-1)' : undefined, width: '18%', maxWidth: 180 }}
    >
      <svg viewBox="0 0 140 400" fill="none" preserveAspectRatio="none" width="100%" height="100%">
        <path d="M140,200 Q60,80 10,40 Q0,100 30,160 Q0,180 20,240 Q0,290 40,340 Q80,300 140,200Z"
          fill={V} fillOpacity="0.15" stroke={V} strokeWidth="0.5"/>
        <path d="M140,200 Q80,120 40,80"  stroke={V} strokeWidth="0.5" strokeOpacity="0.3"/>
        <path d="M140,200 Q70,160 30,150" stroke={V} strokeWidth="0.5" strokeOpacity="0.25"/>
        <path d="M140,200 Q90,240 50,280" stroke={V} strokeWidth="0.5" strokeOpacity="0.2"/>
      </svg>
    </div>
  )
}

export function DragonHeadIcon({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <path d="M12 3 Q16 6 18 12 Q16 18 12 21 Q8 18 6 12 Q8 6 12 3Z" stroke={V} strokeWidth="1.2" fill="none"/>
      <circle cx="12" cy="12" r="2" fill={V} fillOpacity="0.5"/>
    </svg>
  )
}
