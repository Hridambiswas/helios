// Shared dragon decorative elements used across all sections

/** Subtle SVG dragon scale overlay */
export function DragonScales({ opacity = 0.07 }: { opacity?: number }) {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden" style={{ opacity }}>
      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="dragonScales" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse">
            <path d="M0,26 Q15,0 30,26 Q45,0 60,26" fill="none" stroke="#C9A227" strokeWidth="0.8" />
            <path d="M-30,52 Q-15,26 0,52 Q15,26 30,52 Q45,26 60,52 Q75,26 90,52" fill="none" stroke="#C9A227" strokeWidth="0.8" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#dragonScales)" />
      </svg>
    </div>
  )
}

/** Corner claw mark — top-left by default */
export function ClawMark({ corner = 'tl', size = 120, opacity = 0.08 }: {
  corner?: 'tl' | 'tr' | 'bl' | 'br'
  size?: number
  opacity?: number
}) {
  const pos: Record<string, React.CSSProperties> = {
    tl: { top: 0, left: 0 },
    tr: { top: 0, right: 0, transform: 'scaleX(-1)' },
    bl: { bottom: 0, left: 0, transform: 'scaleY(-1)' },
    br: { bottom: 0, right: 0, transform: 'scale(-1,-1)' },
  }
  return (
    <div className="absolute pointer-events-none" style={{ ...pos[corner], opacity, width: size, height: size }}>
      <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        {/* Three curved claw slashes */}
        <path d="M5,5 Q30,20 20,55" stroke="#C9A227" strokeWidth="2.5" strokeLinecap="round"/>
        <path d="M22,5 Q45,18 38,52" stroke="#C9A227" strokeWidth="2" strokeLinecap="round"/>
        <path d="M40,5 Q60,16 58,48" stroke="#C9A227" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    </div>
  )
}

/** Dragon eye glyph — for section headers */
export function DragonEye({ size = 32, opacity = 0.5 }: { size?: number; opacity?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ opacity }}>
      <ellipse cx="16" cy="16" rx="14" ry="9" stroke="#C9A227" strokeWidth="1.5"/>
      <ellipse cx="16" cy="16" rx="5"  ry="9" fill="#C9A227" fillOpacity="0.2" stroke="#C9A227" strokeWidth="1.2"/>
      <circle  cx="16" cy="16" r="3"   fill="#FF6B00"/>
      <circle  cx="16" cy="16" r="1.5" fill="#000"/>
      <line x1="2"  y1="16" x2="8"  y2="16" stroke="#C9A227" strokeWidth="0.8" strokeOpacity="0.5"/>
      <line x1="24" y1="16" x2="30" y2="16" stroke="#C9A227" strokeWidth="0.8" strokeOpacity="0.5"/>
    </svg>
  )
}

/** Horizontal fire divider between sections */
export function FireDivider({ flip = false }: { flip?: boolean }) {
  return (
    <div className={`relative h-16 overflow-hidden ${flip ? 'scale-y-[-1]' : ''}`}>
      <svg viewBox="0 0 1400 64" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
        <path d="M0,0 C200,64 400,0 700,32 C1000,64 1200,0 1400,32 L1400,64 L0,64 Z"
          fill="rgba(201,162,39,0.035)" />
        <path d="M0,20 C150,64 350,8 600,36 C850,64 1100,8 1400,40 L1400,64 L0,64 Z"
          fill="rgba(201,162,39,0.02)" />
      </svg>
      <div className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-gold/25 to-transparent" />
      {/* Ember dots */}
      <div className="absolute top-1/2 left-1/4 w-1 h-1 rounded-full bg-fire/50 -translate-y-1/2" />
      <div className="absolute top-1/2 left-1/2 w-1.5 h-1.5 rounded-full bg-gold/40 -translate-y-1/2"
        style={{ boxShadow: '0 0 6px rgba(201,162,39,0.6)' }} />
      <div className="absolute top-1/2 left-3/4 w-1 h-1 rounded-full bg-fire/50 -translate-y-1/2" />
    </div>
  )
}

/** Wing silhouette for footer / section edges */
export function WingSilhouette({ side = 'left', opacity = 0.06 }: { side?: 'left' | 'right'; opacity?: number }) {
  const isRight = side === 'right'
  return (
    <div
      className="absolute bottom-0 pointer-events-none"
      style={{
        [isRight ? 'right' : 'left']: 0,
        opacity,
        transform: isRight ? 'scaleX(-1)' : undefined,
        width: 'clamp(160px, 22vw, 300px)',
        height: 'clamp(110px, 16vw, 200px)',
      }}
    >
      <svg viewBox="0 0 300 200" fill="none" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        {/* Bat-style dragon wing */}
        <path
          d="M0,200 Q20,140 60,80 Q80,50 100,30 Q120,10 150,0 Q130,40 120,80 Q110,120 105,160 Q100,180 100,200 Z"
          fill="#C9A227"
        />
        {/* Wing membrane lines */}
        <path d="M0,200 Q30,120 80,50 Q110,20 140,5"  stroke="#FFD700" strokeWidth="0.6" strokeOpacity="0.5"/>
        <path d="M30,200 Q55,140 90,90 Q110,60 130,40" stroke="#FFD700" strokeWidth="0.5" strokeOpacity="0.4"/>
        <path d="M60,200 Q78,155 100,120 Q115,95 125,75" stroke="#FFD700" strokeWidth="0.4" strokeOpacity="0.3"/>
        <path d="M85,200 Q98,170 108,148 Q118,125 122,105" stroke="#FFD700" strokeWidth="0.4" strokeOpacity="0.25"/>
        {/* Wing bone ribs */}
        <path d="M100,200 Q95,160 90,120 Q82,75 70,40"  stroke="#C9A227" strokeWidth="1" strokeOpacity="0.6"/>
        <path d="M100,200 Q102,155 108,115 Q115,80 125,50" stroke="#C9A227" strokeWidth="0.8" strokeOpacity="0.5"/>
      </svg>
    </div>
  )
}

/** Small dragon head icon (for pipeline / cards) */
export function DragonHeadIcon({ size = 24, color = '#C9A227' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M4,12 Q4,5 12,5 Q18,5 20,10 Q22,14 20,18 Q18,21 14,22 Q10,21 8,18 L4,19 L6,15 Q4,13 4,12Z"
        fill={color} fillOpacity="0.15" stroke={color} strokeWidth="1.2" strokeLinejoin="round"/>
      <circle cx="15" cy="10" r="1.5" fill="#FF6B00"/>
      <path d="M12,5 L14,2 M16,4 L17,1" stroke={color} strokeWidth="1" strokeLinecap="round"/>
    </svg>
  )
}
