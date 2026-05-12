import { useState, useEffect } from 'react'
import { LogOut, User, Menu, X } from 'lucide-react'
import type { User as UserType } from '../hooks/useAuth'
import { api } from '../api/client'

export function Navbar({ user, onAuthClick, onLogout }: {
  user: UserType | null
  onAuthClick: () => void
  onLogout: () => void
}) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [apiStatus,  setApiStatus]  = useState<'ok' | 'degraded' | 'down' | null>(null)
  const [scrolled,   setScrolled]   = useState(false)

  useEffect(() => {
    const check = () =>
      api.get('/health')
        .then(({ data }) => setApiStatus(data.status))
        .catch(() => setApiStatus('down'))
    check()
    const id = setInterval(check, 60_000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const navLinks = [
    { href: '#query-section',    label: 'QUERY'    },
    { href: '#pipeline-section', label: 'PIPELINE' },
    { href: '#history-section',  label: 'HISTORY'  },
    { href: '#upload-section',   label: 'INGEST'   },
  ]

  const statusColor =
    apiStatus === 'ok'       ? '#22c55e' :
    apiStatus === 'degraded' ? '#eab308' : '#8b5cf6'

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-40 transition-all duration-300"
      style={{
        background: scrolled ? 'rgba(0,0,0,0.92)' : 'transparent',
        backdropFilter: scrolled ? 'blur(16px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(139,92,246,0.1)' : '1px solid transparent',
      }}
    >
      <div className="max-w-6xl mx-auto px-4 h-12 flex items-center justify-between">

        {/* Logo */}
        <div className="flex items-center gap-3">
          {/* Minimal orb icon */}
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ opacity: 0.8 }}>
            <circle cx="7" cy="7" r="6" stroke="#8b5cf6" strokeWidth="1"/>
            <circle cx="7" cy="7" r="3" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="0.8"/>
            <circle cx="7" cy="7" r="1.2" fill="#a78bfa"/>
          </svg>
          <a
            href="#"
            className="font-display text-lg tracking-tight"
            style={{ fontFamily: 'Impact, Arial Black, sans-serif', textDecoration: 'none' }}
          >
            <span className="text-white">HEL</span>
            <span style={{ color: '#8b5cf6', textShadow: '0 0 16px rgba(139,92,246,0.6)' }}>IOS</span>
          </a>
          {apiStatus && (
            <div className="hidden sm:flex items-center gap-1" title={`API: ${apiStatus}`}>
              <div className="w-1.5 h-1.5 rounded-full" style={{ background: statusColor, boxShadow: `0 0 4px ${statusColor}` }} />
              <span className="font-mono text-[9px]" style={{ color: statusColor }}>{apiStatus.toUpperCase()}</span>
            </div>
          )}
        </div>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-6">
          {navLinks.map(({ href, label }) => (
            <a
              key={label}
              href={href}
              className="font-mono text-[10px] tracking-widest uppercase transition-all duration-200"
              style={{ color: 'rgba(255,255,255,0.3)', textDecoration: 'none' }}
              onMouseEnter={e => {
                e.currentTarget.style.color      = '#fff'
                e.currentTarget.style.textShadow = 'none'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.color      = 'rgba(255,255,255,0.3)'
                e.currentTarget.style.textShadow = 'none'
              }}
            >
              {label}
            </a>
          ))}
        </div>

        {/* Auth */}
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <div className="hidden md:flex items-center gap-2">
                <User size={12} style={{ color: 'rgba(139,92,246,0.6)' }} />
                <span className="font-mono text-[10px] max-w-[100px] truncate" style={{ color: 'rgba(255,255,255,0.4)' }}>
                  {user.username}
                </span>
              </div>
              <button
                onClick={onLogout}
                className="flex items-center gap-1.5 font-mono text-[10px] tracking-wider uppercase transition-colors"
                style={{ color: 'rgba(255,255,255,0.3)' }}
                onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.3)')}
              >
                <LogOut size={12} />
                <span className="hidden sm:inline">LOGOUT</span>
              </button>
            </>
          ) : (
            <button
              onClick={onAuthClick}
              className="font-mono text-[10px] tracking-widest uppercase px-3 py-1 transition-all duration-200"
              style={{
                color: '#8b5cf6',
                border: '1px solid rgba(139,92,246,0.4)',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background   = 'rgba(139,92,246,0.1)'
                e.currentTarget.style.borderColor  = 'rgba(139,92,246,0.8)'
                e.currentTarget.style.boxShadow    = '0 0 16px rgba(139,92,246,0.25)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background   = 'transparent'
                e.currentTarget.style.borderColor  = 'rgba(139,92,246,0.4)'
                e.currentTarget.style.boxShadow    = 'none'
              }}
            >
              ⟡ ENTER
            </button>
          )}

          <button
            className="md:hidden transition-colors"
            style={{ color: 'rgba(255,255,255,0.4)' }}
            onClick={() => setMobileOpen(o => !o)}
            onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
            onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.4)')}
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="md:hidden" style={{ borderTop: '1px solid rgba(139,92,246,0.1)', background: 'rgba(0,0,0,0.97)' }}>
          {navLinks.map(({ href, label }) => (
            <a
              key={label}
              href={href}
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-3 font-mono text-[10px] tracking-widest uppercase transition-all"
              style={{ color: 'rgba(255,255,255,0.35)', borderBottom: '1px solid rgba(139,92,246,0.06)', textDecoration: 'none' }}
              onMouseEnter={e => { e.currentTarget.style.color = '#fff'; e.currentTarget.style.background = 'rgba(139,92,246,0.05)' }}
              onMouseLeave={e => { e.currentTarget.style.color = 'rgba(255,255,255,0.35)'; e.currentTarget.style.background = 'transparent' }}
            >
              ⟡ {label}
            </a>
          ))}
          {user && (
            <button
              onClick={() => { setMobileOpen(false); onLogout() }}
              className="w-full text-left px-4 py-3 font-mono text-[10px] uppercase flex items-center gap-2 transition-colors"
              style={{ color: 'rgba(255,255,255,0.35)' }}
            >
              <LogOut size={12} /> LOGOUT
            </button>
          )}
        </div>
      )}
    </nav>
  )
}
