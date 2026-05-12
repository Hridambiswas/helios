import { useState, useEffect } from 'react'
import { LogOut, Menu, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
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
    const handler = () => setScrolled(window.scrollY > 30)
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const navLinks = [
    { href: '#query-section',    label: 'Query'    },
    { href: '#pipeline-section', label: 'Pipeline' },
    { href: '#history-section',  label: 'History'  },
    { href: '#upload-section',   label: 'Ingest'   },
  ]

  const statusColor = apiStatus === 'ok' ? '#4ade80' : apiStatus === 'degraded' ? '#facc15' : 'rgba(255,255,255,0.3)'

  return (
    <motion.nav
      initial={{ y: -72, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 40,
        transition: 'background 0.4s, backdrop-filter 0.4s, border-color 0.4s',
        background:     scrolled ? 'rgba(0,0,0,0.72)' : 'transparent',
        backdropFilter: scrolled ? 'blur(24px) saturate(180%)' : 'none',
        borderBottom:   scrolled ? '1px solid rgba(255,255,255,0.06)' : '1px solid transparent',
      }}
    >
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 1.5rem', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>

        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <a href="#" style={{ textDecoration: 'none' }}>
            <span style={{ fontFamily: '"Montserrat", sans-serif', fontWeight: 900, fontSize: 17, letterSpacing: '-0.03em', color: '#fff' }}>
              HELIOS
            </span>
          </a>
          {apiStatus && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }} title={`API: ${apiStatus}`}>
              <div style={{ width: 5, height: 5, borderRadius: '50%', background: statusColor, boxShadow: `0 0 5px ${statusColor}` }} />
            </div>
          )}
        </div>

        {/* Desktop nav links */}
        <div className="hidden md:flex" style={{ gap: 32, alignItems: 'center' }}>
          {navLinks.map(({ href, label }) => (
            <a
              key={label}
              href={href}
              style={{
                fontFamily: '"IBM Plex Mono", monospace',
                fontSize: 10,
                letterSpacing: '0.2em',
                textTransform: 'uppercase',
                color: 'rgba(255,255,255,0.32)',
                textDecoration: 'none',
                transition: 'color 0.2s',
              }}
              onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.85)')}
              onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.32)')}
            >
              {label}
            </a>
          ))}
        </div>

        {/* Auth + mobile toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {user ? (
            <div className="hidden md:flex" style={{ alignItems: 'center', gap: 14 }}>
              <span style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: 10, color: 'rgba(255,255,255,0.28)' }}>
                {user.username}
              </span>
              <button
                onClick={onLogout}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                  letterSpacing: '0.2em', textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.28)', background: 'none', border: 'none',
                  cursor: 'none', transition: 'color 0.2s',
                }}
                onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.75)')}
                onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.28)')}
              >
                <LogOut size={11} /> Out
              </button>
            </div>
          ) : (
            <button
              onClick={onAuthClick}
              className="hidden md:block"
              style={{
                fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                letterSpacing: '0.22em', textTransform: 'uppercase',
                padding: '7px 18px',
                border: '1px solid rgba(255,255,255,0.14)',
                color: 'rgba(255,255,255,0.58)',
                background: 'rgba(255,255,255,0.03)',
                cursor: 'none',
                transition: 'border-color 0.2s, color 0.2s, background 0.2s',
              }}
              onMouseEnter={e => {
                const b = e.currentTarget as HTMLButtonElement
                b.style.borderColor = 'rgba(255,255,255,0.38)'
                b.style.color       = '#fff'
                b.style.background  = 'rgba(255,255,255,0.06)'
              }}
              onMouseLeave={e => {
                const b = e.currentTarget as HTMLButtonElement
                b.style.borderColor = 'rgba(255,255,255,0.14)'
                b.style.color       = 'rgba(255,255,255,0.58)'
                b.style.background  = 'rgba(255,255,255,0.03)'
              }}
            >
              Enter
            </button>
          )}

          <button
            className="md:hidden"
            onClick={() => setMobileOpen(o => !o)}
            style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)', cursor: 'none', transition: 'color 0.2s' }}
            onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.color = '#fff')}
            onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.4)')}
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22 }}
            style={{
              borderTop: '1px solid rgba(255,255,255,0.06)',
              background: 'rgba(0,0,0,0.92)',
              backdropFilter: 'blur(24px)',
            }}
            className="md:hidden"
          >
            {navLinks.map(({ href, label }) => (
              <a
                key={label}
                href={href}
                onClick={() => setMobileOpen(false)}
                style={{
                  display: 'block', padding: '14px 24px',
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                  letterSpacing: '0.25em', textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.32)',
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                  textDecoration: 'none', transition: 'color 0.2s',
                }}
                onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.color = '#fff')}
                onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.32)')}
              >
                {label}
              </a>
            ))}
            {user ? (
              <button
                onClick={() => { setMobileOpen(false); onLogout() }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  width: '100%', padding: '14px 24px',
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                  letterSpacing: '0.25em', textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.32)', background: 'none', border: 'none', cursor: 'none',
                }}
              >
                <LogOut size={12} /> Sign out
              </button>
            ) : (
              <button
                onClick={() => { setMobileOpen(false); onAuthClick() }}
                style={{
                  display: 'block', width: '100%', padding: '14px 24px', textAlign: 'left',
                  fontFamily: '"IBM Plex Mono", monospace', fontSize: 10,
                  letterSpacing: '0.25em', textTransform: 'uppercase',
                  color: 'rgba(255,255,255,0.55)', background: 'none', border: 'none', cursor: 'none',
                }}
              >
                Sign in
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}
