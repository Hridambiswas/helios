import { useState, useEffect } from 'react'
import { LogOut, User, Menu, X, Circle } from 'lucide-react'
import type { User as UserType } from '../hooks/useAuth'
import { api } from '../api/client'

export function Navbar({ user, onAuthClick, onLogout }: {
  user: UserType | null
  onAuthClick: () => void
  onLogout: () => void
}) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [apiStatus, setApiStatus] = useState<'ok' | 'degraded' | 'down' | null>(null)

  useEffect(() => {
    api.get('/health').then(({ data }) => setApiStatus(data.status)).catch(() => setApiStatus('down'))
  }, [])

  const navLinks = [
    { href: '#query-section', label: 'QUERY' },
    { href: '#pipeline-section', label: 'PIPELINE' },
    { href: '#history-section', label: 'HISTORY' },
    { href: '#upload-section', label: 'INGEST' },
  ]

  const statusColor = apiStatus === 'ok' ? 'text-green-500' : apiStatus === 'degraded' ? 'text-yellow-500' : 'text-crimson'

  return (
    <nav className="fixed top-0 left-0 right-0 z-40 bg-ink/80 backdrop-blur-md border-b border-white/5">
      <div className="max-w-6xl mx-auto px-4 h-12 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <a href="#" className="font-display text-white text-lg tracking-tight"
            style={{ fontFamily: 'Impact, Arial Black, sans-serif' }}>
            HEL<span className="text-crimson">IOS</span>
          </a>
          {apiStatus && (
            <div className="hidden sm:flex items-center gap-1" title={`API: ${apiStatus}`}>
              <Circle size={6} className={`fill-current ${statusColor}`} />
              <span className={`font-mono text-[9px] ${statusColor}`}>{apiStatus.toUpperCase()}</span>
            </div>
          )}
        </div>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-6">
          {navLinks.map(({ href, label }) => (
            <a key={label} href={href}
              className="font-mono text-[10px] tracking-widest text-[#555] hover:text-crimson transition-colors uppercase">
              {label}
            </a>
          ))}
        </div>

        {/* Auth */}
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <div className="hidden md:flex items-center gap-2">
                <User size={12} className="text-[#555]" />
                <span className="font-mono text-[10px] text-[#555] max-w-[100px] truncate">{user.username}</span>
              </div>
              <button onClick={onLogout}
                className="flex items-center gap-1.5 font-mono text-[10px] text-[#555] hover:text-crimson transition-colors tracking-wider uppercase">
                <LogOut size={12} />
                <span className="hidden sm:inline">LOGOUT</span>
              </button>
            </>
          ) : (
            <button onClick={onAuthClick}
              className="font-mono text-[10px] tracking-widest text-crimson hover:text-crimson-light border border-crimson/30 hover:border-crimson/60 px-3 py-1 transition-all uppercase">
              SIGN IN
            </button>
          )}

          {/* Mobile menu */}
          <button className="md:hidden text-[#555] hover:text-white transition-colors"
            onClick={() => setMobileOpen(o => !o)}>
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="md:hidden border-t border-white/5 bg-ink/95">
          {navLinks.map(({ href, label }) => (
            <a key={label} href={href}
              onClick={() => setMobileOpen(false)}
              className="block px-4 py-3 font-mono text-[10px] tracking-widest text-[#555] hover:text-crimson hover:bg-crimson/5 transition-all uppercase border-b border-white/5">
              {label}
            </a>
          ))}
          {user && (
            <button onClick={() => { setMobileOpen(false); onLogout() }}
              className="w-full text-left px-4 py-3 font-mono text-[10px] text-[#555] hover:text-crimson hover:bg-crimson/5 transition-all uppercase flex items-center gap-2">
              <LogOut size={12} /> LOGOUT
            </button>
          )}
        </div>
      )}
    </nav>
  )
}
