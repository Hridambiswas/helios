import { useState } from 'react'
import { LogOut, User, Menu, X } from 'lucide-react'
import type { User as UserType } from '../hooks/useAuth'

export function Navbar({ user, onAuthClick, onLogout }: {
  user: UserType | null
  onAuthClick: () => void
  onLogout: () => void
}) {
  const [mobileOpen, setMobileOpen] = useState(false)

  const navLinks = [
    { href: '#query-section', label: 'QUERY' },
    { href: '#pipeline-section', label: 'PIPELINE' },
    { href: '#history-section', label: 'HISTORY' },
    { href: '#upload-section', label: 'INGEST' },
  ]

  return (
    <nav className="fixed top-0 left-0 right-0 z-40 bg-ink/80 backdrop-blur-md border-b border-white/5">
      <div className="max-w-6xl mx-auto px-4 h-12 flex items-center justify-between">
        {/* Logo */}
        <a href="#" className="font-display text-white text-lg tracking-tight"
          style={{ fontFamily: 'Impact, Arial Black, sans-serif' }}>
          HEL<span className="text-crimson">IOS</span>
        </a>

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
                <span className="font-mono text-[10px] text-[#555]">{user.username}</span>
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
        </div>
      )}
    </nav>
  )
}
