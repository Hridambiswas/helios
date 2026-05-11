import { useState } from 'react'
import { X, Eye, EyeOff } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL ?? 'https://helios-hridam.ddns.net'

type Props = {
  onClose: () => void
  onLogin: (u: string, p: string) => Promise<void>
  onRegister: (u: string, e: string, p: string) => Promise<void>
}

export function AuthModal({ onClose, onLogin, onRegister }: Props) {
  const [tab, setTab] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    if (!username.trim() || !password.trim()) {
      setError('Username and password are required')
      return
    }
    setError('')
    setLoading(true)
    try {
      if (tab === 'login') {
        await onLogin(username.trim(), password)
      } else {
        await onRegister(username.trim(), email.trim(), password)
      }
      onClose()
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string | { msg: string }[] } } })?.response?.data?.detail
      if (Array.isArray(detail)) {
        setError(detail.map(d => d.msg).join('; '))
      } else {
        setError(detail ?? 'Something went wrong — check your connection')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') submit()
    if (e.key === 'Escape') onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="card-dark w-full max-w-md mx-4 border border-crimson/30 relative">
        <div className="h-0.5 w-full bg-gradient-to-r from-crimson via-crimson-light to-crimson-dark" />

        <div className="p-8">
          <button onClick={onClose} className="absolute top-4 right-4 text-[#555] hover:text-white transition-colors">
            <X size={18} />
          </button>

          {/* Tabs */}
          <div className="flex gap-6 mb-8 border-b border-white/10 pb-4">
            {(['login', 'register'] as const).map(t => (
              <button
                key={t}
                onClick={() => { setTab(t); setError('') }}
                className={`font-mono text-xs tracking-widest uppercase transition-colors ${
                  tab === t ? 'text-crimson' : 'text-[#555] hover:text-white'
                }`}
              >
                {t === 'login' ? 'SIGN IN' : 'REGISTER'}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            <Field label="USERNAME" value={username} onChange={setUsername} onKeyDown={handleKey}
              autoComplete={tab === 'login' ? 'username' : 'new-password'} />
            {tab === 'register' && (
              <Field label="EMAIL" value={email} onChange={setEmail} type="email" onKeyDown={handleKey}
                autoComplete="email" />
            )}
            <div>
              <label className="block font-mono text-[10px] tracking-widest uppercase text-[#555] mb-1.5">PASSWORD</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onKeyDown={handleKey}
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  className="w-full bg-white/5 border border-white/10 text-white font-mono text-sm px-3 py-2.5 pr-10 outline-none input-red transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#555] hover:text-white transition-colors"
                >
                  {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-crimson/10 border border-crimson/30 text-crimson text-xs font-mono">
              {error}
            </div>
          )}

          <button
            onClick={submit}
            disabled={loading}
            className="w-full mt-6 py-3 bg-crimson hover:bg-crimson-light disabled:opacity-50 transition-colors text-white font-mono text-xs tracking-widest uppercase"
          >
            {loading ? 'PROCESSING...' : tab === 'login' ? 'AUTHENTICATE' : 'CREATE ACCOUNT'}
          </button>

          {/* OAuth divider */}
          <div className="flex items-center gap-3 my-5">
            <div className="flex-1 h-px bg-white/10" />
            <span className="font-mono text-[9px] tracking-widest text-[#444] uppercase">or continue with</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* Social buttons */}
          <a
            href={`${API_BASE}/api/v1/auth/github`}
            className="w-full flex items-center justify-center gap-2 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all text-white text-xs font-mono tracking-wider"
          >
            <GitHubIcon />
            Continue with GitHub
          </a>

          {tab === 'login' && (
            <p className="mt-4 text-center font-mono text-[10px] text-[#444]">
              No account?{' '}
              <button onClick={() => setTab('register')} className="text-crimson hover:text-crimson-light">
                Register
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function GitHubIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
    </svg>
  )
}

function Field({ label, value, onChange, type = 'text', onKeyDown, autoComplete }: {
  label: string
  value: string
  onChange: (v: string) => void
  type?: string
  onKeyDown?: (e: React.KeyboardEvent) => void
  autoComplete?: string
}) {
  return (
    <div>
      <label className="block font-mono text-[10px] tracking-widest uppercase text-[#555] mb-1.5">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        autoComplete={autoComplete}
        className="w-full bg-white/5 border border-white/10 text-white font-mono text-sm px-3 py-2.5 outline-none input-red transition-all"
      />
    </div>
  )
}
