import { useState } from 'react'
import { X, Eye, EyeOff } from 'lucide-react'
import { BASE as API_BASE } from '../api/client'

type Props = {
  onClose: () => void
  onLogin: (u: string, p: string) => Promise<void>
  onRegister: (u: string, e: string, p: string) => Promise<void>
}

/**
 * AuthModal — winter-sun repaint. Same UX as before (tabbed login /
 * register + OAuth), typography and palette rebuilt on --frost with
 * Atkinson Hyperlegible text. No all-caps labels, no letter-spaced
 * eyebrows (both violated the brief).
 */

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
      setError('Username and password are required.')
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
        setError(detail ?? "Can't reach the Helios API right now. Try again in a minute.")
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
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(20, 29, 51, 0.75)',
        backdropFilter: 'blur(8px)',
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={tab === 'login' ? 'Sign in' : 'Register'}
        style={{
          width: '100%', maxWidth: 440, margin: '0 16px',
          background: 'var(--frost)',
          border: '1px solid var(--frost-hairline)',
          borderRadius: 'var(--radius-lg)',
          position: 'relative',
        }}
      >
        <button
          onClick={onClose}
          aria-label="Close"
          className="helios-focus"
          style={{
            position: 'absolute', top: 14, right: 14,
            background: 'transparent', border: 'none',
            color: 'var(--snow-shadow)', cursor: 'pointer',
            padding: 6, borderRadius: 'var(--radius-sm)',
          }}
        >
          <X size={16} />
        </button>

        <div style={{ padding: '28px 28px 32px' }}>
          <h2 className="display" style={{ color: 'var(--snow)', fontSize: 'var(--step-2)', marginBottom: 16 }}>
            {tab === 'login' ? 'Sign in' : 'Create account'}
          </h2>

          <div style={{
            display: 'flex', gap: 4,
            marginBottom: 22,
            background: 'var(--polar-night)',
            borderRadius: 'var(--radius-md)',
            padding: 4,
            border: '1px solid var(--frost-hairline)',
          }}>
            {(['login', 'register'] as const).map(t => (
              <button
                key={t}
                onClick={() => { setTab(t); setError('') }}
                className="text helios-focus"
                aria-pressed={tab === t}
                style={{
                  flex: 1,
                  padding: '8px 12px',
                  background: tab === t ? 'var(--frost)' : 'transparent',
                  border: 'none',
                  color: tab === t ? 'var(--snow)' : 'var(--snow-shadow)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  fontSize: 'var(--step--1)',
                }}
              >
                {t === 'login' ? 'Sign in' : 'Register'}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Field label="Username" value={username} onChange={setUsername} onKeyDown={handleKey}
              autoComplete={tab === 'login' ? 'username' : 'new-password'} />
            {tab === 'register' && (
              <Field label="Email" value={email} onChange={setEmail} type="email" onKeyDown={handleKey}
                autoComplete="email" />
            )}
            <div>
              <label className="text--meta" htmlFor="helios-password" style={{ display: 'block', marginBottom: 6 }}>
                Password
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  id="helios-password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onKeyDown={handleKey}
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  className="text helios-focus"
                  style={{
                    width: '100%',
                    background: 'var(--polar-night)',
                    border: '1px solid var(--frost-hairline)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--snow)',
                    padding: '10px 40px 10px 12px',
                    fontSize: 'var(--step-0)',
                    outline: 'none',
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                    background: 'transparent', border: 'none',
                    color: 'var(--snow-shadow)', cursor: 'pointer',
                  }}
                >
                  {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>
          </div>

          {error && (
            <div
              role="alert"
              className="text"
              style={{
                marginTop: 14,
                padding: '10px 12px',
                background: 'var(--corona-soft)',
                border: '1px solid var(--corona)',
                color: 'var(--snow)',
                borderRadius: 'var(--radius-sm)',
                fontSize: 'var(--step--1)',
              }}
            >
              {error}
            </div>
          )}

          <button
            onClick={submit}
            disabled={loading}
            className="text helios-focus"
            style={{
              width: '100%', marginTop: 22,
              padding: '12px 16px',
              background: 'var(--sun)',
              border: 'none', borderRadius: 'var(--radius-md)',
              color: '#1B1305',
              fontSize: 'var(--step-0)', fontWeight: 600,
              cursor: loading ? 'wait' : 'pointer',
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? 'Working…' : tab === 'login' ? 'Sign in' : 'Create account'}
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '22px 0 16px' }}>
            <div style={{ flex: 1, height: 1, background: 'var(--frost-hairline)' }} />
            <span className="text--meta">or continue with</span>
            <div style={{ flex: 1, height: 1, background: 'var(--frost-hairline)' }} />
          </div>

          <a
            href={`${API_BASE}/api/v1/auth/github`}
            className="text helios-focus"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '10px 12px',
              background: 'var(--polar-night)',
              border: '1px solid var(--frost-hairline)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--snow)',
              fontSize: 'var(--step-0)',
              textDecoration: 'none',
            }}
          >
            <GitHubIcon />
            Continue with GitHub
          </a>

          {tab === 'login' && (
            <p className="text--meta" style={{ marginTop: 14, textAlign: 'center' }}>
              No account?{' '}
              <button
                onClick={() => setTab('register')}
                className="helios-focus"
                style={{ background: 'transparent', border: 'none', color: 'var(--sun)', cursor: 'pointer', padding: 0 }}
              >
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
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
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
  const id = `helios-field-${label.toLowerCase()}`
  return (
    <div>
      <label htmlFor={id} className="text--meta" style={{ display: 'block', marginBottom: 6 }}>
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        autoComplete={autoComplete}
        className="text helios-focus"
        style={{
          width: '100%',
          background: 'var(--polar-night)',
          border: '1px solid var(--frost-hairline)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--snow)',
          padding: '10px 12px',
          fontSize: 'var(--step-0)',
          outline: 'none',
        }}
      />
    </div>
  )
}
