import { useState } from 'react'
import { X } from 'lucide-react'

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
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    setError('')
    setLoading(true)
    try {
      if (tab === 'login') {
        await onLogin(username, password)
      } else {
        await onRegister(username, email, password)
      }
      onClose()
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="card-dark w-full max-w-md mx-4 border border-crimson/30 relative">
        {/* Red top bar */}
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
                onClick={() => setTab(t)}
                className={`font-mono text-xs tracking-widest uppercase transition-colors ${
                  tab === t ? 'text-crimson' : 'text-[#555] hover:text-white'
                }`}
              >
                {t === 'login' ? 'SIGN IN' : 'REGISTER'}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            <Field label="USERNAME" value={username} onChange={setUsername} />
            {tab === 'register' && (
              <Field label="EMAIL" value={email} onChange={setEmail} type="email" />
            )}
            <Field label="PASSWORD" value={password} onChange={setPassword} type="password" />
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
        </div>
      </div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'text' }: {
  label: string; value: string; onChange: (v: string) => void; type?: string
}) {
  return (
    <div>
      <label className="block font-mono text-[10px] tracking-widest uppercase text-[#555] mb-1.5">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-white/5 border border-white/10 text-white font-mono text-sm px-3 py-2.5 outline-none input-red transition-all"
      />
    </div>
  )
}
