import { Home, MessageSquare, Upload, LogIn, LogOut } from 'lucide-react'

type Props = {
  chatMode: boolean
  onHome: () => void
  onChat: () => void
  onUpload: () => void
  user: { username: string } | null
  onAuthClick: () => void
  onLogout: () => void
}

export function MobileBottomNav({ chatMode, onHome, onChat, onUpload, user, onAuthClick, onLogout }: Props) {
  const btn = (icon: React.ReactNode, label: string, onClick: () => void, active = false) => (
    <button
      onClick={onClick}
      className="text--meta helios-focus"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 4,
        flex: 1,
        padding: '10px 4px',
        background: 'transparent',
        border: 'none',
        color: active ? 'var(--sun)' : 'var(--snow-shadow)',
        cursor: 'pointer',
        transition: 'color 0.2s',
        fontSize: 10,
      }}
    >
      {icon}
      <span style={{ letterSpacing: 0.3 }}>{label}</span>
    </button>
  )

  return (
    <nav
      aria-label="Primary"
      className="mobile-bottom-nav"
      style={{
        position: 'fixed', bottom: 0, left: 0, right: 0,
        zIndex: 40,
        display: 'flex', alignItems: 'center',
        background: 'var(--frost)',
        borderTop: '1px solid var(--frost-hairline)',
        paddingBottom: 'env(safe-area-inset-bottom)',
      }}
    >
      {btn(<Home size={16} />, 'Home', onHome, !chatMode)}
      {btn(<MessageSquare size={16} />, 'Chat', onChat, chatMode)}
      {btn(<Upload size={16} />, 'Upload', onUpload)}
      {user
        ? btn(<LogOut size={16} />, 'Sign out', onLogout)
        : btn(<LogIn size={16} />, 'Sign in', onAuthClick)}
    </nav>
  )
}
