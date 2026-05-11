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
      className={`flex flex-col items-center gap-0.5 flex-1 py-2 transition-colors ${
        active ? 'text-crimson' : 'text-[#555] hover:text-white'
      }`}
    >
      {icon}
      <span className="font-mono text-[8px] tracking-wider uppercase">{label}</span>
    </button>
  )

  return (
    <div className="bottom-nav sm:hidden fixed bottom-0 left-0 right-0 z-40 flex items-center bg-[#0d0d0d] border-t border-white/8">
      {btn(<Home size={16} />, 'Home', onHome, !chatMode)}
      {btn(<MessageSquare size={16} />, 'Chat', onChat, chatMode)}
      {btn(<Upload size={16} />, 'Upload', onUpload)}
      {user
        ? btn(<LogOut size={16} />, 'Sign out', onLogout)
        : btn(<LogIn size={16} />, 'Sign in', onAuthClick)}
    </div>
  )
}
