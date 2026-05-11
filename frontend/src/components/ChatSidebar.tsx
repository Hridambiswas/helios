import { Plus, MessageSquare, Trash2, ChevronLeft, ChevronRight, Upload, LogOut, LogIn } from 'lucide-react'
import type { Conversation } from '../hooks/useConversations'

type Props = {
  conversations: Conversation[]
  activeId: string | null
  onSelect: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  collapsed: boolean
  onToggle: () => void
  user: { username: string } | null
  onAuthClick: () => void
  onLogout: () => void
  onUploadClick: () => void
}

export function ChatSidebar({ conversations, activeId, onSelect, onNew, onDelete, collapsed, onToggle, user, onAuthClick, onLogout, onUploadClick }: Props) {
  return (
    <div className={`flex flex-col h-full bg-[#0d0d0d] border-r border-white/8 transition-all duration-300 shrink-0 ${collapsed ? 'w-12' : 'w-64'}`}>

      {/* Top: logo + toggle */}
      <div className={`flex items-center border-b border-white/8 h-14 px-3 ${collapsed ? 'justify-center' : 'justify-between'}`}>
        {!collapsed && (
          <span className="font-mono text-xs tracking-[0.25em] uppercase text-crimson select-none">HELIOS</span>
        )}
        <button onClick={onToggle} className="text-[#555] hover:text-white transition-colors">
          {collapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
        </button>
      </div>

      {/* New chat button */}
      <div className="p-2 border-b border-white/5">
        <button
          onClick={onNew}
          className={`flex items-center gap-2 w-full px-3 py-2 border border-crimson/30 hover:border-crimson hover:bg-crimson/8 text-crimson transition-all font-mono text-xs ${collapsed ? 'justify-center' : ''}`}
        >
          <Plus size={14} />
          {!collapsed && 'New Chat'}
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-1">
        {!collapsed && conversations.length === 0 && (
          <p className="font-mono text-[10px] text-[#333] px-3 py-4 text-center">No conversations yet</p>
        )}
        {conversations.map(c => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`group relative flex items-center gap-2 px-3 py-2.5 cursor-pointer transition-colors ${
              c.id === activeId ? 'bg-crimson/10 text-white' : 'text-[#666] hover:bg-white/4 hover:text-[#ccc]'
            }`}
          >
            <MessageSquare size={12} className="shrink-0 opacity-60" />
            {!collapsed && (
              <>
                <span className="flex-1 font-mono text-[11px] truncate">{c.title}</span>
                <button
                  onClick={e => { e.stopPropagation(); onDelete(c.id) }}
                  className="opacity-0 group-hover:opacity-100 text-[#444] hover:text-crimson transition-all shrink-0"
                >
                  <Trash2 size={11} />
                </button>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Bottom actions */}
      <div className="border-t border-white/8 p-2 space-y-1">
        <button
          onClick={onUploadClick}
          className={`flex items-center gap-2 w-full px-3 py-2 text-[#555] hover:text-white hover:bg-white/5 transition-colors font-mono text-xs ${collapsed ? 'justify-center' : ''}`}
        >
          <Upload size={13} />
          {!collapsed && 'Upload Docs'}
        </button>

        {user ? (
          <button
            onClick={onLogout}
            className={`flex items-center gap-2 w-full px-3 py-2 text-[#555] hover:text-crimson hover:bg-crimson/5 transition-colors font-mono text-xs ${collapsed ? 'justify-center' : ''}`}
          >
            <LogOut size={13} />
            {!collapsed && <span className="truncate">Sign out ({user.username})</span>}
          </button>
        ) : (
          <button
            onClick={onAuthClick}
            className={`flex items-center gap-2 w-full px-3 py-2 text-[#555] hover:text-white hover:bg-white/5 transition-colors font-mono text-xs ${collapsed ? 'justify-center' : ''}`}
          >
            <LogIn size={13} />
            {!collapsed && 'Sign in'}
          </button>
        )}
      </div>
    </div>
  )
}
