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
    <div
      className={`flex flex-col h-full border-r transition-all duration-300 shrink-0 ${collapsed ? 'w-12' : 'w-60'}`}
      style={{ background: '#050508', borderColor: 'rgba(255,255,255,0.06)' }}
    >
      {/* Header */}
      <div
        className={`flex items-center h-14 px-3 border-b shrink-0 ${collapsed ? 'justify-center' : 'justify-between'}`}
        style={{ borderColor: 'rgba(255,255,255,0.06)' }}
      >
        {!collapsed && (
          <div className="flex items-center gap-2">
            <span
              className="font-mono text-[11px] tracking-[0.3em] uppercase select-none"
              style={{ color: '#8b5cf6' }}
            >
              HELIOS
            </span>
          </div>
        )}
        <button
          onClick={onToggle}
          className="transition-colors"
          style={{ color: 'rgba(255,255,255,0.2)' }}
          onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.7)')}
          onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.2)')}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* New chat */}
      <div className="p-2.5" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <button
          onClick={onNew}
          className={`flex items-center gap-2.5 w-full px-3 py-2 font-mono text-[10px] tracking-wider uppercase transition-all ${collapsed ? 'justify-center' : ''}`}
          style={{
            border: '1px solid rgba(139,92,246,0.25)',
            color: 'rgba(139,92,246,0.8)',
            background: 'rgba(139,92,246,0.04)',
          }}
          onMouseEnter={e => {
            ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.5)'
            ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(139,92,246,0.09)'
            ;(e.currentTarget as HTMLButtonElement).style.color = '#a78bfa'
          }}
          onMouseLeave={e => {
            ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.25)'
            ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(139,92,246,0.04)'
            ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(139,92,246,0.8)'
          }}
        >
          <Plus size={13} />
          {!collapsed && 'New Chat'}
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-1">
        {!collapsed && conversations.length === 0 && (
          <div className="flex flex-col items-center justify-center gap-3 py-10 px-4">
            <MessageSquare size={20} style={{ color: 'rgba(255,255,255,0.08)' }} />
            <p className="font-mono text-[9px] text-center" style={{ color: 'rgba(255,255,255,0.2)', lineHeight: 1.6 }}>
              No conversations yet.<br />Ask anything to start.
            </p>
          </div>
        )}

        {conversations.map(c => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            className="group relative flex items-center gap-2.5 mx-1.5 my-0.5 px-2.5 py-2 cursor-pointer rounded-sm transition-colors"
            style={{
              background: c.id === activeId ? 'rgba(139,92,246,0.09)' : 'transparent',
              color: c.id === activeId ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.38)',
            }}
            onMouseEnter={e => {
              if (c.id !== activeId) (e.currentTarget as HTMLDivElement).style.background = 'rgba(255,255,255,0.03)'
            }}
            onMouseLeave={e => {
              if (c.id !== activeId) (e.currentTarget as HTMLDivElement).style.background = 'transparent'
            }}
          >
            {/* Active accent line */}
            {c.id === activeId && (
              <div
                className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-full"
                style={{ background: '#8b5cf6' }}
              />
            )}

            <MessageSquare size={11} className="shrink-0 opacity-40" />

            {!collapsed && (
              <>
                <span className="flex-1 font-mono text-[10px] truncate">{c.title}</span>
                <button
                  onClick={e => { e.stopPropagation(); onDelete(c.id) }}
                  className="opacity-0 group-hover:opacity-100 transition-all shrink-0 p-0.5 rounded"
                  style={{ color: 'rgba(255,255,255,0.3)' }}
                  onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.color = '#f87171')}
                  onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.3)')}
                >
                  <Trash2 size={10} />
                </button>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-2 space-y-0.5" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <SidebarBtn
          icon={<Upload size={13} />}
          label="Upload Docs"
          collapsed={collapsed}
          onClick={onUploadClick}
        />

        {user ? (
          <div className={`flex items-center gap-2 w-full px-3 py-2 ${collapsed ? 'justify-center' : ''}`}>
            {!collapsed && (
              <>
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center font-mono text-[9px] shrink-0"
                  style={{ background: 'rgba(139,92,246,0.2)', color: '#a78bfa', border: '1px solid rgba(139,92,246,0.3)' }}
                >
                  {user.username[0].toUpperCase()}
                </div>
                <span className="flex-1 font-mono text-[10px] truncate" style={{ color: 'rgba(255,255,255,0.35)' }}>
                  {user.username}
                </span>
              </>
            )}
            <button
              onClick={onLogout}
              title="Sign out"
              className="transition-colors p-0.5"
              style={{ color: 'rgba(255,255,255,0.2)' }}
              onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.color = '#f87171')}
              onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.2)')}
            >
              <LogOut size={12} />
            </button>
          </div>
        ) : (
          <SidebarBtn
            icon={<LogIn size={13} />}
            label="Sign in"
            collapsed={collapsed}
            onClick={onAuthClick}
          />
        )}
      </div>
    </div>
  )
}

function SidebarBtn({ icon, label, collapsed, onClick }: {
  icon: React.ReactNode
  label: string
  collapsed: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2.5 w-full px-3 py-2 font-mono text-[10px] transition-colors ${collapsed ? 'justify-center' : ''}`}
      style={{ color: 'rgba(255,255,255,0.25)' }}
      onMouseEnter={e => {
        ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.7)'
        ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.03)'
      }}
      onMouseLeave={e => {
        ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.25)'
        ;(e.currentTarget as HTMLButtonElement).style.background = 'transparent'
      }}
    >
      {icon}
      {!collapsed && label}
    </button>
  )
}
