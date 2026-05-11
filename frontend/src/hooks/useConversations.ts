import { useState, useCallback, useEffect, useRef } from 'react'
import type { QueryResponse } from '../api/client'
import { conversations as convApi } from '../api/client'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  result?: QueryResponse
  timestamp: number
  error?: string
  step?: string
  serverId?: string  // id from server-persisted message
}

export type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
  serverId?: string  // id of the corresponding server-side conversation
}

const KEY = 'helios_conversations'
const MAX = 50

function load(): Conversation[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? '[]') } catch { return [] }
}

function persist(c: Conversation[]) {
  localStorage.setItem(KEY, JSON.stringify(c.slice(0, MAX)))
}

export function useConversations(isLoggedIn: boolean) {
  const [conversations, setConversations] = useState<Conversation[]>(load)
  const [activeId, setActiveId] = useState<string | null>(() => load()[0]?.id ?? null)
  const syncedRef = useRef(false)

  const active = conversations.find(c => c.id === activeId) ?? null

  const update = useCallback((next: Conversation[]) => {
    setConversations(next)
    persist(next)
  }, [])

  // Sync from server when user logs in
  useEffect(() => {
    if (!isLoggedIn || syncedRef.current) return
    syncedRef.current = true
    convApi.list(50).then(({ data: serverConvs }) => {
      setConversations(prev => {
        // Merge: keep local-only conversations, add server ones not already present
        const serverIds = new Set(serverConvs.map(c => c.id))
        const localOnly = prev.filter(c => !c.serverId && !serverIds.has(c.id))
        const fromServer: Conversation[] = serverConvs.map(sc => {
          const existing = prev.find(c => c.serverId === sc.id || c.id === sc.id)
          return existing ? { ...existing, serverId: sc.id, title: sc.title } : {
            id: sc.id, serverId: sc.id, title: sc.title,
            messages: [], createdAt: sc.created_at ? new Date(sc.created_at).getTime() : Date.now(),
            updatedAt: sc.updated_at ? new Date(sc.updated_at).getTime() : Date.now(),
          }
        })
        const merged = [...fromServer, ...localOnly]
          .sort((a, b) => b.updatedAt - a.updatedAt)
        persist(merged)
        return merged
      })
    }).catch(() => { /* network unavailable — keep local */ })
  }, [isLoggedIn])

  // Reset sync flag on logout so next login triggers a fresh sync
  useEffect(() => {
    if (!isLoggedIn) syncedRef.current = false
  }, [isLoggedIn])

  const newConversation = useCallback((): string => {
    const id = crypto.randomUUID()
    const c: Conversation = { id, title: 'New Chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() }

    if (isLoggedIn) {
      convApi.create('New Chat').then(({ data }) => {
        setConversations(prev => {
          const next = prev.map(cv => cv.id === id ? { ...cv, serverId: data.id } : cv)
          persist(next)
          return next
        })
      }).catch(() => {})
    }

    setConversations(prev => { const n = [c, ...prev]; persist(n); return n })
    setActiveId(id)
    return id
  }, [isLoggedIn])

  const selectConversation = useCallback((id: string) => {
    setActiveId(id)
    // Lazy-load messages from server if needed
    if (isLoggedIn) {
      setConversations(prev => {
        const conv = prev.find(c => c.id === id)
        if (!conv || conv.messages.length > 0) return prev
        const serverId = conv.serverId ?? conv.id
        convApi.get(serverId).then(({ data }) => {
          setConversations(p => {
            const cur = p.find(c => c.id === id)
            const serverMsgIds = new Set(data.messages.map(m => m.id))
            // Keep local messages that haven't been persisted to server yet
            const localPending = (cur?.messages ?? []).filter(m => !m.serverId && !serverMsgIds.has(m.id))
            const serverMessages = data.messages.map(m => ({
              id: m.id, serverId: m.id, role: m.role as 'user' | 'assistant',
              content: m.content, timestamp: new Date(m.created_at).getTime(),
            }))
            const next = p.map(c => c.id === id ? {
              ...c,
              messages: [...serverMessages, ...localPending],
              title: data.title,
            } : c)
            persist(next)
            return next
          })
        }).catch(() => {})
        return prev
      })
    }
  }, [isLoggedIn])

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => {
      const conv = prev.find(c => c.id === id)
      if (isLoggedIn && conv?.serverId) {
        convApi.delete(conv.serverId).catch(() => {})
      }
      const next = prev.filter(c => c.id !== id)
      persist(next)
      setActiveId(cur => cur === id ? (next[0]?.id ?? null) : cur)
      return next
    })
  }, [isLoggedIn])

  const addUserMessage = useCallback((convId: string, content: string): string => {
    const msgId = crypto.randomUUID()
    setConversations(prev => {
      const next = prev.map(c => {
        if (c.id !== convId) return c
        const isFirst = c.messages.length === 0
        const msg: ChatMessage = { id: msgId, role: 'user', content, timestamp: Date.now() }
        const updated = { ...c, title: isFirst ? content.slice(0, 55) : c.title, messages: [...c.messages, msg], updatedAt: Date.now() }
        if (isLoggedIn) {
          const serverId = c.serverId ?? c.id
          convApi.addMessage(serverId, 'user', content).then(({ data: sm }) => {
            setConversations(p => {
              const n = p.map(cv => cv.id !== convId ? cv : {
                ...cv, messages: cv.messages.map(m => m.id === msgId ? { ...m, serverId: sm.id } : m)
              })
              persist(n)
              return n
            })
          }).catch(() => {})
        }
        return updated
      })
      persist(next)
      return next
    })
    return msgId
  }, [isLoggedIn])

  const addAssistantPlaceholder = useCallback((convId: string): string => {
    const msgId = crypto.randomUUID()
    setConversations(prev => {
      const next = prev.map(c => {
        if (c.id !== convId) return c
        const msg: ChatMessage = { id: msgId, role: 'assistant', content: '', timestamp: Date.now(), step: 'planning' }
        return { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
      })
      persist(next)
      return next
    })
    return msgId
  }, [])

  const updateMessage = useCallback((convId: string, msgId: string, patch: Partial<ChatMessage>) => {
    setConversations(prev => {
      const next = prev.map(c => {
        if (c.id !== convId) return c
        return { ...c, messages: c.messages.map(m => m.id === msgId ? { ...m, ...patch } : m), updatedAt: Date.now() }
      })
      persist(next)
      // Persist completed assistant message to server
      if (patch.step === 'done' && patch.content && isLoggedIn) {
        const conv = next.find(c => c.id === convId)
        const serverId = conv?.serverId ?? convId
        convApi.addMessage(serverId, 'assistant', patch.content).catch(() => {})
      }
      return next
    })
  }, [isLoggedIn])

  return {
    conversations,
    active,
    activeId,
    selectConversation,
    newConversation,
    deleteConversation,
    addUserMessage,
    addAssistantPlaceholder,
    updateMessage,
    update,
  }
}
