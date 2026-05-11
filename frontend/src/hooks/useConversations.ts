import { useState, useCallback } from 'react'
import type { QueryResponse } from '../api/client'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  result?: QueryResponse
  timestamp: number
  error?: string
  step?: string
}

export type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

const KEY = 'helios_conversations'
const MAX = 50

function load(): Conversation[] {
  try { return JSON.parse(localStorage.getItem(KEY) ?? '[]') } catch { return [] }
}

function persist(c: Conversation[]) {
  localStorage.setItem(KEY, JSON.stringify(c.slice(0, MAX)))
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(load)
  const [activeId, setActiveId] = useState<string | null>(() => load()[0]?.id ?? null)

  const active = conversations.find(c => c.id === activeId) ?? null

  const update = useCallback((next: Conversation[]) => {
    setConversations(next)
    persist(next)
  }, [])

  const newConversation = useCallback((): string => {
    const id = crypto.randomUUID()
    const c: Conversation = { id, title: 'New Chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() }
    setConversations(prev => { const n = [c, ...prev]; persist(n); return n })
    setActiveId(id)
    return id
  }, [])

  const selectConversation = useCallback((id: string) => {
    setActiveId(id)
  }, [])

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id)
      persist(next)
      setActiveId(cur => cur === id ? (next[0]?.id ?? null) : cur)
      return next
    })
  }, [])

  const addUserMessage = useCallback((convId: string, content: string): string => {
    const msgId = crypto.randomUUID()
    setConversations(prev => {
      const next = prev.map(c => {
        if (c.id !== convId) return c
        const isFirst = c.messages.length === 0
        const msg: ChatMessage = { id: msgId, role: 'user', content, timestamp: Date.now() }
        return { ...c, title: isFirst ? content.slice(0, 55) : c.title, messages: [...c.messages, msg], updatedAt: Date.now() }
      })
      persist(next)
      return next
    })
    return msgId
  }, [])

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
      return next
    })
  }, [])

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
