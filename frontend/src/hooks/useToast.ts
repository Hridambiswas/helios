import { useState, useCallback } from 'react'

export type Toast = { id: string; message: string; type: 'success' | 'error' | 'info' }

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([])

  const add = useCallback((message: string, type: Toast['type'] = 'info', duration = 3500) => {
    const id = Math.random().toString(36).slice(2)
    setToasts(t => [...t, { id, message, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), duration)
  }, [])

  const remove = useCallback((id: string) => {
    setToasts(t => t.filter(x => x.id !== id))
  }, [])

  return { toasts, add, remove }
}
