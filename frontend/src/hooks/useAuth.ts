import { useState, useEffect, useCallback } from 'react'
import { auth } from '../api/client'

export type User = { id: string; username: string; email: string; is_active: boolean }

function _isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    return payload.exp * 1000 < Date.now()
  } catch {
    return true
  }
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchMe = useCallback(async () => {
    const token = localStorage.getItem('access_token')
    if (!token || _isTokenExpired(token)) { setLoading(false); return }
    try {
      const { data } = await auth.me()
      setUser(data)
    } catch {
      localStorage.clear()
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchMe()
    const handler = () => setUser(null)
    window.addEventListener('helios:logout', handler)
    return () => window.removeEventListener('helios:logout', handler)
  }, [fetchMe])

  const login = async (username: string, password: string) => {
    const { data } = await auth.login(username.trim().toLowerCase(), password)
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)
    await fetchMe()
  }

  const register = async (username: string, email: string, password: string) => {
    const { data } = await auth.register(username.trim().toLowerCase(), email.trim().toLowerCase(), password)
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)
    await fetchMe()
  }

  const logout = async () => {
    const refresh = localStorage.getItem('refresh_token')
    if (refresh) { try { await auth.logout(refresh) } catch { /* ignore */ } }
    localStorage.clear()
    setUser(null)
  }

  return { user, loading, login, register, logout }
}
