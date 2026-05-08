import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL ?? ''

export const api = axios.create({
  baseURL: `${BASE}/api/v1`,
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT from localStorage on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-refresh on 401
api.interceptors.response.use(
  (r) => r,
  async (err) => {
    if (err.response?.status === 401 && !err.config._retry) {
      err.config._retry = true
      const refresh = localStorage.getItem('refresh_token')
      if (refresh) {
        try {
          const { data } = await axios.post(`${BASE}/api/v1/auth/refresh`, { refresh_token: refresh })
          localStorage.setItem('access_token', data.access_token)
          localStorage.setItem('refresh_token', data.refresh_token)
          err.config.headers.Authorization = `Bearer ${data.access_token}`
          return api(err.config)
        } catch {
          localStorage.clear()
          window.dispatchEvent(new Event('helios:logout'))
        }
      }
    }
    return Promise.reject(err)
  }
)

export type QueryResponse = {
  query_id: string
  query: string
  answer: string
  plan: { query_type: string; subtasks: { id: number; type: string; description: string }[] } | null
  retrieved_docs: { id: string; document: string; metadata: Record<string, unknown>; score: number; source: string }[]
  execution_result: { stdout: string; stderr: string; success: boolean } | null
  critic_scores: { groundedness: number; faithfulness: number; completeness: number; overall: number; pass: boolean; reasoning: string } | null
  critic_passed: boolean | null
  latency_ms: number
  status: string
}

export type HistoryItem = {
  id: string
  query_text: string
  answer: string | null
  status: string
  latency_ms: number | null
  created_at: string
  critic_scores: QueryResponse['critic_scores']
}

export const auth = {
  register: (username: string, email: string, password: string) =>
    api.post('/auth/register', { username, email, password }),
  login: (username: string, password: string) => {
    const form = new FormData()
    form.append('username', username)
    form.append('password', password)
    return axios.post(`${BASE}/api/v1/auth/login`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  logout: (refresh_token: string) => api.post('/auth/logout', { refresh_token }),
  me: () => api.get('/auth/me'),
}

export const queries = {
  run: (query: string) => api.post<QueryResponse>('/query', { query }),
  history: (limit = 20, offset = 0) => api.get<HistoryItem[]>(`/query/history?limit=${limit}&offset=${offset}`),
  get: (id: string) => api.get<HistoryItem>(`/query/${id}`),
}

export const documents = {
  upload: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return api.post('/ingest', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
  },
  list: (limit = 50, offset = 0) => api.get(`/documents?limit=${limit}&offset=${offset}`),
  get: (id: string) => api.get(`/documents/${id}`),
  delete: (id: string) => api.delete(`/documents/${id}`),
}

// WebSocket connection for streaming queries
export function connectQueryWS(
  token: string,
  onEvent: (event: string, data: unknown) => void,
  onClose: () => void
): WebSocket {
  const wsBase = (import.meta.env.VITE_API_URL ?? window.location.origin)
    .replace(/^http/, 'ws')
  const ws = new WebSocket(`${wsBase}/ws/query?token=${token}`)
  ws.onmessage = (e) => {
    try {
      const { event, data } = JSON.parse(e.data)
      onEvent(event, data)
    } catch { /* ignore */ }
  }
  ws.onclose = onClose
  return ws
}
