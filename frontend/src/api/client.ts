import axios from 'axios'

export const BASE = import.meta.env.VITE_API_URL ?? ''

export const api = axios.create({
  baseURL: `${BASE}/api/v1`,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60_000,
})

/**
 * Turn an axios / fetch error into a message the user can act on.
 *
 * Before this helper, both ChatView and QueryInterface fell back to the
 * literal string "Request failed" whenever there was no
 * response.data.detail — which is every network failure (DNS, CORS,
 * TLS, timeout). See 001_report.md for the incident where this made
 * three different root causes indistinguishable in the UI.
 *
 * We keep the message short (the UI truncates long strings) but include
 * enough signal that a user can tell "backend is down" from "you sent
 * a malformed request". The full error is always console.error'd.
 */
export function humanReadableError(e: unknown): string {
  // Always log the raw shape for debugging.
  // eslint-disable-next-line no-console
  console.error('[helios] request failed:', e)

  const err = e as {
    code?: string
    message?: string
    response?: { status?: number; data?: { detail?: string | Array<{ msg?: string }> } }
    request?: unknown
  }

  const detail = err.response?.data?.detail
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map(d => d?.msg ?? String(d)).join('; ')
  }
  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }

  const status = err.response?.status
  if (typeof status === 'number') {
    if (status === 401) return 'Session expired — please sign in again.'
    if (status === 403) return 'You are not allowed to run this query.'
    if (status === 413) return 'The upload is too large for this Space.'
    if (status === 429) return 'Rate limit hit — wait a moment and try again.'
    if (status === 503) return 'The API is starting up — retry in ~30 seconds.'
    if (status >= 500) return `Backend error (HTTP ${status}). Try again shortly.`
    return `HTTP ${status} from the backend.`
  }

  const code = err.code
  if (code === 'ECONNABORTED' || /timeout/i.test(err.message ?? '')) {
    return 'The query took longer than 60 s and was cancelled.'
  }
  if (code === 'ERR_NETWORK' || err.request) {
    const host = safeHost(BASE)
    return host
      ? `Cannot reach the Helios API at ${host}. Check your connection or the API deploy status.`
      : 'Cannot reach the Helios API. Check your connection or the API deploy status.'
  }
  return err.message?.slice(0, 200) || 'Unexpected error contacting the backend.'
}

function safeHost(url: string): string {
  try {
    return url ? new URL(url).host : ''
  } catch {
    return ''
  }
}

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

export type WebSource = {
  title: string
  url: string
  snippet: string
}

export type QueryResponse = {
  query_id: string
  query: string
  answer: string
  plan: { query_type: string; subtasks: { id: number; type: string; description: string }[] } | null
  retrieved_docs: { id: string; document: string; metadata: Record<string, unknown>; score: number; source: string }[]
  web_sources: WebSource[]
  execution_result: { stdout: string; stderr: string; success: boolean } | null
  critic_scores: { groundedness: number; faithfulness: number; completeness: number; overall: number; pass: boolean; reasoning: string; suggestions?: string[] } | null
  critic_passed: boolean | null
  verifier_scores: { groundedness: number; faithfulness: number; agreement: number; overall: number; pass: boolean; reasoning: string; flags?: string[] } | null
  verifier_passed: boolean | null
  follow_up_questions: string[]
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
  verifier_scores: QueryResponse['verifier_scores']
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

export type HistoryMessage = { role: 'user' | 'assistant'; content: string }

export const queries = {
  run: (query: string, history: HistoryMessage[] = []) =>
    api.post<QueryResponse>('/query', { query, history }),
  history: (limit = 20, offset = 0) => api.get<HistoryItem[]>(`/query/history?limit=${limit}&offset=${offset}`),
  get: (id: string) => api.get<HistoryItem>(`/query/${id}`),
}

export type ServerConversation = {
  id: string; title: string; created_at: string; updated_at: string; message_count: number
}
export type ServerMessage = { id: string; role: string; content: string; created_at: string }
export type ServerConversationDetail = ServerConversation & { messages: ServerMessage[] }

export const conversations = {
  list: (limit = 50) => api.get<ServerConversation[]>(`/conversations?limit=${limit}`),
  create: (title = 'New Chat') => api.post<ServerConversation>('/conversations', { title }),
  get: (id: string) => api.get<ServerConversationDetail>(`/conversations/${id}`),
  delete: (id: string) => api.delete(`/conversations/${id}`),
  addMessage: (convId: string, role: 'user' | 'assistant', content: string) =>
    api.post<ServerMessage>(`/conversations/${convId}/messages`, { role, content }),
}

export type ChunkPreview = { chunk_index: number; text: string; char_count: number }
export type DocumentChunksResponse = { document_id: string; filename: string; total_chunks: number; chunks: ChunkPreview[] }
export type TestRetrievalResult = { chunk_index: number; text: string; score: number; source: string }
export type TestRetrievalResponse = { query: string; results: TestRetrievalResult[] }

export const documents = {
  upload: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return api.post('/ingest', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
  },
  list: (limit = 50, offset = 0) => api.get(`/documents?limit=${limit}&offset=${offset}`),
  get: (id: string) => api.get(`/documents/${id}`),
  delete: (id: string) => api.delete(`/documents/${id}`),
  chunks: (id: string, limit = 20, offset = 0) =>
    api.get<DocumentChunksResponse>(`/documents/${id}/chunks?limit=${limit}&offset=${offset}`),
  testSearch: (id: string, query: string) =>
    api.post<TestRetrievalResponse>(`/documents/${id}/search`, { query }),
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

export function sendWSQuery(ws: WebSocket, query: string, history: HistoryMessage[] = []) {
  ws.send(JSON.stringify({ query, history }))
}
