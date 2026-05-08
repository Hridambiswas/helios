import { useState, useRef, useEffect } from 'react'
import { Send, Zap, Search, Code, Shield, CheckCircle, XCircle, Loader } from 'lucide-react'
import { queries, connectQueryWS, type QueryResponse } from '../api/client'

const GUEST_KEY = 'helios_guest_queries'
const getGuestCount = () => parseInt(localStorage.getItem(GUEST_KEY) ?? '0', 10)
const incGuestCount = () => localStorage.setItem(GUEST_KEY, String(getGuestCount() + 1))

type PipelineStep = 'idle' | 'planning' | 'retrieving' | 'executing' | 'evaluating' | 'done' | 'error'

const STEP_LABELS: Record<string, string> = {
  planning: 'PLANNING',
  retrieving: 'RETRIEVING',
  executing: 'EXECUTING',
  evaluating: 'EVALUATING',
  done: 'COMPLETE',
  error: 'ERROR',
}

const STEP_ICONS: Record<string, React.ReactNode> = {
  planning: <Zap size={14} />,
  retrieving: <Search size={14} />,
  executing: <Code size={14} />,
  evaluating: <Shield size={14} />,
  done: <CheckCircle size={14} />,
  error: <XCircle size={14} />,
}

const STEPS: PipelineStep[] = ['planning', 'retrieving', 'executing', 'evaluating', 'done']

export function QueryInterface({ initialQuery, onNewResult, isLoggedIn, onAuthRequired }: {
  initialQuery?: string
  onNewResult?: (r: QueryResponse) => void
  isLoggedIn?: boolean
  onAuthRequired?: (q: string) => void
}) {
  const [query, setQuery] = useState(initialQuery ?? '')
  const [step, setStep] = useState<PipelineStep>('idle')
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [streaming, setStreaming] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery)
      runQuery(initialQuery)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery])

  const runQuery = async (q: string) => {
    if (!q.trim() || step !== 'idle') return

    if (!isLoggedIn && getGuestCount() >= 1) {
      onAuthRequired?.(q)
      return
    }

    setResult(null)
    setErrorMsg('')

    const token = localStorage.getItem('access_token')

    // Use WebSocket for streaming if token is available
    if (token) {
      setStreaming(true)
      setStep('planning')
      wsRef.current?.close()

      const ws = connectQueryWS(
        token,
        (event, data) => {
          if (event === 'planning') setStep('planning')
          else if (event === 'retrieving') setStep('retrieving')
          else if (event === 'evaluating') setStep('evaluating')
          else if (event === 'done') {
            setStep('done')
            setStreaming(false)
            // Fetch full result for critic scores etc.
            queries.run(q).then(({ data }) => {
              setResult(data)
              onNewResult?.(data)
            }).catch(() => {
              // Use WS data as fallback
              const d = data as QueryResponse
              setResult(d)
            })
          } else if (event === 'error') {
            const d = data as { message: string }
            setErrorMsg(d.message ?? 'Pipeline error')
            setStep('error')
            setStreaming(false)
          }
        },
        () => { setStreaming(false) }
      )
      wsRef.current = ws
      ws.onopen = () => ws.send(JSON.stringify({ query: q }))
    } else {
      // Fallback: REST API
      setStep('planning')
      try {
        const { data } = await queries.run(q)
        setResult(data)
        setStep('done')
        if (!isLoggedIn) incGuestCount()
        onNewResult?.(data)
      } catch (e: unknown) {
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        setErrorMsg(msg ?? 'Request failed')
        setStep('error')
      }
    }
  }

  const handleSubmit = () => {
    const q = query.trim()
    if (!q) return
    setStep('idle')
    setTimeout(() => runQuery(q), 0)
  }

  const currentStepIdx = STEPS.indexOf(step)

  return (
    <section id="query-section" className="relative py-20 px-4">
      {/* Section header */}
      <div className="max-w-4xl mx-auto mb-12">
        <div className="flex items-baseline gap-6">
          <span className="section-number" style={{ fontSize: 'clamp(40px,6vw,72px)' }}>02</span>
          <div>
            <div className="hr-red w-16 mb-2" />
            <h2 className="font-mono text-xs tracking-[0.3em] uppercase text-crimson">Query Interface</h2>
            <p className="text-white text-2xl font-light mt-1">Run the Full Pipeline</p>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto">
        {/* Input */}
        <div className="card-dark border border-crimson/25 mb-6 relative group hover:border-crimson/50 transition-all">
          <div className="h-0.5 w-full bg-gradient-to-r from-crimson to-crimson-dark" />
          <div className="p-4 flex gap-3 items-start">
            <span className="font-mono text-crimson text-sm mt-0.5 shrink-0">{'>'}_</span>
            <textarea
              ref={inputRef}
              rows={3}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() } }}
              placeholder="Enter your query... (Shift+Enter for newline)"
              className="flex-1 bg-transparent text-white font-mono text-sm placeholder-[#444] outline-none resize-none"
            />
            <button
              onClick={handleSubmit}
              disabled={step !== 'idle' && step !== 'done' && step !== 'error'}
              className="shrink-0 p-2 bg-crimson hover:bg-crimson-light disabled:opacity-40 transition-colors text-white"
            >
              {streaming ? <Loader size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
        </div>

        {/* Pipeline progress */}
        {step !== 'idle' && (
          <div className="mb-8">
            <div className="flex items-center gap-0 overflow-hidden">
              {STEPS.map((s, i) => {
                const active = i === currentStepIdx
                const done = i < currentStepIdx
                const isError = step === 'error' && i === currentStepIdx
                return (
                  <div key={s} className="flex items-center flex-1">
                    <div className={`flex items-center gap-1.5 px-3 py-2 text-[10px] font-mono tracking-wider uppercase transition-all border-y ${
                      isError
                        ? 'bg-red-900/30 text-red-400 border-red-900/50'
                        : active
                        ? 'bg-crimson/15 text-crimson border-crimson/40'
                        : done
                        ? 'bg-white/5 text-green-400 border-white/10'
                        : 'text-[#444] border-white/5'
                    } ${i === 0 ? 'border-l' : ''} ${i === STEPS.length - 1 ? 'border-r' : ''}`}>
                      {active && !isError ? (
                        <Loader size={10} className="animate-spin" />
                      ) : STEP_ICONS[s]}
                      <span className="hidden sm:inline">{STEP_LABELS[s]}</span>
                    </div>
                    {i < STEPS.length - 1 && (
                      <div className={`h-px flex-1 transition-colors ${done ? 'bg-green-900/50' : 'bg-white/5'}`} />
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Error */}
        {step === 'error' && (
          <div className="card-dark border border-crimson/50 p-4 mb-6">
            <p className="text-crimson font-mono text-sm">{errorMsg || 'Pipeline failed'}</p>
          </div>
        )}

        {/* Result */}
        {result && step === 'done' && (
          <ResultCard result={result} />
        )}
      </div>
    </section>
  )
}

function ResultCard({ result }: { result: QueryResponse }) {
  const [activeTab, setActiveTab] = useState<'answer' | 'docs' | 'plan' | 'eval'>('answer')

  const tabs = [
    { id: 'answer', label: 'ANSWER' },
    { id: 'docs', label: `SOURCES (${result.retrieved_docs.length})` },
    { id: 'plan', label: 'PLAN' },
    { id: 'eval', label: 'EVALUATION' },
  ] as const

  return (
    <div className="card-dark border border-crimson/25 overflow-hidden animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-3">
          <CheckCircle size={14} className="text-green-400" />
          <span className="font-mono text-xs text-[#888]">
            {result.query.slice(0, 60)}{result.query.length > 60 ? '…' : ''}
          </span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono text-[#555]">
          <span className={result.critic_passed ? 'text-green-400' : 'text-crimson'}>
            {result.critic_passed ? '✓ CRITIC PASS' : '✗ CRITIC FAIL'}
          </span>
          <span>{result.latency_ms.toFixed(0)} ms</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/5">
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`px-4 py-2.5 font-mono text-[10px] tracking-wider uppercase border-b-2 transition-colors ${
              activeTab === id
                ? 'border-crimson text-crimson'
                : 'border-transparent text-[#555] hover:text-white'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'answer' && (
          <div className="prose prose-invert max-w-none">
            <p className="text-white/90 leading-relaxed whitespace-pre-wrap text-sm">{result.answer}</p>
          </div>
        )}

        {activeTab === 'docs' && (
          <div className="space-y-3">
            {result.retrieved_docs.length === 0 ? (
              <p className="text-[#555] font-mono text-sm">No documents retrieved.</p>
            ) : result.retrieved_docs.map((doc, i) => (
              <div key={doc.id} className="border border-white/5 p-3 hover:border-crimson/20 transition-colors">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-[10px] text-crimson">[{i + 1}] {doc.id.slice(0, 32)}...</span>
                  <div className="flex gap-3 text-[10px] font-mono text-[#555]">
                    <span className="text-[#888]">{(doc.score * 100).toFixed(0)}% match</span>
                    <span className="text-crimson/60 uppercase">{doc.source}</span>
                  </div>
                </div>
                <p className="text-[#888] text-xs leading-relaxed line-clamp-3">{doc.document}</p>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'plan' && result.plan && (
          <div className="space-y-3">
            <div className="flex gap-4 mb-4">
              <div>
                <span className="font-mono text-[10px] text-[#555] uppercase">Query Type</span>
                <p className="text-crimson font-mono text-sm uppercase">{result.plan.query_type}</p>
              </div>
            </div>
            {result.plan.subtasks.map(t => (
              <div key={t.id} className="flex gap-3 border border-white/5 p-3">
                <span className="font-mono text-crimson text-sm shrink-0">{String(t.id).padStart(2, '0')}</span>
                <div>
                  <span className="font-mono text-[10px] text-[#555] uppercase">{t.type}</span>
                  <p className="text-white/80 text-sm mt-0.5">{t.description}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'eval' && result.critic_scores && (
          <div className="space-y-4">
            {(['groundedness', 'faithfulness', 'completeness', 'overall'] as const).map(dim => {
              const score = result.critic_scores![dim]
              const pct = Math.round(score * 100)
              return (
                <div key={dim}>
                  <div className="flex justify-between mb-1.5">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-[#888]">{dim}</span>
                    <span className={`font-mono text-xs ${pct >= 70 ? 'text-green-400' : pct >= 40 ? 'text-yellow-400' : 'text-crimson'}`}>
                      {pct}%
                    </span>
                  </div>
                  <div className="h-1 bg-white/5">
                    <div
                      className={`h-full transition-all duration-700 ${pct >= 70 ? 'bg-green-500' : pct >= 40 ? 'bg-yellow-500' : 'bg-crimson'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              )
            })}
            {result.critic_scores.reasoning && (
              <div className="mt-4 p-3 bg-white/3 border border-white/5">
                <p className="text-[#888] text-xs leading-relaxed italic">{result.critic_scores.reasoning}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
