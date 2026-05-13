import { useState, useRef, useEffect } from 'react'
import { flushSync } from 'react-dom'
import { Send, Loader, Copy, Check, Zap, Search, Code, Shield, CheckCircle, XCircle, ChevronRight, Globe, Sparkles } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { queries, connectQueryWS, sendWSQuery, type QueryResponse, type HistoryMessage } from '../api/client'
import type { ChatMessage, Conversation } from '../hooks/useConversations'

// ── Floating particles ────────────────────────────────────────────────────────
const PARTICLES = Array.from({ length: 28 }, (_, i) => {
  const r = (seed: number) => { const x = Math.sin(seed) * 43758.5453; return x - Math.floor(x) }
  return {
    x:     r(i * 13.1),
    y:     r(i * 7.7),
    vx:    (r(i * 3.3) - 0.5) * 0.000055,
    vy:    (r(i * 17.9) - 0.5) * 0.000038 - 0.000018,
    size:  1.0 + r(i * 5.5) * 2.2,
    phase: i * 1.618,
    shape: i % 3,
    alpha: 0.04 + r(i * 11.3) * 0.055,
  }
})

function FloatingParticles() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef    = useRef(0)
  const posRef    = useRef(PARTICLES.map(p => ({ x: p.x, y: p.y })))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    const resize = () => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    let start: number | null = null
    const frame = (ts: number) => {
      if (!start) start = ts
      const t  = (ts - start) / 1000
      const W  = canvas.width
      const H  = canvas.height
      ctx.clearRect(0, 0, W, H)

      PARTICLES.forEach((p, i) => {
        const pos = posRef.current[i]
        pos.x += p.vx
        pos.y += p.vy
        if (pos.x < -0.05) pos.x = 1.05
        if (pos.x >  1.05) pos.x = -0.05
        if (pos.y < -0.05) pos.y = 1.05
        if (pos.y >  1.05) pos.y = -0.05

        const x  = pos.x * W
        const y  = pos.y * H
        const a  = p.alpha * (0.6 + Math.sin(t * 0.55 + p.phase) * 0.4)
        const sz = p.size  * (0.85 + Math.sin(t * 0.85 + p.phase * 1.3) * 0.15)

        ctx.save()
        ctx.globalAlpha = a

        if (p.shape === 0) {
          // dot
          ctx.fillStyle = '#ffffff'
          ctx.beginPath()
          ctx.arc(x, y, sz, 0, Math.PI * 2)
          ctx.fill()
        } else if (p.shape === 1) {
          // tiny square
          ctx.fillStyle = '#ffffff'
          ctx.fillRect(x - sz, y - sz, sz * 2, sz * 2)
        } else {
          // dash
          ctx.globalAlpha = a * 0.45
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth   = 0.6
          ctx.beginPath()
          ctx.moveTo(x - sz * 5, y)
          ctx.lineTo(x + sz * 5, y)
          ctx.stroke()
        }

        ctx.restore()
      })

      rafRef.current = requestAnimationFrame(frame)
    }

    rafRef.current = requestAnimationFrame(frame)
    return () => { cancelAnimationFrame(rafRef.current); ro.disconnect() }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 0 }}
    />
  )
}

const PIPELINE_STEPS = ['planning', 'retrieving', 'executing', 'synthesizing', 'evaluating', 'done'] as const
const STEP_ICON: Record<string, React.ReactNode> = {
  planning:    <Zap size={10} />,
  retrieving:  <Search size={10} />,
  executing:   <Code size={10} />,
  synthesizing:<Zap size={10} />,
  evaluating:  <Shield size={10} />,
  done:        <CheckCircle size={10} />,
  error:       <XCircle size={10} />,
}
const STEP_LABEL: Record<string, string> = {
  planning: 'Planning', retrieving: 'Retrieve', executing: 'Execute',
  synthesizing: 'Writing', evaluating: 'Evaluate', done: 'Done', error: 'Error',
}

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) }}
      className="flex items-center gap-1.5 font-mono text-[9px] transition-colors px-2 py-1 border"
      style={{
        borderColor: 'rgba(255,255,255,0.06)',
        color: copied ? '#86efac' : 'rgba(255,255,255,0.25)',
        background: copied ? 'rgba(134,239,172,0.06)' : 'transparent',
      }}
    >
      {copied ? <Check size={10} /> : <Copy size={10} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function PipelineIndicator({ step }: { step: string }) {
  const idx = PIPELINE_STEPS.indexOf(step as typeof PIPELINE_STEPS[number])
  return (
    <div className="flex items-center gap-1 mb-3">
      {PIPELINE_STEPS.filter(s => s !== 'done').map((s, i) => {
        const active = i === idx
        const done   = i < idx
        return (
          <div
            key={s}
            className={`flex items-center gap-1 font-mono text-[9px] px-2 py-1 transition-all`}
            style={{
              border: `1px solid ${active ? 'rgba(139,92,246,0.5)' : done ? 'rgba(134,239,172,0.2)' : 'rgba(255,255,255,0.05)'}`,
              color:  active ? '#a78bfa' : done ? 'rgba(134,239,172,0.55)' : 'rgba(255,255,255,0.15)',
              background: active ? 'rgba(139,92,246,0.08)' : done ? 'rgba(134,239,172,0.04)' : 'transparent',
            }}
          >
            {active ? <Loader size={9} className="animate-spin" /> : STEP_ICON[s]}
            <span className="hidden sm:inline">{STEP_LABEL[s]}</span>
          </div>
        )
      })}
    </div>
  )
}

function AssistantBubble({ msg, isStreaming }: { msg: ChatMessage; isStreaming: boolean }) {
  const result = msg.result

  return (
    <div className="flex gap-4 max-w-3xl">
      {/* Left accent line */}
      <div className="shrink-0 flex flex-col items-center gap-1 pt-0.5">
        <div
          className="w-5 h-5 flex items-center justify-center font-mono text-[9px] shrink-0"
          style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.25)', color: '#a78bfa' }}
        >
          H
        </div>
        {(msg.content || isStreaming) && (
          <div className="w-px flex-1" style={{ background: 'rgba(139,92,246,0.1)', minHeight: 20 }} />
        )}
      </div>

      <div className="flex-1 min-w-0 pb-4">
        {/* Pipeline steps */}
        {isStreaming && msg.step && msg.step !== 'done' && (
          <PipelineIndicator step={msg.step} />
        )}

        {/* Error */}
        {msg.error && (
          <div
            className="px-4 py-3 font-mono text-xs"
            style={{ border: '1px solid rgba(248,113,113,0.3)', background: 'rgba(248,113,113,0.05)', color: '#f87171' }}
          >
            {msg.error}
          </div>
        )}

        {/* Answer content */}
        {msg.content && (
          <div
            className="text-sm leading-[1.8]"
            style={{ color: 'rgba(255,255,255,0.85)' }}
          >
            <ReactMarkdown
              components={{
                a: ({ href, children }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: '#a78bfa', textDecoration: 'none' }}
                    onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.textDecoration = 'underline')}
                    onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.textDecoration = 'none')}
                  >
                    {children}
                  </a>
                ),
                code: ({ children }) => (
                  <code
                    className="px-1.5 py-0.5 rounded font-mono text-[12px]"
                    style={{ background: 'rgba(139,92,246,0.1)', color: '#c4b5fd', border: '1px solid rgba(139,92,246,0.15)' }}
                  >
                    {children}
                  </code>
                ),
                pre: ({ children }) => (
                  <pre
                    className="text-xs overflow-x-auto my-4 p-4 font-mono"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', color: 'rgba(255,255,255,0.7)' }}
                  >
                    {children}
                  </pre>
                ),
                strong: ({ children }) => <strong style={{ color: '#ffffff', fontWeight: 600 }}>{children}</strong>,
                ul: ({ children }) => <ul className="list-disc list-inside space-y-1.5 my-3">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside space-y-1.5 my-3">{children}</ol>,
                li: ({ children }) => <li style={{ color: 'rgba(255,255,255,0.75)' }}>{children}</li>,
                h1: ({ children }) => <h1 className="text-lg font-semibold mt-5 mb-2" style={{ color: '#ffffff' }}>{children}</h1>,
                h2: ({ children }) => <h2 className="text-base font-semibold mt-4 mb-1.5" style={{ color: '#ffffff' }}>{children}</h2>,
                h3: ({ children }) => <h3 className="font-medium mt-3 mb-1" style={{ color: 'rgba(255,255,255,0.9)' }}>{children}</h3>,
                p: ({ children }) => <p className="mb-2.5">{children}</p>,
                blockquote: ({ children }) => (
                  <blockquote
                    className="pl-4 my-3 italic"
                    style={{ borderLeft: '2px solid rgba(139,92,246,0.4)', color: 'rgba(255,255,255,0.5)' }}
                  >
                    {children}
                  </blockquote>
                ),
              }}
            >
              {msg.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Streaming cursor */}
        {isStreaming && !msg.error && (
          <span
            className={`inline-block w-1 h-4 animate-pulse align-middle ${msg.content ? 'ml-0.5' : 'mt-1'}`}
            style={{ background: 'rgba(139,92,246,0.7)' }}
          />
        )}

        {/* Metadata */}
        {result && !isStreaming && (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span className="font-mono text-[9px]" style={{ color: 'rgba(255,255,255,0.2)' }}>
              {result.latency_ms.toFixed(0)}ms
            </span>
            {result.retrieved_docs.length > 0 && (
              <span className="font-mono text-[9px]" style={{ color: 'rgba(255,255,255,0.2)' }}>
                {result.retrieved_docs.length} sources
              </span>
            )}
            {result.web_sources && result.web_sources.length > 0 && (
              <span className="flex items-center gap-1 font-mono text-[9px]" style={{ color: 'rgba(255,255,255,0.2)' }}>
                <Globe size={9} />{result.web_sources.length} web
              </span>
            )}
            <CopyBtn text={msg.content} />
          </div>
        )}

        {/* Citations */}
        {result && !isStreaming && (result.web_sources?.length > 0 || result.retrieved_docs?.length > 0) && (
          <div className="mt-4 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
            <p className="font-mono text-[9px] tracking-[0.25em] uppercase mb-2.5" style={{ color: 'rgba(255,255,255,0.18)' }}>
              Sources
            </p>
            <div className="space-y-2">
              {result.web_sources?.map((src, i) => (
                <div key={`w${i}`} className="flex items-start gap-2">
                  <span className="font-mono text-[9px] shrink-0 mt-0.5" style={{ color: '#a78bfa' }}>[W{i + 1}]</span>
                  <div className="min-w-0 flex-1">
                    <a
                      href={src.url} target="_blank" rel="noopener noreferrer"
                      className="font-mono text-[10px] block truncate hover:underline"
                      style={{ color: '#a78bfa' }}
                    >
                      {src.title || src.url}
                    </a>
                    <p className="font-mono text-[9px] mt-0.5 truncate" style={{ color: 'rgba(255,255,255,0.22)' }}>
                      {src.url}
                    </p>
                  </div>
                </div>
              ))}
              {result.retrieved_docs?.length > 0 && result.retrieved_docs.map((doc, i) => (
                <div key={`d${i}`} className="flex items-start gap-2">
                  <span className="font-mono text-[9px] shrink-0" style={{ color: 'rgba(255,255,255,0.28)' }}>[D{i + 1}]</span>
                  <p className="font-mono text-[9px] flex-1 truncate" style={{ color: 'rgba(255,255,255,0.28)' }}>
                    {(doc.metadata?.filename as string) || doc.id.slice(0, 48)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Follow-ups */}
        {result?.follow_up_questions && result.follow_up_questions.length > 0 && !isStreaming && (
          <div className="mt-5">
            <p className="font-mono text-[9px] tracking-[0.25em] uppercase mb-2" style={{ color: 'rgba(255,255,255,0.2)' }}>
              Related questions
            </p>
            <div className="space-y-1.5">
              {result.follow_up_questions.map((q, i) => (
                <button
                  key={i}
                  className="follow-up-btn flex items-center gap-2.5 text-left font-mono text-[11px] w-full px-3 py-2.5 transition-all"
                  data-question={q}
                  style={{ border: '1px solid rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.4)', background: 'transparent' }}
                  onMouseEnter={e => {
                    ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.3)'
                    ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(167,139,250,0.9)'
                    ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(139,92,246,0.04)'
                  }}
                  onMouseLeave={e => {
                    ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.06)'
                    ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.4)'
                    ;(e.currentTarget as HTMLButtonElement).style.background = 'transparent'
                  }}
                >
                  <ChevronRight size={11} style={{ color: '#8b5cf6', flexShrink: 0 }} />
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function UserBubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div
        className="font-sans text-sm leading-relaxed whitespace-pre-wrap break-words px-4 py-3 max-w-xl"
        style={{
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.10)',
          color: 'rgba(255,255,255,0.88)',
          borderRadius: '60% 40% 30% 70% / 60% 30% 70% 40%',
        }}
      >
        {msg.content}
      </div>
    </div>
  )
}

function EmptyState({ onExample }: { onExample: (q: string) => void }) {
  const examples = [
    { label: 'Summarize my documents', icon: '⟐' },
    { label: 'What are the main topics?', icon: '◈' },
    { label: 'Explain the pipeline architecture', icon: '⟡' },
    { label: 'Find the most important findings', icon: '◇' },
  ]
  return (
    <div className="flex flex-col items-center justify-center h-full gap-10 pb-16 px-6">
      {/* Branding */}
      <div className="text-center space-y-3">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div
            className="w-10 h-10 flex items-center justify-center"
            style={{ background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.25)' }}
          >
            <Sparkles size={18} style={{ color: '#8b5cf6' }} />
          </div>
        </div>
        <h2
          className="font-mono tracking-[0.3em] uppercase"
          style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', letterSpacing: '0.3em' }}
        >
          HELIOS
        </h2>
        <p className="font-mono text-[10px] tracking-wider" style={{ color: 'rgba(255,255,255,0.2)' }}>
          Distributed Multi-Agent AI · Five agents · One pipeline
        </p>
      </div>

      {/* Divider */}
      <div className="w-24 h-px" style={{ background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent)' }} />

      {/* Examples */}
      <div className="w-full max-w-lg space-y-2">
        <p className="font-mono text-[9px] tracking-[0.3em] uppercase text-center mb-4" style={{ color: 'rgba(255,255,255,0.18)' }}>
          Try asking
        </p>
        {examples.map(ex => (
          <button
            key={ex.label}
            onClick={() => onExample(ex.label)}
            className="flex items-center gap-3 w-full px-4 py-3 text-left font-mono text-[11px] transition-all"
            style={{ border: '1px solid rgba(255,255,255,0.05)', color: 'rgba(255,255,255,0.35)', background: 'rgba(255,255,255,0.01)' }}
            onMouseEnter={e => {
              ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.25)'
              ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(167,139,250,0.8)'
              ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(139,92,246,0.04)'
            }}
            onMouseLeave={e => {
              ;(e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.05)'
              ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.35)'
              ;(e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.01)'
            }}
          >
            <span style={{ color: 'rgba(139,92,246,0.5)', fontFamily: 'monospace' }}>{ex.icon}</span>
            {ex.label}
            <ChevronRight size={11} className="ml-auto shrink-0" style={{ color: 'rgba(139,92,246,0.3)' }} />
          </button>
        ))}
      </div>
    </div>
  )
}

type Props = {
  conversation: Conversation | null
  isLoggedIn: boolean
  onAuthRequired: () => void
  onAddUserMessage: (convId: string, content: string) => string
  onAddPlaceholder: (convId: string) => string
  onUpdateMessage: (convId: string, msgId: string, patch: Partial<ChatMessage>) => void
  onNeedConversation: () => string
}

export function ChatView({ conversation, isLoggedIn, onAuthRequired, onAddUserMessage, onAddPlaceholder, onUpdateMessage, onNeedConversation }: Props) {
  const [input, setInput] = useState('')
  const [busyMsgId, setBusyMsgId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const autoFiredRef = useRef<Set<string>>(new Set())
  const convId = conversation?.id ?? null

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages.length])

  const buildHistory = (msgs: Conversation['messages'], beforeId: string): HistoryMessage[] => {
    const turns: HistoryMessage[] = []
    for (const m of msgs) {
      if (m.id === beforeId) break
      if (m.role === 'user' || (m.role === 'assistant' && m.content)) {
        turns.push({ role: m.role, content: m.content })
      }
    }
    return turns.slice(-12)
  }

  const fireQuery = (q: string, cid: string, assistantMsgId: string) => {
    setBusyMsgId(assistantMsgId)
    const token = localStorage.getItem('access_token')
    const msgs = conversation?.messages ?? []
    const history = buildHistory(msgs, assistantMsgId)

    if (token) {
      wsRef.current?.close()
      let accumulated = ''
      let executingShown = false
      const ws = connectQueryWS(
        token,
        (event, data) => {
          if (['planning', 'retrieving', 'evaluating'].includes(event)) {
            // flushSync ensures each step gets its own render frame — prevents React 18
            // automatic batching from collapsing planning+retrieving into one update
            flushSync(() => onUpdateMessage(cid, assistantMsgId, { step: event }))
          } else if (event === 'token') {
            const t = (data as { token?: string })?.token ?? ''
            accumulated += t
            if (!executingShown) {
              // Synthesise the Execute step — backend never emits it explicitly.
              // Show it for 300ms before switching to Writing (synthesizing).
              executingShown = true
              flushSync(() => onUpdateMessage(cid, assistantMsgId, { step: 'executing' }))
              setTimeout(() => {
                onUpdateMessage(cid, assistantMsgId, { content: accumulated, step: 'synthesizing' })
              }, 300)
            } else {
              onUpdateMessage(cid, assistantMsgId, { content: accumulated, step: 'synthesizing' })
            }
          } else if (event === 'done') {
            if (!accumulated) {
              queries.run(q, history).then(({ data: r }) => {
                onUpdateMessage(cid, assistantMsgId, { content: r.answer, result: r, step: 'done' })
                setBusyMsgId(null)
              }).catch(() => setBusyMsgId(null))
            } else {
              queries.run(q, history).then(({ data: r }) => {
                onUpdateMessage(cid, assistantMsgId, { result: r, step: 'done' })
                setBusyMsgId(null)
              }).catch(() => {
                onUpdateMessage(cid, assistantMsgId, { step: 'done' })
                setBusyMsgId(null)
              })
            }
          } else if (event === 'retrying') {
            accumulated = ''
            onUpdateMessage(cid, assistantMsgId, { content: '', step: 'planning' })
          } else if (event === 'error') {
            const d = data as { message?: string }
            onUpdateMessage(cid, assistantMsgId, { error: d.message ?? 'Pipeline error', step: 'error' })
            setBusyMsgId(null)
          }
        },
        () => setBusyMsgId(null)
      )
      wsRef.current = ws
      ws.onopen = () => sendWSQuery(ws, q, history)
    } else {
      // Guest / no-token path — REST only, no WS events.
      // Simulate step progression visually so the indicator always animates.
      const timers: ReturnType<typeof setTimeout>[] = []
      flushSync(() => onUpdateMessage(cid, assistantMsgId, { step: 'planning' }))
      timers.push(setTimeout(() => flushSync(() => onUpdateMessage(cid, assistantMsgId, { step: 'retrieving' })),  500))
      timers.push(setTimeout(() => flushSync(() => onUpdateMessage(cid, assistantMsgId, { step: 'executing' })),   1100))
      timers.push(setTimeout(() => flushSync(() => onUpdateMessage(cid, assistantMsgId, { step: 'synthesizing' })), 1800))

      queries.run(q, history).then(({ data: r }) => {
        timers.forEach(clearTimeout)
        onUpdateMessage(cid, assistantMsgId, { step: 'evaluating' })
        setTimeout(() => {
          onUpdateMessage(cid, assistantMsgId, { content: r.answer, result: r, step: 'done' })
          setBusyMsgId(null)
        }, 400)
      }).catch((e: unknown) => {
        timers.forEach(clearTimeout)
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Request failed'
        onUpdateMessage(cid, assistantMsgId, { error: msg, step: 'error' })
        setBusyMsgId(null)
      })
    }
  }

  useEffect(() => {
    if (!conversation || busyMsgId) return
    const msgs = conversation.messages
    if (msgs.length === 0) return
    const last = msgs[msgs.length - 1]
    if (last.role !== 'user') return
    if (autoFiredRef.current.has(last.id)) return
    autoFiredRef.current.add(last.id)
    const assistantMsgId = onAddPlaceholder(conversation.id)
    fireQuery(last.content, conversation.id, assistantMsgId)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversation?.id, conversation?.messages.length, busyMsgId])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const btn = (e.target as Element).closest('.follow-up-btn') as HTMLElement | null
      if (btn) {
        const q = btn.dataset.question
        if (q) sendQuery(q)
      }
    }
    document.addEventListener('click', handler)
    return () => document.removeEventListener('click', handler)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [convId, busyMsgId])

  const sendQuery = (q: string) => {
    if (!q.trim() || busyMsgId) return
    let cid = convId
    if (!cid) cid = onNeedConversation()
    onAddUserMessage(cid, q)
    const assistantMsgId = onAddPlaceholder(cid)
    fireQuery(q, cid, assistantMsgId)
  }

  const handleSubmit = () => {
    const q = input.trim().slice(0, 500)
    if (!q || busyMsgId) return
    setInput('')
    sendQuery(q)
  }

  const messages = conversation?.messages ?? []

  return (
    <div className="flex flex-col h-full" style={{ background: '#030305' }}>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-8 space-y-8 relative">
        <FloatingParticles />
        {messages.length === 0
          ? <EmptyState onExample={q => { setInput(q); inputRef.current?.focus() }} />
          : messages.map(msg =>
              msg.role === 'user'
                ? <UserBubble key={msg.id} msg={msg} />
                : <AssistantBubble key={msg.id} msg={msg} isStreaming={msg.id === busyMsgId} />
            )
        }
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div
        className="px-4 sm:px-8 py-4"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)', background: '#030305' }}
      >
        <div className="max-w-3xl mx-auto">
          <div
            className="flex items-end gap-3 px-4 py-3 transition-all duration-200"
            style={{
              border: '1px solid rgba(255,255,255,0.08)',
              background: 'rgba(255,255,255,0.02)',
            }}
            onFocusCapture={e => {
              ;(e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(139,92,246,0.4)'
              ;(e.currentTarget as HTMLDivElement).style.boxShadow  = '0 0 0 1px rgba(139,92,246,0.1), 0 8px 32px rgba(0,0,0,0.4)'
            }}
            onBlurCapture={e => {
              ;(e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.08)'
              ;(e.currentTarget as HTMLDivElement).style.boxShadow  = 'none'
            }}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={e => {
                setInput(e.target.value)
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
              }}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() } }}
              placeholder="Ask anything about your documents…"
              className="flex-1 bg-transparent font-sans text-sm outline-none resize-none leading-relaxed placeholder:text-[rgba(255,255,255,0.18)]"
              style={{ minHeight: '24px', maxHeight: '160px', color: 'rgba(255,255,255,0.88)', caretColor: '#8b5cf6' }}
            />
            <button
              onClick={handleSubmit}
              disabled={!!busyMsgId || !input.trim()}
              className="shrink-0 w-8 h-8 flex items-center justify-center transition-all"
              style={{
                background: busyMsgId || !input.trim() ? 'rgba(139,92,246,0.1)' : 'rgba(139,92,246,0.85)',
                color: busyMsgId || !input.trim() ? 'rgba(139,92,246,0.4)' : '#fff',
                cursor: busyMsgId || !input.trim() ? 'not-allowed' : 'pointer',
              }}
            >
              {busyMsgId ? <Loader size={14} className="animate-spin" /> : <Send size={14} />}
            </button>
          </div>
          <p className="font-mono text-[9px] mt-2 text-center" style={{ color: 'rgba(255,255,255,0.12)' }}>
            Enter ↵ to send · Shift+Enter for newline
            {!isLoggedIn && ' · Sign in for unlimited queries'}
          </p>
        </div>
      </div>
    </div>
  )
}
