import { useState, useRef, useEffect } from 'react'
import { Send, Loader, Copy, Check, Zap, Search, Code, Shield, CheckCircle, XCircle, ChevronRight, Globe } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { queries, connectQueryWS, type QueryResponse } from '../api/client'
import type { ChatMessage, Conversation } from '../hooks/useConversations'

const PIPELINE_STEPS = ['planning', 'retrieving', 'executing', 'evaluating', 'done'] as const
const STEP_ICON: Record<string, React.ReactNode> = {
  planning:   <Zap size={11} />,
  retrieving: <Search size={11} />,
  executing:  <Code size={11} />,
  evaluating: <Shield size={11} />,
  done:       <CheckCircle size={11} />,
  error:      <XCircle size={11} />,
}
const STEP_LABEL: Record<string, string> = {
  planning: 'Planning', retrieving: 'Retrieving', executing: 'Executing',
  evaluating: 'Evaluating', done: 'Done', error: 'Error',
}

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) }}
      className="flex items-center gap-1 font-mono text-[10px] text-[#555] hover:text-crimson transition-colors"
    >
      {copied ? <Check size={11} className="text-green-400" /> : <Copy size={11} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function PipelineIndicator({ step }: { step: string }) {
  const idx = PIPELINE_STEPS.indexOf(step as typeof PIPELINE_STEPS[number])
  return (
    <div className="flex items-center gap-1.5 mt-2">
      {PIPELINE_STEPS.filter(s => s !== 'done').map((s, i) => {
        const active = i === idx
        const done = i < idx
        return (
          <div key={s} className={`flex items-center gap-1 font-mono text-[9px] px-1.5 py-0.5 border transition-colors ${
            active ? 'border-crimson/50 text-crimson bg-crimson/10' :
            done   ? 'border-green-900/40 text-green-500/60 bg-green-900/10' :
                     'border-white/5 text-[#333]'
          }`}>
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
    <div className="flex gap-3 max-w-3xl">
      {/* Avatar */}
      <div className="shrink-0 w-7 h-7 bg-crimson/15 border border-crimson/30 flex items-center justify-center text-crimson font-mono text-[10px] mt-1">
        H
      </div>

      <div className="flex-1 min-w-0">
        {/* Streaming pipeline indicator */}
        {isStreaming && msg.step && msg.step !== 'done' && (
          <PipelineIndicator step={msg.step} />
        )}

        {/* Error */}
        {msg.error && (
          <div className="mt-2 px-3 py-2 border border-crimson/40 bg-crimson/5 text-crimson font-mono text-xs">
            {msg.error}
          </div>
        )}

        {/* Answer */}
        {msg.content && (
          <div className="mt-2 text-white/90 text-sm leading-relaxed">
            <ReactMarkdown
              components={{
                a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-crimson hover:underline">{children}</a>,
                code: ({ children }) => <code className="bg-white/10 px-1 py-0.5 rounded text-xs font-mono text-crimson/90">{children}</code>,
                pre: ({ children }) => <pre className="bg-white/5 border border-white/10 p-3 rounded text-xs overflow-x-auto my-3">{children}</pre>,
                strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
                ul: ({ children }) => <ul className="list-disc list-inside space-y-1 my-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 my-2">{children}</ol>,
                li: ({ children }) => <li className="text-white/80">{children}</li>,
                h1: ({ children }) => <h1 className="text-white text-lg font-semibold mt-4 mb-2">{children}</h1>,
                h2: ({ children }) => <h2 className="text-white text-base font-semibold mt-3 mb-1">{children}</h2>,
                h3: ({ children }) => <h3 className="text-white/90 font-medium mt-2 mb-1">{children}</h3>,
                p: ({ children }) => <p className="mb-2">{children}</p>,
              }}
            >
              {msg.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Blinking cursor while streaming */}
        {isStreaming && !msg.content && !msg.error && (
          <div className="mt-2 flex items-center gap-1">
            <div className="w-1.5 h-4 bg-crimson/60 animate-pulse" />
          </div>
        )}

        {/* Metadata row */}
        {result && !isStreaming && (
          <div className="mt-3 flex flex-wrap items-center gap-4 text-[10px] font-mono text-[#444]">
            <span className={result.critic_passed ? 'text-green-500/70' : 'text-crimson/70'}>
              {result.critic_passed ? '✓ critic pass' : '✗ critic fail'}
            </span>
            <span>{result.latency_ms.toFixed(0)} ms</span>
            {result.retrieved_docs.length > 0 && <span>{result.retrieved_docs.length} sources</span>}
            {result.web_sources && result.web_sources.length > 0 && (
              <span className="flex items-center gap-1"><Globe size={9} />{result.web_sources.length} web</span>
            )}
            <CopyBtn text={msg.content} />
          </div>
        )}

        {/* Follow-up questions */}
        {result?.follow_up_questions && result.follow_up_questions.length > 0 && !isStreaming && (
          <div className="mt-4 space-y-1.5">
            <p className="font-mono text-[9px] text-[#444] uppercase tracking-wider mb-2">You might ask</p>
            {result.follow_up_questions.map((q, i) => (
              <button
                key={i}
                className="follow-up-btn flex items-center gap-2 text-left font-mono text-xs text-[#777] hover:text-white border border-white/5 hover:border-crimson/30 px-3 py-2 transition-all w-full"
                data-question={q}
              >
                <ChevronRight size={11} className="text-crimson shrink-0" />
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function UserBubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-xl bg-white/6 border border-white/10 px-4 py-2.5 text-white/90 text-sm leading-relaxed font-sans whitespace-pre-wrap">
        {msg.content}
      </div>
    </div>
  )
}

function EmptyState({ onExample }: { onExample: (q: string) => void }) {
  const examples = [
    'Summarize the key concepts in my documents',
    'What are the main topics in the knowledge base?',
    'Explain the architecture of this system',
    'What are the most important findings?',
  ]
  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 pb-20 px-4">
      <div className="text-center">
        <div className="font-mono text-4xl text-crimson mb-2">HELIOS</div>
        <p className="text-[#555] font-mono text-xs tracking-wider">Your AI research assistant</p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
        {examples.map(ex => (
          <button
            key={ex}
            onClick={() => onExample(ex)}
            className="text-left border border-white/8 hover:border-crimson/30 bg-white/2 hover:bg-crimson/5 px-4 py-3 text-[#777] hover:text-white font-mono text-[11px] transition-all leading-relaxed"
          >
            {ex}
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

  // Core API call — given a question and an already-created placeholder message ID
  const fireQuery = (q: string, cid: string, assistantMsgId: string) => {
    setBusyMsgId(assistantMsgId)
    const token = localStorage.getItem('access_token')

    if (token) {
      wsRef.current?.close()
      const ws = connectQueryWS(
        token,
        (event, data) => {
          if (['planning', 'retrieving', 'executing', 'evaluating'].includes(event)) {
            onUpdateMessage(cid, assistantMsgId, { step: event })
          } else if (event === 'done') {
            onUpdateMessage(cid, assistantMsgId, { step: 'done' })
            queries.run(q).then(({ data: r }) => {
              onUpdateMessage(cid, assistantMsgId, { content: r.answer, result: r, step: 'done' })
              setBusyMsgId(null)
            }).catch(() => setBusyMsgId(null))
          } else if (event === 'error') {
            const d = data as { message?: string }
            onUpdateMessage(cid, assistantMsgId, { error: d.message ?? 'Pipeline error', step: 'error' })
            setBusyMsgId(null)
          }
        },
        () => setBusyMsgId(null)
      )
      wsRef.current = ws
      ws.onopen = () => ws.send(JSON.stringify({ query: q }))
    } else {
      queries.run(q).then(({ data: r }) => {
        onUpdateMessage(cid, assistantMsgId, { content: r.answer, result: r, step: 'done' })
        setBusyMsgId(null)
      }).catch((e: unknown) => {
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Request failed'
        onUpdateMessage(cid, assistantMsgId, { error: msg, step: 'error' })
        setBusyMsgId(null)
      })
    }
  }

  // Auto-fire when a user message has no assistant reply yet (e.g. opened from landing page)
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

  // Wire follow-up button clicks via event delegation
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
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
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
      <div className="border-t border-white/8 bg-[#0d0d0d] px-4 py-3">
        <div className="max-w-3xl mx-auto relative">
          <div className="flex items-end gap-2 bg-white/4 border border-white/10 hover:border-crimson/30 focus-within:border-crimson/50 transition-all px-4 py-3">
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
              className="flex-1 bg-transparent text-white font-sans text-sm placeholder-[#444] outline-none resize-none leading-relaxed"
              style={{ minHeight: '24px', maxHeight: '160px' }}
            />
            <button
              onClick={handleSubmit}
              disabled={!!busyMsgId || !input.trim()}
              className="shrink-0 p-1.5 bg-crimson hover:bg-crimson-light disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-white"
            >
              {busyMsgId ? <Loader size={15} className="animate-spin" /> : <Send size={15} />}
            </button>
          </div>
          <p className="font-mono text-[9px] text-[#333] mt-1.5 text-center">
            Enter to send · Shift+Enter for newline
            {!isLoggedIn && ' · Sign in for full access'}
          </p>
        </div>
      </div>
    </div>
  )
}
