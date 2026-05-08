import { useEffect, useState } from 'react'
import { Clock, CheckCircle, XCircle, Loader } from 'lucide-react'
import { queries, type HistoryItem } from '../api/client'

export function HistorySection({ isLoggedIn, refreshTrigger }: {
  isLoggedIn: boolean
  refreshTrigger: number
}) {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!isLoggedIn) return
    setLoading(true)
    queries.history(10)
      .then(({ data }) => setItems(data))
      .catch(() => {/* ignore */})
      .finally(() => setLoading(false))
  }, [isLoggedIn, refreshTrigger])

  if (!isLoggedIn) return null

  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-baseline gap-6 mb-12">
          <span className="section-number" style={{ fontSize: 'clamp(40px,6vw,72px)' }}>04</span>
          <div>
            <div className="hr-red w-16 mb-2" />
            <h2 className="font-mono text-xs tracking-[0.3em] uppercase text-crimson">Query History</h2>
            <p className="text-white text-2xl font-light mt-1">Recent Sessions</p>
          </div>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-[#555] font-mono text-xs">
            <Loader size={12} className="animate-spin" />
            LOADING...
          </div>
        )}

        {!loading && items.length === 0 && (
          <div className="card-dark border border-white/5 p-8 text-center">
            <Clock size={24} className="text-[#333] mx-auto mb-3" />
            <p className="text-[#555] font-mono text-sm">No queries yet. Run your first one above.</p>
          </div>
        )}

        <div className="space-y-2">
          {items.map((item, i) => (
            <div key={item.id} className="card-dark border border-white/5 p-4 flex items-start gap-4 hover:border-crimson/20 transition-all group">
              {/* Index */}
              <span className="font-mono text-[#444] text-xs shrink-0 w-6 text-right">{String(i + 1).padStart(2, '0')}</span>

              {/* Status icon */}
              <div className="shrink-0 mt-0.5">
                {item.status === 'done' ? (
                  <CheckCircle size={14} className="text-green-500" />
                ) : item.status === 'failed' ? (
                  <XCircle size={14} className="text-crimson" />
                ) : (
                  <Loader size={14} className="text-[#555] animate-spin" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <p className="text-white/80 text-sm truncate group-hover:text-white transition-colors">
                  {item.query_text}
                </p>
                {item.answer && (
                  <p className="text-[#555] text-xs mt-1 line-clamp-1">{item.answer}</p>
                )}
              </div>

              {/* Meta */}
              <div className="shrink-0 text-right space-y-1">
                {item.critic_scores && (
                  <div className="font-mono text-[10px] text-crimson">
                    {Math.round(item.critic_scores.overall * 100)}%
                  </div>
                )}
                {item.latency_ms && (
                  <div className="font-mono text-[10px] text-[#444]">{item.latency_ms.toFixed(0)}ms</div>
                )}
                <div className="font-mono text-[10px] text-[#333]">
                  {new Date(item.created_at).toLocaleDateString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
