import { useEffect, useState, useCallback } from 'react'
import { Clock, CheckCircle, XCircle, Loader, Search, ChevronDown } from 'lucide-react'
import { queries, type HistoryItem } from '../api/client'

function SkeletonRow() {
  return (
    <div className="card-dark border border-white/5 p-4 flex items-start gap-4 animate-pulse">
      <div className="w-6 h-3 bg-white/5 rounded" />
      <div className="w-3.5 h-3.5 bg-white/5 rounded-full mt-0.5" />
      <div className="flex-1 space-y-2">
        <div className="h-3 bg-white/5 rounded w-3/4" />
        <div className="h-2 bg-white/5 rounded w-1/2" />
      </div>
      <div className="space-y-1 text-right">
        <div className="h-2 bg-white/5 rounded w-10 ml-auto" />
        <div className="h-2 bg-white/5 rounded w-8 ml-auto" />
      </div>
    </div>
  )
}

export function HistorySection({ isLoggedIn, refreshTrigger }: {
  isLoggedIn: boolean
  refreshTrigger: number
}) {
  const [items, setItems] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(false)
  const [search, setSearch] = useState('')
  const PAGE = 10

  const fetchPage = useCallback(async (offset: number, replace: boolean) => {
    const setter = offset === 0 ? setLoading : setLoadingMore
    setter(true)
    try {
      const { data } = await queries.history(PAGE + 1, offset)
      const page = data.slice(0, PAGE)
      setHasMore(data.length > PAGE)
      setItems(prev => replace ? page : [...prev, ...page])
    } catch { /* ignore */ }
    finally { setter(false) }
  }, [])

  useEffect(() => {
    if (!isLoggedIn) return
    setItems([])
    fetchPage(0, true)
  }, [isLoggedIn, refreshTrigger, fetchPage])

  if (!isLoggedIn) return null

  const filtered = search.trim()
    ? items.filter(it => it.query_text.toLowerCase().includes(search.toLowerCase()))
    : items

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

        {/* Search */}
        <div className="relative mb-6">
          <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#444]" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter queries..."
            className="w-full bg-white/3 border border-white/10 text-white font-mono text-xs pl-8 pr-3 py-2 outline-none focus:border-crimson/30 transition-colors placeholder-[#333]"
          />
        </div>

        {loading && (
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => <SkeletonRow key={i} />)}
          </div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="card-dark border border-white/5 p-8 text-center">
            <Clock size={24} className="text-[#333] mx-auto mb-3" />
            <p className="text-[#555] font-mono text-sm">
              {search ? 'No matching queries.' : 'No queries yet. Run your first one above.'}
            </p>
          </div>
        )}

        <div className="space-y-2">
          {filtered.map((item, i) => (
            <div key={item.id} className="card-dark border border-white/5 p-4 flex items-start gap-4 hover:border-crimson/20 transition-all group">
              <span className="font-mono text-[#444] text-xs shrink-0 w-6 text-right">{String(i + 1).padStart(2, '0')}</span>

              <div className="shrink-0 mt-0.5">
                {item.status === 'done' ? (
                  <CheckCircle size={14} className="text-green-500" />
                ) : item.status === 'failed' ? (
                  <XCircle size={14} className="text-crimson" />
                ) : (
                  <Loader size={14} className="text-[#555] animate-spin" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-white/80 text-sm truncate group-hover:text-white transition-colors">
                  {item.query_text}
                </p>
                {item.answer && (
                  <p className="text-[#555] text-xs mt-1 line-clamp-1">{item.answer}</p>
                )}
              </div>

              <div className="shrink-0 text-right space-y-1">
                {item.critic_scores && (
                  <div className={`font-mono text-[10px] ${
                    (item.critic_scores.overall ?? 0) >= 0.7 ? 'text-green-400' :
                    (item.critic_scores.overall ?? 0) >= 0.4 ? 'text-yellow-400' : 'text-crimson'
                  }`}>
                    {Math.round((item.critic_scores.overall ?? 0) * 100)}%
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

        {hasMore && !search && (
          <div className="mt-4 text-center">
            <button
              onClick={() => fetchPage(items.length, false)}
              disabled={loadingMore}
              className="flex items-center gap-2 mx-auto font-mono text-[10px] text-[#555] hover:text-crimson transition-colors uppercase disabled:opacity-40"
            >
              {loadingMore ? <Loader size={12} className="animate-spin" /> : <ChevronDown size={12} />}
              {loadingMore ? 'LOADING...' : 'LOAD MORE'}
            </button>
          </div>
        )}
      </div>
    </section>
  )
}
