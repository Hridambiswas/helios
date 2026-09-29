import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
  ReactNode,
} from 'react'
import {
  AgentId, PHASE_TO_AGENTS, PipelinePhase, PipelineResult, PipelineState,
  IDLE_STATE,
} from './events'
import { runDemoStream } from './demoStream'

/**
 * PipelineProvider — single source of truth for the current query's
 * pipeline state. Wraps everything that needs to react to agent
 * events (Sol's mascot rig, the sun-path arc, the answer view).
 *
 * Two backends:
 *  • Demo mode (VITE_DEMO_MODE=true) — the run() call plays a scripted
 *    event stream via runDemoStream(). No network, no auth needed.
 *  • Live mode — TODO in a follow-up commit; wires the same setter to
 *    the real /ws/query WebSocket. The shape of the state doesn't
 *    change, so consumers work in both modes.
 */

interface PipelineApi {
  state: PipelineState
  run: (query: string) => void
  reset: () => void
  isDemoMode: boolean
}

const PipelineContext = createContext<PipelineApi | null>(null)

export function usePipeline() {
  const ctx = useContext(PipelineContext)
  if (!ctx) throw new Error('usePipeline must be used inside PipelineProvider')
  return ctx
}

const IS_DEMO = import.meta.env.VITE_DEMO_MODE === 'true'

export function PipelineProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PipelineState>({ ...IDLE_STATE, isDemo: IS_DEMO })
  const runningRef = useRef(false)
  const timingIntervalRef = useRef<number | null>(null)

  const clearTimingLoop = useCallback(() => {
    if (timingIntervalRef.current !== null) {
      window.clearInterval(timingIntervalRef.current)
      timingIntervalRef.current = null
    }
  }, [])

  // Bump `timings` every 100ms for whichever agents are currently active.
  const ensureTimingLoop = useCallback(() => {
    if (timingIntervalRef.current !== null) return
    timingIntervalRef.current = window.setInterval(() => {
      setState(prev => {
        if (prev.phase === 'idle' || prev.phase === 'done' || prev.phase === 'error') {
          return prev
        }
        const now = performance.now()
        const timings = { ...prev.timings }
        for (const agent of prev.activeAgents) {
          const start = prev.startedAt[agent]
          if (start !== undefined) timings[agent] = now - start
        }
        return { ...prev, timings }
      })
    }, 100)
  }, [])

  const applyPhase = useCallback((phase: PipelinePhase, data?: Record<string, unknown>) => {
    setState(prev => {
      const now = performance.now()
      const nextAgents = PHASE_TO_AGENTS[phase]
      const startedAt = { ...prev.startedAt }
      const finishedAt = { ...prev.finishedAt }

      // Any agents that were active but aren't in the next phase are done.
      for (const a of prev.activeAgents) {
        if (!nextAgents.includes(a)) finishedAt[a] = now
      }
      // Any agents that are newly active get a start timestamp.
      for (const a of nextAgents) {
        if (startedAt[a] === undefined) startedAt[a] = now
      }

      let result: PipelineResult | undefined = prev.result
      let errorMessage: string | undefined = prev.errorMessage
      if (phase === 'done' && data) {
        result = data as unknown as PipelineResult
      }
      if (phase === 'error') {
        errorMessage = (data?.message as string) ?? 'Something went wrong.'
      }

      return {
        ...prev,
        phase,
        activeAgents: nextAgents,
        startedAt,
        finishedAt,
        result,
        errorMessage,
      }
    })
  }, [])

  const reset = useCallback(() => {
    clearTimingLoop()
    runningRef.current = false
    setState({ ...IDLE_STATE, isDemo: IS_DEMO })
  }, [clearTimingLoop])

  const run = useCallback((query: string) => {
    if (runningRef.current) return
    runningRef.current = true
    clearTimingLoop()
    setState({ ...IDLE_STATE, isDemo: IS_DEMO, phase: 'planning', activeAgents: PHASE_TO_AGENTS.planning, startedAt: { planner: performance.now() } })
    ensureTimingLoop()

    if (IS_DEMO) {
      runDemoStream(query, (phase, data) => {
        applyPhase(phase, data)
        if (phase === 'done' || phase === 'error') {
          runningRef.current = false
          clearTimingLoop()
        }
      })
      return
    }

    // Live-WebSocket branch: wired in a follow-up commit.
    // For now, hand the query back so the existing ChatPage flow still
    // uses its own WebSocket path.
    runningRef.current = false
    clearTimingLoop()
    setState({ ...IDLE_STATE, isDemo: IS_DEMO })
  }, [applyPhase, clearTimingLoop, ensureTimingLoop])

  useEffect(() => () => clearTimingLoop(), [clearTimingLoop])

  const api = useMemo<PipelineApi>(() => ({
    state,
    run,
    reset,
    isDemoMode: IS_DEMO,
  }), [state, run, reset])

  return (
    <PipelineContext.Provider value={api}>{children}</PipelineContext.Provider>
  )
}

export type { AgentId }
