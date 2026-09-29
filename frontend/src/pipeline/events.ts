/**
 * Pipeline event types shared between the real backend WebSocket
 * client and the demo-mode scripted stream. The backend today emits
 * five 'active-agent' phases (planning, retrieving, executing,
 * synthesizing, evaluating) plus done/error/retrying. This module
 * fans them out into a per-agent state model the UI can react to.
 */

export type AgentId =
  | 'planner'
  | 'retriever'
  | 'executor'
  | 'synthesizer'
  | 'critic'
  | 'verifier'

export type PipelinePhase =
  | 'idle'
  | 'planning'
  | 'retrieving'
  | 'executing'
  | 'synthesizing'
  | 'evaluating'
  | 'done'
  | 'error'

export interface Score {
  groundedness?: number
  faithfulness?: number
  completeness?: number
  overall?: number
  pass?: boolean
}

export interface Source {
  title: string
  domain?: string
  snippet?: string
  url?: string
}

export interface PipelineResult {
  answer: string
  sources: Source[]
  critic_scores?: Score
  verifier_scores?: Score
  follow_ups?: string[]
  latency_ms?: number
}

export interface PipelineState {
  phase: PipelinePhase
  activeAgents: AgentId[]              // which agents are currently working
  timings: Partial<Record<AgentId, number>>  // ms elapsed per agent (rolling)
  finishedAt: Partial<Record<AgentId, number>>
  startedAt: Partial<Record<AgentId, number>>
  errorMessage?: string
  result?: PipelineResult
  isDemo: boolean
}

export const IDLE_STATE: PipelineState = {
  phase: 'idle',
  activeAgents: [],
  timings: {},
  finishedAt: {},
  startedAt: {},
  isDemo: false,
}

// Which agents each phase activates. `evaluating` runs critic+verifier
// together which matches how the backend evaluates today.
export const PHASE_TO_AGENTS: Record<PipelinePhase, AgentId[]> = {
  idle:         [],
  planning:     ['planner'],
  retrieving:   ['retriever'],
  executing:    ['executor'],
  synthesizing: ['synthesizer'],
  evaluating:   ['critic', 'verifier'],
  done:         [],
  error:        [],
}
