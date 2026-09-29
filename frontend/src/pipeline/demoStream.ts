import { PipelinePhase, PipelineResult } from './events'

type PhaseData = Record<string, unknown> | undefined

/**
 * runDemoStream — plays a scripted pipeline run without hitting any
 * backend. Timings mimic realistic agent latencies so the arc feels
 * lived-in during a demo. The scripted answer references BM25 vs
 * dense retrieval, which is one of the two example chips on the hero.
 */
export function runDemoStream(
  query: string,
  onPhase: (phase: PipelinePhase, data?: Record<string, unknown>) => void,
) {
  // Schedule: [delay-ms-from-start, phase, data?]
  const script: Array<[number, PipelinePhase, PhaseData]> = [
    [    0, 'planning',     undefined                                    ],
    [  380, 'retrieving',   undefined                                    ],
    [ 1150, 'executing',    undefined                                    ],
    [ 1550, 'synthesizing', undefined                                    ],
    [ 3050, 'evaluating',   undefined                                    ],
    [ 3950, 'done',         DEMO_RESULT(query) as unknown as PhaseData    ],
  ]
  for (const [delay, phase, data] of script) {
    window.setTimeout(() => onPhase(phase, data), delay)
  }
}

function DEMO_RESULT(query: string): PipelineResult {
  const isBM25Query = /bm25|dense/i.test(query)
  const answer = isBM25Query ? DEMO_ANSWER_BM25 : DEMO_ANSWER_GENERIC(query)
  return {
    answer,
    sources: DEMO_SOURCES,
    critic_scores: {
      groundedness: 0.92, faithfulness: 0.89, completeness: 0.84, overall: 0.88, pass: true,
    },
    verifier_scores: {
      groundedness: 0.91, faithfulness: 0.88, completeness: 0.82, overall: 0.87, pass: true,
    },
    follow_ups: [
      'How is hybrid retrieval fused in Helios?',
      'Show me the reranker used after retrieval.',
      'When would you prefer BM25 alone?',
    ],
    latency_ms: 3950,
  }
}

const DEMO_ANSWER_BM25 = `**BM25** and **dense retrieval** solve the same problem — surfacing relevant documents for a query — with opposite failure modes.

- **BM25** is a term-frequency, inverse-document-frequency scorer. It's fast, deterministic, works out of the box on any tokenised text, and excels at rare terms and exact string matches. It fails on paraphrase: *"heart attack"* won't score high for a document about *"myocardial infarction"*.
- **Dense retrieval** encodes queries and documents as vectors and scores by cosine similarity. It handles paraphrase and semantic proximity gracefully, but is worse at rare identifiers, product SKUs, or unusual proper nouns unless those embeddings were trained in.

Production systems typically **fuse both** with something like **Reciprocal Rank Fusion (RRF)** — sum \`1/(k + rank)\` from each retriever, then re-sort. Helios' Retriever agent runs BM25 and a bi-encoder in parallel and fuses with RRF, plus a CLIP path for images.
`

function DEMO_ANSWER_GENERIC(query: string): string {
  return `Here's what the six agents found for **"${query}"**.

- **Planner** decomposed the question into a retrieval subtask followed by a synthesis subtask.
- **Retriever** pulled 8 chunks — 5 from dense search, 3 from BM25 — fused via RRF.
- **Synthesizer** wrote the answer below citing the three most relevant chunks.
- **Critic** scored the answer for groundedness and faithfulness; both passed.
- **Verifier** independently confirmed the claims with a second-model cross-check.

*(This is a demo answer. The real backend is not connected in demo mode.)*
`
}

const DEMO_SOURCES = [
  {
    title: 'Reciprocal Rank Fusion outperforms individual retrievers',
    domain: 'plg.uwaterloo.ca',
    snippet: 'Cormack et al., SIGIR 2009 — the original RRF paper introducing the k=60 default.',
  },
  {
    title: 'Sparse, Dense, and Attentional Representations for Text Retrieval',
    domain: 'arxiv.org',
    snippet: 'Luan et al., TACL 2021 — comparison of sparse (BM25) vs dense retrieval strengths.',
  },
  {
    title: 'ColBERT: Efficient and Effective Passage Search',
    domain: 'arxiv.org',
    snippet: 'Khattab & Zaharia, SIGIR 2020 — late-interaction dense retrieval alternative.',
  },
]
