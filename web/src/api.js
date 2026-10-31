// Fetch helpers for the read-only registry API.
//
// Every call here is a GET, except the three on the pipeline page (below) that start a night. The UI
// has no write path into the registry by design: the registry's integrity rules are enforced where
// rows are written, and a console that could write would be a second, unpoliced way into the tables.

async function getJson(url) {
  const response = await fetch(url)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return response.json()
}

export const fetchFunnel = () => getJson('/api/funnel')
export const fetchAgentFunnel = () => getJson('/api/agent-funnel')
export const fetchAgents = (days) => getJson(`/api/agents?days=${days}`)
export const fetchIdeas = () => getJson('/api/ideas')
export const fetchIdea = (ideaId) => getJson(`/api/ideas/${encodeURIComponent(ideaId)}`)

export const fetchForward = (ideaId) =>
  getJson(`/api/ideas/${encodeURIComponent(ideaId)}/forward`)

export const fetchOverlays = (ideaId) =>
  getJson(`/api/ideas/${encodeURIComponent(ideaId)}/overlays`)

export const fetchIdeaPage = (ideaId, resource, { limit, offset }) =>
  getJson(
    `/api/ideas/${encodeURIComponent(ideaId)}/${resource}?limit=${limit}&offset=${offset}`,
  )

export const fetchTrials = ({ limit, offset, noise = false }) =>
  getJson(`/api/trials?limit=${limit}&offset=${offset}&noise=${noise}`)

export const fetchCards = () => getJson('/api/cards')
export const fetchCard = (ideaId) => getJson(`/api/cards/${encodeURIComponent(ideaId)}`)

// The pipeline page (docs/PIPELINE.md). The three POSTs start a night on its own process; they carry the
// console's header, which the API requires and which a page on another site cannot send.
async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Qlab-Console': '1' },
    body: JSON.stringify(body),
  })
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = null
  }
  if (!response.ok) {
    throw new Error(data?.detail ?? `${response.status} ${response.statusText}: ${text}`)
  }
  return data
}

export const fetchPipeline = () => getJson('/api/pipeline')
export const fetchPipelineStep = (stepId) =>
  getJson(`/api/pipeline/steps/${encodeURIComponent(stepId)}`)
export const fetchPipelineStepRunItems = (stepRunId) =>
  getJson(`/api/pipeline/step-runs/${encodeURIComponent(stepRunId)}/items`)
export const fetchPipelineRuns = () => getJson('/api/pipeline/runs?limit=20')
export const previewPipeline = ({ off = [], restart = null }) =>
  postJson('/api/pipeline/preview', { off, restart })
export const runPipeline = ({ off = [] }) => postJson('/api/pipeline/run', { off })
export const restartStep = ({ step, off = [] }) =>
  postJson('/api/pipeline/restart', { step, off })
