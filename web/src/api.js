// Fetch helpers for the read-only registry API.
//
// Every call here is a GET. The UI has no write path by design: the
// registry's integrity rules are enforced where rows are written, and a
// console that could write would be a second, unpoliced way into the tables.

async function getJson(url) {
  const response = await fetch(url)
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return response.json()
}

export const fetchFunnel = () => getJson('/api/funnel')
export const fetchIdeas = () => getJson('/api/ideas')
export const fetchIdea = (ideaId) => getJson(`/api/ideas/${encodeURIComponent(ideaId)}`)

export const fetchIdeaPage = (ideaId, resource, { limit, offset }) =>
  getJson(
    `/api/ideas/${encodeURIComponent(ideaId)}/${resource}?limit=${limit}&offset=${offset}`,
  )

export const fetchTrials = ({ limit, offset }) =>
  getJson(`/api/trials?limit=${limit}&offset=${offset}`)

export const fetchCards = () => getJson('/api/cards')
export const fetchCard = (ideaId) => getJson(`/api/cards/${encodeURIComponent(ideaId)}`)
