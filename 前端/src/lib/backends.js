import { pollEvaluation } from './evaluationJobs.js'
import { normalizeQuestionGroup } from './questionGroups.js'
const ENV = import.meta.env || {}
const DEFAULT_BACKENDS = {
  python: {
    id: 'python',
    label: 'ResolveFlow API',
    baseUrl: runtimeConfig().pythonApiUrl || ENV.VITE_PYTHON_API_URL || '/api/python',
    port: '8000'
  }
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    pythonToken: '',
    backend: 'python',
    userId: saved.userId || 'u1001',
    // 页面消息未做持久化恢复时，不能暗中续接旧会话。
    // 否则界面显示“新对话”，后端却会读到旧任务状态。
    conversationId: '',
    endpoints: {
      python: saved.endpoints?.python || DEFAULT_BACKENDS.python.baseUrl
    }
  }
}

export function saveSettings(settings) {
  const { conversationId: _conversationId, pythonToken: _pythonToken, ...persistent } = settings
  localStorage.setItem('resolveflow.frontend.settings', JSON.stringify(persistent))
}

export function backendMeta(type, settings) {
  const meta = DEFAULT_BACKENDS[type] || DEFAULT_BACKENDS.python
  return {
    ...meta,
    baseUrl: normalizeBaseUrl(settings.endpoints[type] || meta.baseUrl)
  }
}

export async function requestHealth(type, settings) {
  return requestFor(type, settings, '/health')
}

export async function requestMonitor(type, settings) {
  return requestFor(type, settings, '/monitor')
}

export async function requestProductAnalytics(type, settings) {
  return requestFor(type, settings, '/analytics/summary')
}

export async function recordInteractionEvent(type, settings, event) {
  return requestFor(type, settings, '/analytics/events', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...event,
      user_id: settings.userId || 'anonymous'
    })
  })
}

export async function requestSkills(type, settings) {
  return requestFor(type, settings, '/skills')
}

export async function reloadSkills(type, settings) {
  return requestFor(type, settings, '/skills/reload', { method: 'POST' })
}

export async function requestKnowledgeStats(type, settings) {
  return requestFor(type, settings, '/knowledge/stats')
}

export async function requestLatestEvaluation(type, settings) {
  return requestFor(type, settings, '/eval/latest')
}

export async function runEvaluation(type, settings, body = null, options = {}) {
  const taskKey = `resolveflow.eval:${type}:${backendMeta(type, settings).baseUrl}:${settings.userId}`
  const savedJob = sessionStorage.getItem(taskKey)
  const initial = savedJob ? {job_id: savedJob, status: 'queued', completed: 0, total: 0} : await requestFor(type, settings, '/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
  if (!initial.job_id) return initial
  sessionStorage.setItem(taskKey, initial.job_id)
  try {
    const result = await pollEvaluation(initial, jobId => requestFor(type, settings,
      `/eval/jobs/${encodeURIComponent(jobId)}`), options)
    if (result) sessionStorage.removeItem(taskKey)
    return result
  } catch (error) {
    if (error.terminal || error.status === 404) sessionStorage.removeItem(taskKey)
    throw error
  }
}

export async function requestSearch(type, settings, query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  return requestFor(type, settings, `/search?${params}`, { method: 'POST' })
}

export async function requestChat(type, settings, message) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  const attemptKey = JSON.stringify([type, meta.baseUrl, settings.userId, settings.conversationId, message])
  if (type === 'python') {
    if (!chatAttempts.has(attemptKey)) {
      if (chatAttempts.size > 100) chatAttempts.clear()
      chatAttempts.set(attemptKey, crypto.randomUUID())
    }
    payload.client_request_id = chatAttempts.get(attemptKey)
  }
  const raw = await requestFor(type, settings, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  chatAttempts.delete(attemptKey)
  return normalizeChatResponse(type, raw)
}

export async function requestToolTrace(type, settings, requestId) {
  if (!requestId) return null
  const raw = await requestFor(type, settings, `/trace/tool/${encodeURIComponent(requestId)}`)
  return normalizeToolTraceResponse(raw)
}

export async function addKnowledge(type, settings, documents) {
  return requestFor(type, settings, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(type, settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestFor(type, settings, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

function buildChatPayload(type, settings, message) {
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conv_id: settings.conversationId || undefined
  }
}

export function normalizeChatResponse(type, raw) {
  return {
    backend: type,
    conversationId: raw.conversation_id || raw.conversationId || raw.conv_id || '',
    requestId: raw.request_id || raw.requestId || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    intentGroup: raw.intent_group || raw.intentGroup || 'other',
    agentType: raw.agent_type || raw.agentType || '',
    agentTypes: raw.agent_types || raw.agentTypes || [],
    primaryAgent: raw.primary_agent || raw.primaryAgent || '',
    supportingAgents: raw.supporting_agents || raw.supportingAgents || [],
    routingReason: raw.routing_reason || raw.routingReason || '',
    routingConfidence: Number(raw.routing_confidence ?? raw.routingConfidence ?? 0),
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? raw.intentConfidence ?? 0),
    intentSourceScores: raw.intent_source_scores || raw.intentSourceScores || {},
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    taskState: raw.task_state || raw.taskState || {},
    questionGroup: normalizeQuestionGroup(raw.question_group || raw.questionGroup),
    status: raw.status || '',
    retryable: Boolean(raw.retryable),
    errorCode: raw.error_code || raw.errorCode || null,
    verified: raw.verified,
    grounded: raw.grounded,
    raw
  }
}

function normalizeToolTraceResponse(raw) {
  const trace = raw?.trace || {}
  return {
    requestId: raw?.request_id || raw?.requestId || '',
    found: Boolean(raw?.found),
    trace: {
      ...trace,
      toolsUsed: trace.tools_used || trace.toolsUsed || [],
      toolCalls: trace.tool_calls || trace.toolCalls || [],
      taskState: trace.task_state || trace.taskState || {}
    },
    raw
  }
}

const chatAttempts = new Map()
async function requestFor(type, settings, path, options = {}) {
  const headers = new Headers(options.headers || {})
  if (type === 'python' && settings.pythonToken) headers.set('Authorization', `Bearer ${settings.pythonToken}`)
  return requestJson(backendMeta(type, settings).baseUrl, path, { ...options, headers,
    signal: options.signal || AbortSignal.timeout(path === '/chat' ? 55000 : 20000) })
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    const error = new Error(`${response.status} ${response.statusText}: ${detail}`)
    error.status = response.status
    throw error
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    const saved = localStorage.getItem('resolveflow.frontend.settings')
    return JSON.parse(saved || '{}')
  } catch {
    return {}
  }
}

function runtimeConfig() {
  if (typeof window === 'undefined') return {}
  return window.__RESOLVEFLOW_CONFIG__ || {}
}
