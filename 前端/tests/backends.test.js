import test from 'node:test'
import assert from 'node:assert/strict'
import { createInitialSettings, normalizeChatResponse, saveSettings } from '../src/lib/backends.js'

test('normalizes after-sales task state for the product inspector', () => {
  const result = normalizeChatResponse('python', {
    conv_id: 'conv-1',
    request_id: 'req-1',
    response: '已完成核验',
    intent: 'exchange_request',
    primary_agent: 'diagnostic',
    supporting_agents: ['resolution'],
    task_state: {
      facts: { order_verified: true, warranty_status: 'in_warranty' },
      resolution_plan: { status: 'awaiting_action', steps: [{ id: 'verify', status: 'completed' }] }
    },
    question_group: {
      id: 'req-1:safety_check',
      stage: 'safety_check',
      type: 'single_select',
      prompt: '请选择当前安全情况',
      options: [{ id: 'yes', label: '有', value: '有' }]
    }
  })

  assert.equal(result.conversationId, 'conv-1')
  assert.equal(result.primaryAgent, 'diagnostic')
  assert.deepEqual(result.supportingAgents, ['resolution'])
  assert.equal(result.taskState.facts.order_verified, true)
  assert.equal(result.taskState.resolution_plan.steps[0].status, 'completed')
  assert.equal(result.questionGroup.type, 'single_select')
  assert.equal(result.questionGroup.options[0].value, '有')
})

test('a page reload never silently resumes a conversation that is not rendered', () => {
  let stored = JSON.stringify({
    backend: 'python',
    userId: 'u1001',
    conversationId: 'stale-conversation',
    endpoints: { python: '/api/python' }
  })
  globalThis.localStorage = {
    getItem: () => stored,
    setItem: (_key, value) => { stored = value }
  }

  const settings = createInitialSettings()
  assert.equal(settings.conversationId, '')
  settings.conversationId = 'current-session-only'
  saveSettings(settings)
  assert.equal(JSON.parse(stored).conversationId, undefined)
})
