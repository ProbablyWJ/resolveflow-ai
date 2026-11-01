import test from 'node:test'
import assert from 'node:assert/strict'
import { composeQuestionAnswer, normalizeQuestionGroup } from '../src/lib/questionGroups.js'

test('normalizes the structured question group returned by the API', () => {
  const group = normalizeQuestionGroup({
    id: 'req-1:safety_check',
    stage: 'safety_check',
    type: 'single_select',
    prompt: '请选择当前安全情况',
    options: [{ id: 'yes', label: '有', value: '有' }],
    allow_text: true,
    text_placeholder: '直接描述',
    submit_label: '提交'
  })

  assert.equal(group.type, 'single_select')
  assert.equal(group.options[0].value, '有')
  assert.equal(group.allowText, true)
  assert.equal(group.textPlaceholder, '直接描述')
})

test('composes multi-select and free-text answers into one user message', () => {
  const group = { type: 'multi_select' }
  assert.equal(composeQuestionAnswer(group, ['明显发烫', '烧焦味'], '外壳变形'), '明显发烫、烧焦味、外壳变形')
  assert.equal(composeQuestionAnswer({ type: 'text' }, [], 'FlowBuds Pro'), 'FlowBuds Pro')
})
