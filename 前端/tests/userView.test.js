import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'

const appSource = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
const inspectorSource = fs.readFileSync(new URL('../src/components/RequestInspector.vue', import.meta.url), 'utf8')
const questionGroupSource = fs.readFileSync(new URL('../src/components/QuestionGroup.vue', import.meta.url), 'utf8')

test('customer chat hides internal routing metadata and tool traces by default', () => {
  const conversation = appSource.slice(
    appSource.indexOf('<section class="conversation"'),
    appSource.indexOf('<aside v-if=')
  )

  assert.doesNotMatch(conversation, /currentBackend\.label \}\} Agent/)
  assert.doesNotMatch(conversation, /class="agent-tag"/)
  assert.doesNotMatch(conversation, /class="message-trace"/)
  assert.match(appSource, /const showInternals = ref\(false\)/)
  assert.match(appSource, /messages\.length && showInternals/)
})

test('product surface exposes one ResolveFlow service instead of legacy backend choices', () => {
  assert.doesNotMatch(appSource, /选择后端/)
  assert.doesNotMatch(appSource, /\['java', 'python'\]/)
  assert.match(appSource, /本地演示环境/)
})

test('handoff details stay available in the internal request inspector', () => {
  assert.match(inspectorSource, /人工交接摘要/)
  assert.match(inspectorSource, /state\.value\.handoff_summary/)
  assert.match(inspectorSource, /待接入人工队列/)
  assert.match(inspectorSource, /已识别为新任务/)
  assert.match(inspectorSource, /state\.value\.task_transition/)
})

test('customer chat renders structured single, multi and free-text question groups', () => {
  assert.match(appSource, /<QuestionGroup/)
  assert.match(questionGroupSource, /single_select/)
  assert.match(questionGroupSource, /multi_select/)
  assert.match(questionGroupSource, /question-custom/)
  assert.match(questionGroupSource, /@submit\.prevent="submit"/)
})

test('clarification flow supports back, abandon and keyboard escape without exposing internals', () => {
  assert.match(questionGroupSource, /返回上一问题/)
  assert.match(questionGroupSource, /结束本次流程/)
  assert.match(questionGroupSource, /emit\('back'\)/)
  assert.match(questionGroupSource, /emit\('abandon'\)/)
  assert.match(appSource, /returnToPreviousQuestion/)
  assert.match(appSource, /abandonCurrentFlow/)
  assert.match(appSource, /@keydown\.esc\.prevent="abandonCurrentFlow\(\)"/)
})

test('evaluation surface exposes product funnel and trajectory metrics', () => {
  assert.match(appSource, /澄清完成率/)
  assert.match(appSource, /平均澄清轮次/)
  assert.match(appSource, /流程放弃率/)
  assert.match(appSource, /人工升级率/)
  assert.match(appSource, /trajectory_pass_rate: '调用链通过率'/)
})
