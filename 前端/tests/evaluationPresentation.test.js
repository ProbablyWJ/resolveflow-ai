import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'

const appSource = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')

test('evaluation page separates dataset coverage from result records', () => {
  assert.match(appSource, /dataset_summary\?\.intent_cases/)
  assert.match(appSource, /dataset_summary\?\.intent_labels/)
  assert.match(appSource, /dataset_summary\?\.dialog_scenarios/)
  assert.match(appSource, /dataset_summary\?\.dialog_turns/)
  assert.match(appSource, /结果记录 \/ 评分失败/)
})

test('evaluation page discloses confidence interval and class support', () => {
  assert.match(appSource, /accuracy_ci95/)
  assert.match(appSource, /class_support/)
  assert.match(appSource, /95% 置信区间/)
  assert.match(appSource, /不再将小样本全对等同于稳定 100%/)
})
