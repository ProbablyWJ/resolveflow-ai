export function composeQuestionAnswer(group, selectedValues = [], customText = '') {
  const selected = Array.from(new Set(selectedValues.map(value => String(value).trim()).filter(Boolean)))
  const text = String(customText || '').trim()
  if (group?.type === 'single_select') return selected[0] || text
  if (group?.type === 'text') return text
  return [...selected, ...(text ? [text] : [])].join('、')
}

export function normalizeQuestionGroup(raw) {
  if (!raw || typeof raw !== 'object' || !raw.id || !raw.type) return null
  const supportedTypes = new Set(['single_select', 'multi_select', 'text'])
  if (!supportedTypes.has(raw.type)) return null
  return {
    id: String(raw.id),
    stage: String(raw.stage || ''),
    type: raw.type,
    prompt: String(raw.prompt || ''),
    options: Array.isArray(raw.options)
      ? raw.options
          .filter(option => option && option.id && option.label && option.value)
          .map(option => ({ id: String(option.id), label: String(option.label), value: String(option.value) }))
      : [],
    required: raw.required !== false,
    allowText: raw.allow_text !== false && raw.allowText !== false,
    textPlaceholder: String(raw.text_placeholder || raw.textPlaceholder || ''),
    submitLabel: String(raw.submit_label || raw.submitLabel || '提交')
  }
}
