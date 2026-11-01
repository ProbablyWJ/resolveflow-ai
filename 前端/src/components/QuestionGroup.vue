<template>
  <fieldset class="question-group" :disabled="disabled" :aria-label="group.prompt">
    <legend>{{ group.prompt }}</legend>

    <div v-if="group.options.length" class="question-options" :class="`is-${group.type}`">
      <button
        v-for="option in group.options"
        :key="option.id"
        type="button"
        class="question-option"
        :class="{ selected: selected.includes(option.value) }"
        :aria-pressed="selected.includes(option.value)"
        @click="choose(option.value)"
      >
        <span v-if="group.type === 'multi_select'" class="option-check" aria-hidden="true">
          {{ selected.includes(option.value) ? '✓' : '' }}
        </span>
        {{ option.label }}
      </button>
    </div>

    <form v-if="group.allowText || group.type === 'text'" class="question-custom" @submit.prevent="submit">
      <input
        v-model="customText"
        :placeholder="group.textPlaceholder || '直接填写'"
        :aria-label="group.textPlaceholder || '直接填写'"
      />
      <button
        v-if="group.type === 'text' || group.type === 'multi_select' || customText.trim()"
        type="submit"
        class="question-submit"
        :disabled="!answer"
      >
        {{ group.submitLabel }}
      </button>
    </form>

    <button
      v-if="group.type === 'multi_select' && !group.allowText"
      type="button"
      class="question-submit"
      :disabled="!answer"
      @click="submit"
    >
      {{ group.submitLabel }}
    </button>

    <div v-if="!disabled" class="question-flow-actions">
      <button
        v-if="group.stage !== 'safety_check'"
        type="button"
        class="question-secondary-action"
        @click="emit('back')"
      >
        返回上一问题
      </button>
      <button
        type="button"
        class="question-secondary-action is-abandon"
        @click="emit('abandon')"
      >
        结束本次流程
      </button>
    </div>

    <small v-if="disabled" class="question-answered">已回答</small>
  </fieldset>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { composeQuestionAnswer } from '../lib/questionGroups'

const props = defineProps({
  group: { type: Object, required: true },
  disabled: { type: Boolean, default: false }
})
const emit = defineEmits(['submit', 'back', 'abandon'])
const selected = ref([])
const customText = ref('')
const answer = computed(() => composeQuestionAnswer(props.group, selected.value, customText.value))

watch(() => props.group.id, () => {
  selected.value = []
  customText.value = ''
})

function choose(value) {
  if (props.disabled) return
  if (props.group.type === 'single_select') {
    selected.value = [value]
    emit('submit', value)
    return
  }
  selected.value = selected.value.includes(value)
    ? selected.value.filter(item => item !== value)
    : [...selected.value, value]
}

function submit() {
  if (props.disabled || !answer.value) return
  emit('submit', answer.value)
}
</script>
