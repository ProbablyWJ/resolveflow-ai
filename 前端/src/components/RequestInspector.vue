<template>
  <div class="request-inspector">
    <section class="inspector-section">
      <div class="inspector-section-heading">
        <h3>请求追踪</h3>
        <span class="state-label">
          <i :class="{ working: busy, complete: lastResponse && !busy }"></i>
          {{ busy ? '处理中' : lastResponse ? '已完成' : '待开始' }}
        </span>
      </div>
      <p class="inspector-intro">{{ busy ? '正在识别意图、调用工具并更新任务状态。' : lastResponse ? '以下是本轮真实运行的决策链路。' : '发送一条消息，查看 Agent 如何做决策。' }}</p>
      <div v-if="lastResponse" class="routing-summary">
        <span>{{ intentLabel }}</span>
        <AppIcon name="chevron" />
        <strong>{{ agentLabel }}</strong>
        <span v-if="lastResponse.supportingAgents?.length">+ {{ lastResponse.supportingAgents.map(labelAgent).join('、') }}</span>
      </div>
      <div v-if="taskTransition?.type === 'new_task'" class="task-transition">
        <strong>已识别为新任务</strong>
        <p>旧任务事实、工单与工具状态未带入本轮 · {{ transitionReason }}</p>
      </div>
      <div v-if="lastResponse?.escalated || plan.status === 'handoff_required'" class="safety-stop">
        <strong>自动处理已停止</strong>
        <p>安全风险或人工请求已触发升级，不再继续普通工具操作。</p>
      </div>
      <ol class="trace-steps task-plan" :class="{ complete: lastResponse && !busy }">
        <li v-for="step in planSteps" :key="step.id" :class="`step-${step.status}`">
          <span class="trace-step-icon"><AppIcon :name="step.icon" /></span>
          <div>
            <strong>{{ step.label }}</strong>
            <p>{{ step.description }}</p>
          </div>
          <span class="step-status">{{ statusLabels[step.status] || step.status }}</span>
        </li>
      </ol>
      <div v-if="factRows.length" class="fact-board">
        <h4>已核验事实</h4>
        <dl>
          <template v-for="item in factRows" :key="item.label">
            <dt>{{ item.label }}</dt>
            <dd>{{ item.value }}</dd>
          </template>
        </dl>
      </div>
      <div v-if="pendingAction" class="pending-action">
        <strong>待用户确认</strong>
        <p>{{ ticketTypeLabel }} · {{ pendingActionStatus }}</p>
        <small>未确认前不会执行写操作</small>
      </div>
      <div v-if="handoffSummary" class="handoff-summary">
        <h4>人工交接摘要</h4>
        <dl>
          <dt>状态</dt>
          <dd>{{ handoffSummary.queue_connected ? '已接入人工队列' : '摘要已整理，待接入人工队列' }}</dd>
          <dt>原因</dt>
          <dd>{{ handoffReason }}</dd>
          <dt>风险</dt>
          <dd>{{ urgencyLabel }}</dd>
          <dt v-if="handoffSummary.completed_steps?.length">已完成</dt>
          <dd v-if="handoffSummary.completed_steps?.length">{{ handoffSummary.completed_steps.join(' · ') }}</dd>
        </dl>
      </div>
      <details v-if="evidenceRefs.length" class="evidence-list">
        <summary>决策依据 <span class="count-tag">{{ evidenceRefs.length }}</span></summary>
        <p v-for="ref in evidenceRefs" :key="ref">{{ ref }}</p>
      </details>
      <div class="request-metrics">
        <div>
          <span>响应耗时</span>
          <strong>
            {{ lastResponse ? lastResponse.latencyMs : '—' }}
            <small v-if="lastResponse"> ms</small>
          </strong>
        </div>
        <div>
          <span>路由置信度</span>
          <strong>{{ lastResponse ? formatPercent(lastResponse.routingConfidence) : '—' }}</strong>
        </div>
      </div>
      <p v-if="lastResponse?.routingReason" class="routing-note">{{ lastResponse.routingReason }}</p>
      <details v-if="lastTrace?.trace" class="inspector-tools">
        <summary>
          <span>工具调用</span>
          <span class="count-tag">{{ lastTrace.trace.toolCalls?.length || 0 }}</span>
          <AppIcon name="chevron" />
        </summary>
        <div v-for="(call, index) in lastTrace.trace.toolCalls" :key="index" class="trace-call-item">
          <div class="trace-call-meta">
            <strong>{{ toolLabels[call.tool_name] || call.tool_name }}</strong>
            <span :class="call.success ? 'success' : 'danger'">{{ call.success ? '成功' : '失败' }} · {{ call.latency_ms ?? '—' }} ms</span>
          </div>
          <pre>{{ formatJson(call.input || {}) }}</pre>
        </div>
        <p v-if="!lastTrace.trace.toolCalls?.length" class="muted">{{ lastTrace.trace.toolsUsed?.length ? lastTrace.trace.toolsUsed.join(' · ') : '本次没有记录到工具输入。' }}</p>
      </details>
    </section>
    <section class="inspector-section">
      <div class="inspector-section-heading">
        <h3>运行状态</h3>
        <button class="icon-button" @click="$emit('refresh')" aria-label="刷新运行状态" title="刷新运行状态">
          <AppIcon name="refresh" />
        </button>
      </div>
      <div class="runtime-metrics">
        <div>
          <span>请求</span>
          <strong>{{ monitorData.loaded ? totalRequests : '—' }}</strong>
        </div>
        <div>
          <span>Agent</span>
          <strong>{{ monitorData.loaded ? agentCount : '—' }}</strong>
        </div>
        <div>
          <span>告警</span>
          <strong :class="{ danger: activeAlerts.length }">{{ monitorData.loaded ? activeAlerts.length : '—' }}</strong>
        </div>
      </div>
      <p v-if="activeAlerts.length" class="alert-note">
        <AppIcon name="alert" />
        {{ activeAlerts[0].detail || activeAlerts[0].title }}
      </p>
      <p v-else class="runtime-note">{{ monitorData.loaded ? '当前没有活跃告警' : '连接服务后查看运行状态' }}</p>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import AppIcon from './AppIcon.vue'

const props = defineProps({
  lastResponse: { type: Object, default: null },
  lastTrace: { type: Object, default: null },
  busy: { type: Boolean, default: false },
  monitorData: { type: Object, required: true }
})
defineEmits(['refresh'])
const activeAlerts = computed(() => props.monitorData.active_alerts || [])
const agentCount = computed(() => Object.keys(props.monitorData.agent_stats || {}).length)
const totalRequests = computed(() => Object.values(props.monitorData.agent_stats || {}).reduce((sum, item) => sum + Number(item.total || 0), 0))
const state = computed(() => props.lastResponse?.taskState || props.lastTrace?.trace?.taskState || {})
const plan = computed(() => state.value.resolution_plan || {})
const facts = computed(() => state.value.facts || {})
const entities = computed(() => state.value.entities || {})
const pendingAction = computed(() => state.value.pending_action || null)
const evidenceRefs = computed(() => state.value.evidence_refs || [])
const handoffSummary = computed(() => state.value.handoff_summary || null)
const taskTransition = computed(() => state.value.task_transition || null)
const handoffReasons = { device_safety_risk: '设备安全风险', user_requested_human: '用户要求人工服务', manual_review_required: '需要人工核验' }
const handoffReason = computed(() => handoffReasons[handoffSummary.value?.reason_code] || handoffSummary.value?.reason || '需要人工处理')
const transitionReasons = { changed_order: '订单已变更', changed_device: '设备已变更', explicit_new_task: '用户明确开启新任务', left_ticket_tracking: '从工单查询转入新问题', left_handoff_task: '人工交接后开启新问题', changed_service_type: '服务类型已变更', new_fault_after_terminal_task: '旧任务结束后出现新故障' }
const transitionReason = computed(() => transitionReasons[taskTransition.value?.reason] || '业务目标已变更')
const agentNames = { intake: '接待 Agent', diagnostic: '诊断 Agent', resolution: '方案 Agent', escalation: '人工协同 Agent', general: '接待 Agent', technical: '诊断 Agent', billing: '方案 Agent' }
const intentNames = { aftersales: '售后咨询', device_fault: '设备故障', warranty_inquiry: '保修查询', repair_request: '维修申请', exchange_request: '换货申请', parts_request: '配件服务', ticket_status: '工单查询', quality_complaint: '质量投诉', human_handoff: '人工升级' }
const planLabels = {
  collect: ['收集信息', '获取订单、设备和故障现象', 'search'],
  verify: ['核验业务事实', '查询订单、设备与保修状态', 'check'],
  retrieve: ['检索规则与知识', '检索产品手册和售后政策', 'book'],
  diagnose: ['诊断问题', '执行安全分级和故障排查', 'sliders'],
  decide: ['生成处理方案', '根据事实和政策推荐路径', 'route'],
  execute: ['确认后执行', '创建工单或转人工协同', 'chat']
}
const statusLabels = { completed: '已完成', in_progress: '进行中', pending: '待执行', blocked: '已阻断', not_required: '无需执行' }
const toolLabels = { lookup_order: '查询订单', lookup_device: '查询设备', check_warranty: '核验保修', lookup_fault_code: '查询故障码', run_device_diagnostic: '设备诊断', search_knowledge_base: '检索知识库', recommend_resolution: '生成售后方案', prepare_service_ticket: '准备工单', create_service_ticket: '创建工单', get_service_ticket: '查询工单', create_handoff_summary: '生成人工交接摘要' }
const resolutionLabels = { manual_exchange_review: '换货人工审核', in_warranty_repair: '在保维修', paid_repair_quote: '付费维修报价', paid_repair_or_manual_review: '付费维修或人工复核', parts_service_review: '配件服务核验' }
const ticketStatusLabels = { created: '已创建', awaiting_review: '等待售后审核', awaiting_shipment: '等待寄出设备', in_transit: '运输中', repairing: '维修中', completed: '已完成', cancelled: '已取消' }
const ticketTypeLabels = { exchange: '换货工单', repair: '维修工单', parts: '配件服务工单' }
const pendingStatusLabels = { awaiting_user_confirmation: '等待用户确认' }
const urgencyLabels = { CRITICAL: '高风险', HIGH: '较高', MEDIUM: '一般', LOW: '较低' }
const intentLabel = computed(() => intentNames[props.lastResponse?.intent] || props.lastResponse?.intent || '待识别')
const agentLabel = computed(() => labelAgent(props.lastResponse?.primaryAgent || props.lastResponse?.agentType || ''))
const ticketTypeLabel = computed(() => ticketTypeLabels[pendingAction.value?.ticket_type] || '售后工单')
const pendingActionStatus = computed(() => pendingStatusLabels[pendingAction.value?.status] || '等待用户确认')
const urgencyLabel = computed(() => urgencyLabels[handoffSummary.value?.urgency] || '待核验')
const planSteps = computed(() => {
  const returned = plan.value.steps || []
  return Object.entries(planLabels).map(([id, [label, description, icon]]) => {
    const match = returned.find(item => item.id === id)
    const status = plan.value.status === 'handoff_required' && id === 'execute' ? 'blocked' : match?.status || 'pending'
    return { id, label, description, icon, status }
  })
})
const factRows = computed(() => {
  const rows = []
  const order = facts.value.verified_order || {}
  const device = facts.value.verified_device || {}
  const ticketId = firstEntity('ticket_id')
  if (order.order_id) rows.push({ label: '订单', value: `${order.order_id}（${order.status || '已核验'}）` })
  if (device.serial_number) rows.push({ label: '设备', value: `${device.product_model || firstEntity('product_model')} · ${device.serial_number}` })
  if (facts.value.warranty_status && facts.value.warranty_status !== 'unknown') rows.push({ label: '保修', value: facts.value.warranty_status === 'in_warranty' ? `在保至 ${facts.value.warranty_until}` : `已过保（${facts.value.warranty_until || '日期待核验'}）` })
  if (facts.value.diagnosis_status && facts.value.diagnosis_status !== 'not_started') rows.push({ label: '诊断', value: facts.value.diagnosis_status === 'completed' ? '已完成低风险排查' : facts.value.diagnosis_status })
  if (facts.value.recommended_resolution) rows.push({ label: '方案', value: resolutionLabels[facts.value.recommended_resolution] || '处理方案已生成' })
  if (ticketId) rows.push({ label: '工单', value: `${ticketId} · ${ticketStatusLabels[facts.value.ticket_status] || '状态已更新'}` })
  return rows
})
function labelAgent(value) { return agentNames[value] || value || '待路由' }
function firstEntity(name) { const value = entities.value[name]; return Array.isArray(value) ? value[0] : value }
function formatPercent(value) {
  const number = Number(value || 0)
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`
}
function formatJson(value) { return JSON.stringify(value ?? {}, null, 2) }
</script>
