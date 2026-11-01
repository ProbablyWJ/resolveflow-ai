"""会话中的业务字段与售后任务计划；字段存在不等于业务事实已经核验。"""
import re


AFTERSALES_FIELDS = (
    "order_id", "product_model", "serial_number", "purchase_date",
    "fault_code", "fault_symptom", "evidence", "ticket_id",
)

AFTERSALES_SERVICE_INTENTS = {
    "aftersales", "device_fault", "warranty_inquiry", "repair_request",
    "exchange_request", "quality_complaint",
}
TERMINAL_PLAN_STATUSES = {"completed", "handoff_required"}


_TROUBLESHOOTING_DONE = re.compile(
    r"(?:排查|步骤|操作|建议).{0,24}(?:完成|做完|试过|都做了|都完成|已做)"
)
_ISSUE_PERSISTS = re.compile(
    r"(?:仍|还是|依然|仍然|还).{0,16}(?:存在|无声|没声音|故障|异常|不能|问题|无法|不行)"
    r"|(?:故障|问题).{0,8}(?:仍|还是|依然|仍然|还).{0,8}(?:存在|没解决|不行)"
)


def _update_task_progress(progress, message, request_id):
    """保存用户明确声明的任务进度，与工具核验的 facts 分开。"""
    updated = dict(progress or {})
    text = message or ""
    if _TROUBLESHOOTING_DONE.search(text) and _ISSUE_PERSISTS.search(text):
        updated.update({
            "troubleshooting_completed": True,
            "issue_persists": True,
            "source": "user_reported",
            "source_request_id": request_id,
        })
    return updated


def _blocker_resolved(blocker, state, completed_steps):
    failed = str((blocker or {}).get("step") or "")
    fallback = str((blocker or {}).get("fallback_step") or "")
    facts = state.get("facts", {}) or {}
    fallback_stage = {
        "collect_order_id": "collect",
        "collect_device": "collect",
        "collect_fault_symptom": "collect",
        "verify": "verify",
        "retry_retrieve": "retrieve",
        "diagnose": "diagnose",
        "decide": "decide",
    }.get(fallback, fallback)
    if fallback_stage in completed_steps:
        return True
    if failed in {"collect", "collect_order_id", "collect_device", "collect_fault_symptom"}:
        return "collect" in completed_steps
    if failed in {"verify", "verify_order", "verify_device", "lookup_order", "lookup_device"}:
        return "verify" in completed_steps
    if failed in {"retrieve", "retrieve_fault_code", "search_knowledge_base", "lookup_fault_code"}:
        return "retrieve" in completed_steps
    if failed in {"diagnose", "run_device_diagnostic"}:
        return "diagnose" in completed_steps
    if failed in {"decide", "recommend_resolution"}:
        return "decide" in completed_steps
    if failed in {"prepare_ticket", "prepare_service_ticket"}:
        return bool(state.get("pending_action")) or "execute" in completed_steps
    if failed in {"execute", "create_service_ticket"}:
        return "execute" in completed_steps
    return bool(failed and failed in completed_steps)


def _task_family(intent):
    if intent in AFTERSALES_SERVICE_INTENTS:
        return "device_service"
    if intent == "ticket_status":
        return "ticket_tracking"
    if intent == "parts_request":
        return "parts_service"
    if intent == "human_handoff":
        return "handoff"
    return intent or "unknown"


def _meaningful_values(values):
    return {str(value).strip().lower() for value in (values or []) if str(value).strip()}


def _task_reset_reason(previous, entities, intent, message):
    """只在有明确证据时切换任务，避免一刀切地丢失多轮上下文。"""
    old = previous.get("entities") if isinstance(previous.get("entities"), dict) else {}
    changed_order = bool(entities.get("order_id") and old.get("order_id") and entities["order_id"] != old["order_id"])
    changed_device = bool(
        (entities.get("serial_number") and old.get("serial_number") and entities["serial_number"] != old["serial_number"])
        or (entities.get("product_model") and old.get("product_model") and entities["product_model"] != old["product_model"])
    )
    if changed_order:
        return "changed_order"
    if changed_device:
        return "changed_device"
    if re.search(r"(?:换个|新的|另一个)(?:问题|订单|设备)|另一笔订单|另一台设备", message or ""):
        return "explicit_new_task"

    # 通用会话没有售后 task_id，但上面的订单/设备锚点仍然要生效。
    # 下面的意图容器与终态判断只适用于结构化售后任务。
    if not previous.get("task_id"):
        return None

    previous_intent = previous.get("intent")
    previous_family = _task_family(previous_intent)
    current_family = _task_family(intent)

    # 用户要求人工是当前任务的升级，不应丢掉已核验事实。
    if current_family == "handoff":
        return None
    # 从“查已有工单”转到新的设备/配件问题，不应携带旧工单结论。
    if previous_family == "ticket_tracking" and current_family != "ticket_tracking":
        return "left_ticket_tracking"
    if previous_family == "handoff" and current_family not in {"handoff", "unknown"}:
        return "left_handoff_task"
    if previous_family != current_family and {previous_family, current_family} & {"parts_service"}:
        return "changed_service_type"

    plan = previous.get("resolution_plan") if isinstance(previous.get("resolution_plan"), dict) else {}
    old_faults = _meaningful_values(old.get("fault_symptom"))
    new_faults = _meaningful_values(entities.get("fault_symptom"))
    has_different_fault = bool(new_faults and old_faults and not new_faults.intersection(old_faults))
    if plan.get("status") in TERMINAL_PLAN_STATUSES and has_different_fault:
        return "new_fault_after_terminal_task"
    return None


def _after_sales_goal(intent):
    return {
        "device_fault": "diagnose_and_resolve",
        "warranty_inquiry": "verify_warranty",
        "repair_request": "prepare_repair",
        "exchange_request": "prepare_exchange",
        "parts_request": "prepare_parts_service",
        "ticket_status": "track_or_create_ticket",
        "quality_complaint": "resolve_quality_complaint",
        "human_handoff": "handoff_to_human",
    }.get(intent, "clarify_aftersales_need")


def refresh_resolution_plan(state, intent=None):
    """根据已知字段和已核验事实刷新轻量任务计划，不让模型伪造完成状态。"""
    if not isinstance(state, dict) or state.get("domain") != "aftersales":
        return state
    entities = state.setdefault("entities", {})
    facts = state.setdefault("facts", {
        "order_verified": False,
        "device_verified": False,
        "warranty_status": "unknown",
        "diagnosis_status": "not_started",
    })
    plan = state.setdefault("resolution_plan", {})
    plan["goal"] = _after_sales_goal(intent or state.get("intent"))

    intent_value = intent or state.get("intent") or "aftersales"
    ticket_verified = bool(facts.get("verified_ticket"))
    diagnosis_not_required = intent_value in {"ticket_status", "parts_request", "warranty_inquiry"}
    decision_not_required = intent_value in {"ticket_status", "warranty_inquiry"}
    required = {
        "device_fault": ("fault_symptom",),
        "warranty_inquiry": ("order_id", "serial_number"),
        "repair_request": ("order_id", "serial_number", "fault_symptom"),
        "exchange_request": ("order_id", "serial_number", "fault_symptom"),
        "parts_request": ("serial_number",),
        "ticket_status": ("ticket_id",),
        "quality_complaint": ("order_id", "fault_symptom"),
    }.get(intent_value, ("fault_symptom",))
    if intent_value == "ticket_status" and state.get("pending_action"):
        required = ()
    missing = [name for name in required if not entities.get(name)]

    steps = [
        {"id": "collect", "status": "completed" if not missing else "in_progress"},
        {"id": "verify", "status": "completed" if facts.get("order_verified") or facts.get("device_verified") or ticket_verified else "pending"},
        {"id": "retrieve", "status": "completed" if state.get("evidence_refs") else "pending"},
        {"id": "diagnose", "status": "not_required" if diagnosis_not_required else ("completed" if facts.get("diagnosis_status") == "completed" else "pending")},
        {"id": "decide", "status": "not_required" if decision_not_required else ("completed" if facts.get("recommended_resolution") else "pending")},
        {"id": "execute", "status": "completed" if entities.get("ticket_id") else "pending"},
    ]
    if missing:
        next_step, status = "collect", "collecting_information"
    elif intent_value == "ticket_status" and ticket_verified:
        next_step, status = "complete", "completed"
    elif intent_value == "warranty_inquiry" and facts.get("warranty_status") in {"in_warranty", "out_of_warranty"}:
        next_step, status = "complete", "completed"
    elif not (facts.get("order_verified") or facts.get("device_verified")):
        next_step, status = "verify", "verifying_business_facts"
    elif not state.get("evidence_refs"):
        next_step, status = "retrieve", "retrieving_evidence"
    elif facts.get("diagnosis_status") != "completed" and not diagnosis_not_required:
        next_step, status = "diagnose", "diagnosing"
    elif not facts.get("recommended_resolution") and not decision_not_required:
        next_step, status = "decide", "deciding_resolution"
    elif entities.get("ticket_id"):
        next_step, status = "complete", "completed"
    else:
        next_step, status = "execute", "awaiting_action"
    plan.update({"status": status, "steps": steps, "next_step": next_step, "missing_fields": missing})

    # 失败信息只描述当前障碍；对应步骤恢复后不应继续污染已完成任务。
    completed_steps = {step["id"] for step in steps if step.get("status") == "completed"}
    if status == "completed":
        plan.pop("blockers", None)
        plan.pop("last_failed_step", None)
        plan.pop("stop_reason", None)
    elif plan.get("status") != "handoff_required":
        blockers = [
            blocker for blocker in (plan.get("blockers") or [])
            if not _blocker_resolved(blocker, state, completed_steps)
        ]
        if blockers:
            plan["blockers"] = blockers
        else:
            plan.pop("blockers", None)
        if plan.get("last_failed_step") and not any(
            str(item.get("step")) == str(plan.get("last_failed_step")) for item in blockers
        ):
            plan.pop("last_failed_step", None)
    return state


def replan_after_failure(state, failed_step, reason, fallback_step, *, terminal=False):
    """记录失败并只调整受影响步骤；达到预算或安全终止时转人工。"""
    if not isinstance(state, dict) or state.get("domain") != "aftersales":
        return state
    plan = state.setdefault("resolution_plan", {})
    budget = state.setdefault("tool_budget", {"calls_used": 0, "calls_max": 16, "replans_used": 0, "replans_max": 2})
    blockers = plan.setdefault("blockers", [])
    blocker = {"step": failed_step, "reason": str(reason)[:240], "fallback_step": fallback_step}
    if blocker not in blockers:
        blockers.append(blocker)
    exhausted = int(budget.get("replans_used", 0)) >= int(budget.get("replans_max", 2))
    if terminal or exhausted:
        plan.update({"status": "handoff_required", "next_step": "escalate", "stop_reason": str(reason)[:240]})
        return state
    budget["replans_used"] = int(budget.get("replans_used", 0)) + 1
    plan.update({"status": "replanning", "next_step": fallback_step, "last_failed_step": failed_step})
    return state


def merge_state(previous, entities, intent_group, message, request_id, intent=None):
    previous = previous if isinstance(previous, dict) else {}
    previous_task_id = previous.get("task_id")
    domain = {"billing": "commerce", "query": "commerce", "aftersales": "aftersales"}.get(intent_group, intent_group)
    neutral = domain in {"greeting", "feedback", "other", "complaint", "request", "escalation"}
    if intent == "human_handoff":
        domain = previous.get("domain") or "aftersales"
        neutral = False
    old = previous.get("entities", {})
    sources = previous.get("sources", {})
    old = old if isinstance(old, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    reset_reason = _task_reset_reason(previous, entities, intent, message)
    new_task = reset_reason is not None
    if neutral:
        domain = ("commerce" if entities.get("order_id") else "general") if new_task else previous.get("domain", "commerce" if entities.get("order_id") else "general")
    keep = not new_task and (neutral or domain == previous.get("domain"))
    merged = {key: list(value) for key, value in old.items() if isinstance(value, list)} if keep else {}
    origins = dict(sources) if keep else {}
    for key, value in entities.items():
        if value:
            merged[key] = list(value)
            origins[key] = request_id
        else:
            merged.setdefault(key, [])
    if domain != "aftersales" and previous.get("domain") != "aftersales":
        return {"domain": domain, "entities": merged, "sources": origins}

    if not keep:
        previous = {}
    state = {
        "domain": "aftersales",
        "task_id": previous.get("task_id") or request_id,
        "intent": intent or previous.get("intent") or "aftersales",
        "entities": merged,
        "sources": origins,
        "facts": dict(previous.get("facts") or {
            "order_verified": False,
            "device_verified": False,
            "warranty_status": "unknown",
            "diagnosis_status": "not_started",
        }),
        "evidence_refs": list(previous.get("evidence_refs") or []),
        "task_progress": _update_task_progress(
            previous.get("task_progress") if keep else {}, message, request_id,
        ),
        "pending_action": previous.get("pending_action"),
        "support_flow": dict(previous.get("support_flow") or {}),
        "tool_budget": dict(previous.get("tool_budget") or {
            "calls_used": 0, "calls_max": 16, "replans_used": 0, "replans_max": 2,
        }),
        "resolution_plan": dict(previous.get("resolution_plan") or {}),
    }
    if reset_reason:
        state["task_transition"] = {
            "type": "new_task",
            "reason": reset_reason,
            "previous_task_id": previous_task_id,
            "request_id": request_id,
        }
    for field in AFTERSALES_FIELDS:
        state["entities"].setdefault(field, [])
    return refresh_resolution_plan(state, intent=intent)
