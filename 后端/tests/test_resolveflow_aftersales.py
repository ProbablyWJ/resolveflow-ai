import pytest

from agents.agent_orchestrator import (
    AgentResponse,
    AgentOrchestrator,
    AgentType,
    DiagnosticAgent,
    Request,
    ResponseComposer,
)
from agents.tools import (
    billing_tools,
    check_warranty,
    create_service_ticket,
    get_service_ticket,
    lookup_device,
    lookup_fault_code,
    lookup_order,
    prepare_service_ticket,
    recommend_resolution,
    run_device_diagnostic,
    technical_tools,
)
from core.action_confirmation import is_explicit_ticket_confirmation, is_explicit_ticket_rejection
from core.conversation_state import merge_state
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel


def make_request(message="耳机左耳没声音，我想换货", intent=IntentCategory.EXCHANGE_REQUEST):
    entities = {
        "order_id": ["RF202610270001"],
        "product_model": ["FlowBuds Pro"],
        "serial_number": ["FBP-CN-A10001"],
        "fault_code": ["E101"],
        "fault_symptom": ["左耳无声"],
        "ticket_id": [],
    }
    state = merge_state({}, entities, "aftersales", message, "task-test", intent=intent.value)
    return Request(
        message=message,
        user_id="demo_user",
        conv_id="demo-conversation",
        request_id="request-test",
        entities=state["entities"],
        intent=intent,
        intent_group="aftersales",
        urgency=UrgencyLevel.LOW,
        intent_confidence=0.93,
        conversation_state=state,
    )


def test_pattern_intent_entities_and_safety_are_aftersales_specific():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    pattern = recognizer._pattern_recognize("FlowBuds Pro 左耳没声音，我想换货")
    entities = recognizer._extract_entities("订单 RF202610270001，SN FBP-CN-A10001，故障码 E101，左耳无声")
    urgency = recognizer._urgency("设备电池鼓包并有烧焦味", IntentCategory.QUALITY_COMPLAINT)

    assert pattern["intent"] is IntentCategory.EXCHANGE_REQUEST
    assert entities["order_id"] == ["RF202610270001"]
    assert entities["serial_number"] == ["FBP-CN-A10001"]
    assert entities["fault_code"] == ["E101"]
    assert urgency is UrgencyLevel.CRITICAL


def test_ticket_id_with_update_language_has_deterministic_intent_fallback():
    scores = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        "RF-T20261020-001 为什么还没更新",
        IntentCategory.DEVICE_FAULT,
        0.16,
        scores,
    )

    assert intent is IntentCategory.TICKET_STATUS
    assert confidence == 0.99
    assert scores["business_rule"] == 1.0


def test_repair_ticket_status_language_is_not_overridden_by_repair_action_rule():
    scores = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        "帮我看下维修工单的当前状态",
        IntentCategory.TICKET_STATUS,
        0.74,
        scores,
    )

    assert intent is IntentCategory.TICKET_STATUS
    assert confidence == 0.99
    assert scores["business_rule"] == 1.0


def test_explicit_send_for_repair_has_deterministic_intent_fallback():
    scores = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        "摄像头开不了机，我要送修",
        IntentCategory.REFUND,
        0.18,
        scores,
    )

    assert intent is IntentCategory.REPAIR_REQUEST
    assert confidence == 0.95
    assert scores["business_rule"] == 1.0


def test_fault_plus_exchange_routes_diagnosis_before_resolution():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.INTAKE: [object()],
        AgentType.DIAGNOSTIC: [object()],
        AgentType.RESOLUTION: [object()],
        AgentType.ESCALATION: [object()],
    }
    decision = orchestrator._route_decision(make_request())
    assert decision.primary_agent is AgentType.DIAGNOSTIC
    assert decision.supporting_agents == [AgentType.RESOLUTION]
    assert "diagnose_before_resolution" in decision.reason


def test_high_risk_device_routes_directly_to_escalation():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {AgentType.ESCALATION: [object()]}
    request = make_request("耳机电池鼓包并冒烟", IntentCategory.QUALITY_COMPLAINT)
    request.urgency = UrgencyLevel.CRITICAL
    decision = orchestrator._route_decision(request)
    assert decision.primary_agent is AgentType.ESCALATION
    assert decision.supporting_agents == []


def test_business_tools_create_verified_facts_and_evidence():
    request = make_request()
    assert lookup_order(request, {})["success"] is True
    assert lookup_device(request, {})["success"] is True
    warranty = check_warranty(request, {})
    fault = lookup_fault_code(request, {})
    diagnosis = run_device_diagnostic(request, {})

    assert warranty["warranty_status"] == "in_warranty"
    assert fault["meaning"] == "单侧音频通道未正常初始化"
    assert request.conversation_state["facts"]["verified_fault"] == {
        "fault_code": "E101",
        "product_model": "FlowBuds Pro",
        "meaning": "单侧音频通道未正常初始化",
        "stop_conditions": ["设备异常发热", "电池鼓包", "出现烧焦味"],
    }
    assert diagnosis["diagnosis"]["result"] == "low_risk_steps_provided"
    assert request.conversation_state["facts"]["order_verified"] is True
    assert request.conversation_state["facts"]["device_verified"] is True
    assert request.conversation_state["evidence_refs"]


def test_ticket_status_response_uses_verified_business_data_only():
    request = make_request(
        "查询工单 RF-T20261020-001 的进度。",
        IntentCategory.TICKET_STATUS,
    )
    request.entities = {
        **request.entities,
        "order_id": [],
        "serial_number": [],
        "fault_code": [],
        "fault_symptom": [],
        "ticket_id": ["RF-T20261020-001"],
    }
    request.conversation_state["entities"] = request.entities

    result = get_service_ticket(request, {})
    answer = ResponseComposer._compose_ticket_status(request.conversation_state)

    assert result["success"] is True
    assert request.conversation_state["facts"]["verified_ticket"]["status"] == "awaiting_shipment"
    assert request.conversation_state["resolution_plan"]["status"] == "completed"
    assert "RF-T20261020-001" in answer
    assert "等待寄出设备" in answer
    assert "awaiting_shipment" not in answer
    assert "等待用户寄出设备" in answer
    assert "电池鼓包" not in answer
    assert len(answer) <= 500


def test_warranty_response_is_concise_and_does_not_expose_internal_references():
    request = make_request(
        "订单 RF202510010002、SN FCM-CN-B20002 的设备是否过保？",
        IntentCategory.WARRANTY_INQUIRY,
    )
    request.entities = {
        **request.entities,
        "order_id": ["RF202510010002"],
        "serial_number": ["FCM-CN-B20002"],
    }
    request.conversation_state["entities"] = request.entities
    assert lookup_order(request, {})["success"] is True
    assert check_warranty(request, {})["success"] is True

    answer = ResponseComposer._compose_warranty_status(request.conversation_state)

    assert "2026-10-01" in answer
    assert "已过保" in answer
    assert "付费维修或转人工核验" in answer
    assert request.conversation_state["resolution_plan"]["status"] == "completed"
    assert "policy://" not in answer
    assert len(answer) <= 500


def test_parts_response_hides_internal_action_id_and_keeps_user_next_step():
    request = make_request(
        "SN FBP-CN-A10001 的 FlowBuds Pro 右耳耳帽丢了，我想申请原装配件。",
        IntentCategory.PARTS_REQUEST,
    )
    request.entities["fault_symptom"] = []
    request.conversation_state["entities"] = request.entities
    assert lookup_device(request, {})["success"] is True
    assert check_warranty(request, {})["success"] is True
    assert recommend_resolution(request, {"desired_action": "parts"})["success"] is True
    assert prepare_service_ticket(request, {"ticket_type": "parts"})["success"] is True

    answer = ResponseComposer._compose_parts_request(request.conversation_state, request.message)

    assert request.conversation_state["resolution_plan"]["status"] == "awaiting_action"
    assert request.conversation_state["resolution_plan"]["steps"][3]["status"] == "not_required"
    assert "耳帽" in answer
    assert "S / M / L" in answer
    assert "尚未创建" in answer
    assert "创建前再次向你确认" in answer
    assert "原装耳帽" in answer
    assert "action_id" not in answer
    assert "policy://" not in answer
    assert len(answer) <= 500


def test_dangerous_symptom_stops_diagnosis():
    request = make_request("耳机电池鼓包", IntentCategory.DEVICE_FAULT)
    request.entities["fault_symptom"] = ["电池鼓包"]
    result = run_device_diagnostic(request, {"symptom": "电池鼓包"})
    assert result["success"] is False
    assert result["safe_to_continue"] is False
    assert result["requires_escalation"] is True


def test_diagnostic_safety_uses_user_entity_not_hallucinated_tool_argument():
    request = make_request("左耳无声", IntentCategory.DEVICE_FAULT)
    request.entities["fault_symptom"] = ["左耳无声"]
    result = run_device_diagnostic(request, {"symptom": "电池鼓包并冒烟"})
    assert result["success"] is True
    assert result["safe_to_continue"] is True
    assert result["diagnosis"]["symptom"] == "左耳无声"


def test_diagnostic_safety_uses_original_message_for_negated_risk():
    request = make_request(
        "FlowBuds Pro 没有鼓包、没有冒烟，左耳没声音",
        IntentCategory.DEVICE_FAULT,
    )
    # 即使上游结构化实体被错误污染，工具也不能把否定表达升级为真实风险。
    request.entities["fault_symptom"] = ["冒烟", "没声音"]

    result = run_device_diagnostic(request, {"symptom": "冒烟"})

    assert result["success"] is True
    assert result["safe_to_continue"] is True
    assert result["requires_escalation"] is False
    assert result["diagnosis"]["symptom"] == "没声音"


def test_ticket_requires_explicit_confirmation_and_is_idempotent():
    request = make_request()
    assert lookup_order(request, {})["success"] is True
    assert lookup_device(request, {})["success"] is True
    assert check_warranty(request, {})["success"] is True
    assert run_device_diagnostic(request, {})["success"] is True
    assert recommend_resolution(request, {"desired_action": "exchange"})["success"] is True
    prepared = prepare_service_ticket(request, {"ticket_type": "exchange"})
    action_id = prepared["pending_action"]["action_id"]
    pending_answer = ResponseComposer._compose_aftersales_state(request.conversation_state)
    assert "已准备换货演示工单" in pending_answer
    assert "exchange" not in pending_answer
    progress_answer = ResponseComposer._compose_aftersales_state(
        request.conversation_state,
        "三项低风险步骤已经完成，左耳仍然无声。请继续换货方案并准备工单。",
    )
    assert "低风险排查已完成，左耳无声仍然存在" in progress_answer
    assert "已根据你的排查结果准备换货演示工单" in progress_answer
    assert "将双耳放回充电盒" not in progress_answer

    request.message = "我想换货"
    rejected = create_service_ticket(request, {"action_id": action_id})
    assert rejected["created"] is False

    request.message = "不同意创建工单"
    rejected_negative = create_service_ticket(request, {"action_id": action_id})
    assert rejected_negative["created"] is False
    assert request.conversation_state["pending_action"]["action_id"] == action_id
    assert request.conversation_state["facts"].get("ticket_created") is not True

    request.message = "确认创建工单"
    created = create_service_ticket(request, {"action_id": action_id})
    replayed = create_service_ticket(request, {"action_id": action_id})
    assert created["created"] is True
    assert request.conversation_state["facts"]["ticket_status"] == "created"
    assert replayed["idempotent_replay"] is True
    assert replayed["ticket_id"] == created["ticket_id"]
    completed_answer = ResponseComposer._compose_aftersales_state(request.conversation_state)
    assert "已创建" in completed_answer
    assert "等待售后审核" in completed_answer
    assert "created" not in completed_answer


@pytest.mark.parametrize("message", [
    "确认创建工单",
    "我确认创建售后工单。",
    "同意提交工单",
    "请立即办理售后工单",
    "我确认提交这张工单。",
])
def test_explicit_ticket_confirmation_accepts_clear_positive_commands(message):
    assert is_explicit_ticket_confirmation(message) is True


@pytest.mark.parametrize("message", [
    "不同意创建工单",
    "不确认创建工单",
    "请不要创建工单",
    "我不想提交售后工单",
    "先别办理工单",
    "取消创建工单",
    "确认不创建工单",
    "可以创建工单吗？",
    "怎么创建工单？",
    "好的，就这样办",
])
def test_explicit_ticket_confirmation_rejects_negative_ambiguous_or_question(message):
    assert is_explicit_ticket_confirmation(message) is False


@pytest.mark.parametrize("message", [
    "不同意创建工单",
    "请不要创建工单",
    "取消创建工单",
    "算了，不要提交，保留方案就好。",
])
def test_explicit_ticket_rejection_only_accepts_clear_negative_commands(message):
    assert is_explicit_ticket_rejection(message) is True


@pytest.mark.parametrize("message", ["确认创建工单", "可以创建工单吗？", "我不知道怎么创建工单"])
def test_ticket_rejection_rejects_positive_questions_and_unrelated_negation(message):
    assert is_explicit_ticket_rejection(message) is False


@pytest.mark.parametrize(("ticket_type", "expected_intent"), [
    ("exchange", IntentCategory.EXCHANGE_REQUEST),
    ("repair", IntentCategory.REPAIR_REQUEST),
    ("parts", IntentCategory.PARTS_REQUEST),
])
def test_confirm_ticket_inherits_business_intent_from_pending_action(ticket_type, expected_intent):
    request = make_request("确认创建工单。", IntentCategory.TICKET_STATUS)
    request.conversation_state["pending_action"] = {
        "type": "create_service_ticket",
        "ticket_type": ticket_type,
        "action_id": f"pending-{ticket_type}",
    }
    request.intent_confidence = 0.61
    request.intent_source_scores = {"pattern": 0.5}

    resolution = AgentOrchestrator._inherit_pending_action_intent(request)

    assert request.intent is expected_intent
    assert request.intent_group == "aftersales"
    assert request.intent_confidence == 0.99
    assert request.intent_source_scores["contextual_state"] == 1.0
    assert resolution == {
        "source": "pending_action",
        "raw_intent": "ticket_status",
        "resolved_intent": expected_intent.value,
        "action_type": "create_service_ticket",
        "ticket_type": ticket_type,
    }


def test_user_reported_troubleshooting_progress_persists_across_confirmation_turn():
    request = make_request()
    lookup_order(request, {})
    lookup_device(request, {})
    check_warranty(request, {})
    run_device_diagnostic(request, {})
    recommend_resolution(request, {"desired_action": "exchange"})
    prepare_service_ticket(request, {"ticket_type": "exchange"})

    progressed = merge_state(
        request.conversation_state, {}, "aftersales",
        "三项低风险步骤已经完成，左耳仍然无声。",
        "turn-progress", intent="exchange_request",
    )
    confirmed = merge_state(
        progressed, {}, "aftersales", "确认创建工单。",
        "turn-confirm", intent="exchange_request",
    )
    answer = ResponseComposer._compose_aftersales_state(confirmed, "确认创建工单。")

    assert confirmed["task_progress"] == {
        "troubleshooting_completed": True,
        "issue_persists": True,
        "source": "user_reported",
        "source_request_id": "turn-progress",
    }
    assert "低风险排查已完成，左耳无声仍然存在" in answer
    assert "将双耳放回充电盒" not in answer


def test_colloquial_troubleshooting_progress_is_saved():
    progressed = merge_state(
        make_request().conversation_state,
        {},
        "aftersales",
        "你说的排查我都完成了，故障还是存在，请先把工单准备好。",
        "turn-colloquial-progress",
        intent="exchange_request",
    )

    assert progressed["task_progress"]["troubleshooting_completed"] is True
    assert progressed["task_progress"]["issue_persists"] is True


def test_short_command_does_not_inherit_without_matching_pending_action():
    request = make_request("确认创建工单。", IntentCategory.TICKET_STATUS)
    original = request.intent

    assert AgentOrchestrator._inherit_pending_action_intent(request) is None
    assert request.intent is original

    request.conversation_state["pending_action"] = {
        "type": "create_service_ticket",
        "ticket_type": "exchange",
    }
    request.message = "帮我看看工单"
    assert AgentOrchestrator._inherit_pending_action_intent(request) is None
    assert request.intent is original

    request.message = "不同意创建工单"
    assert AgentOrchestrator._inherit_pending_action_intent(request) is None
    assert request.intent is original


def test_reply_explains_user_next_step_without_internal_plan_name():
    request = make_request()
    assert lookup_order(request, {})["success"] is True
    assert lookup_device(request, {})["success"] is True
    assert check_warranty(request, {})["success"] is True
    assert lookup_fault_code(request, {})["success"] is True
    assert run_device_diagnostic(request, {})["success"] is True

    answer = ResponseComposer._compose_aftersales_state(request.conversation_state)

    assert "换货条件核验" in answer
    assert "订单与设备信息已核验" in answer
    assert "设备当前在保" in answer
    assert "最终是否符合换货条件以售后审核结果为准" in answer
    assert "排查已完成，问题仍存在" in answer
    assert "继续核验换货条件" in answer
    assert "当前尚未创建工单" in answer
    assert "继续步骤：" not in answer
    assert "decide" not in answer
    assert "action_id" not in answer


def test_device_fault_reply_explains_code_stop_conditions_and_follow_up():
    request = make_request("SN FBP-CN-A10001 左耳无声，故障码 E101", IntentCategory.DEVICE_FAULT)
    assert lookup_device(request, {})["success"] is True
    assert lookup_fault_code(request, {})["success"] is True
    assert run_device_diagnostic(request, {})["success"] is True

    answer = ResponseComposer._compose_aftersales_state(request.conversation_state)

    assert "E101 已在设备资料中核验：单侧音频通道未正常初始化" in answer
    assert "停止条件：如出现设备异常发热、电池鼓包、出现烧焦味" in answer
    assert "完成后告诉我哪一步有效，或直接回复“都没解决”" in answer
    assert "下一步需根据已核验信息确定处理方案" not in answer
    assert not any(
        term in answer
        for term in ("##", "当前判断", "实际依据", "验证方式", "明确下一步", "tool_name")
    )
    assert len(answer) <= 500


def test_simple_device_diagnosis_is_a_concise_customer_reply_not_an_internal_report():
    request = make_request("我的耳机只有左耳没声音", IntentCategory.DEVICE_FAULT)
    request.entities["fault_code"] = []
    request.conversation_state["entities"]["fault_code"] = []
    assert lookup_device(request, {})["success"] is True
    assert run_device_diagnostic(request, {})["success"] is True

    answer = ResponseComposer._compose_aftersales_state(request.conversation_state)

    assert "可以先做低风险排查" in answer
    assert "将双耳放回充电盒 10 秒后重新取出" in answer
    assert "记录复现步骤和最新现象" not in answer
    assert "完成后告诉我哪一步有效" in answer
    assert not any(
        term in answer
        for term in ("##", "当前判断", "实际依据", "验证方式", "信息依据", "evidence_refs")
    )


def test_diagnostic_steps_generalize_to_computer_audio_without_earbud_actions():
    request = make_request("电脑没声音", IntentCategory.DEVICE_FAULT)
    request.entities.update({
        "product_model": ["ThinkPad X1"],
        "serial_number": [],
        "fault_code": [],
        "fault_symptom": ["电脑没声音"],
    })
    request.conversation_state["entities"] = request.entities
    request.conversation_state["support_flow"] = {
        "type": "device_fault_clarification",
        "status": "ready",
        "stage": "diagnosis_ready",
        "product_category": "电脑",
        "product_model": "ThinkPad X1",
        "symptom_category": "audio",
    }

    result = run_device_diagnostic(request, {})

    assert result["success"] is True
    steps = result["diagnosis"]["safe_steps"]
    assert "将音频输出切换到预期设备" in steps[0]
    assert all("充电盒" not in step for step in steps)


def test_execute_step_asks_for_the_known_business_ticket_type_only():
    request = make_request()
    lookup_order(request, {})
    lookup_device(request, {})
    check_warranty(request, {})
    run_device_diagnostic(request, {})
    recommend_resolution(request, {"desired_action": "exchange"})

    answer = ResponseComposer._compose_aftersales_state(request.conversation_state)

    assert "请准备换货工单" in answer
    assert "维修工单 / 换货工单 / 配件工单" not in answer


def test_tool_permissions_and_side_effect_metadata_are_explicit():
    diagnostic = technical_tools()
    resolution = billing_tools()
    assert set(diagnostic) == {"lookup_device", "lookup_fault_code", "run_device_diagnostic"}
    assert "create_service_ticket" not in diagnostic
    assert resolution["create_service_ticket"].retry_safe is False


def test_second_turn_keeps_known_fields_and_device_switch_clears_old_task():
    first = merge_state(
        {},
        {"fault_symptom": ["左耳无声"], "serial_number": [], "order_id": []},
        "aftersales",
        "耳机左耳没声音",
        "turn-1",
        intent="device_fault",
    )
    second = merge_state(
        first,
        {"order_id": ["RF202610270001"], "serial_number": ["FBP-CN-A10001"]},
        "aftersales",
        "订单号 RF202610270001，SN FBP-CN-A10001",
        "turn-2",
        intent="exchange_request",
    )
    switched = merge_state(
        second,
        {"serial_number": ["FCM-CN-B20002"], "fault_symptom": ["连接失败"]},
        "aftersales",
        "换个设备，SN FCM-CN-B20002 连接失败",
        "turn-3",
        intent="device_fault",
    )

    assert second["entities"]["fault_symptom"] == ["左耳无声"]
    assert second["entities"]["order_id"] == ["RF202610270001"]
    assert switched["task_id"] == "turn-3"
    assert switched["entities"]["serial_number"] == ["FCM-CN-B20002"]
    assert switched["entities"]["order_id"] == []


def test_ticket_tracking_state_does_not_leak_into_new_device_incident():
    tracked = merge_state(
        {},
        {"ticket_id": ["RF-T20261020-001"], "product_model": ["FlowBuds Pro"]},
        "aftersales",
        "查询工单 RF-T20261020-001",
        "turn-ticket",
        intent="ticket_status",
    )
    tracked["facts"].update({"ticket_status": "awaiting_shipment", "recommended_resolution": "寄修"})
    tracked["evidence_refs"] = ["service_ticket:RF-T20261020-001"]
    tracked["resolution_plan"]["status"] = "completed"

    incident = merge_state(
        tracked,
        {"product_model": ["FlowBuds Pro"], "fault_symptom": ["电池鼓包"]},
        "aftersales",
        "FlowBuds Pro 电池鼓包并有烧焦味",
        "turn-incident",
        intent="device_fault",
    )

    assert incident["task_id"] == "turn-incident"
    assert incident["entities"]["ticket_id"] == []
    assert "ticket_status" not in incident["facts"]
    assert "recommended_resolution" not in incident["facts"]
    assert incident["evidence_refs"] == []
    assert incident["task_transition"] == {
        "type": "new_task",
        "reason": "left_ticket_tracking",
        "previous_task_id": "turn-ticket",
        "request_id": "turn-incident",
    }


def test_new_fault_after_terminal_task_resets_task_but_handoff_keeps_context():
    resolved = merge_state(
        {},
        {"serial_number": ["FBP-CN-A10001"], "fault_symptom": ["左耳无声"]},
        "aftersales",
        "左耳无声",
        "turn-old",
        intent="device_fault",
    )
    resolved["resolution_plan"]["status"] = "completed"
    resolved["facts"]["diagnosis_status"] = "completed"

    fresh = merge_state(
        resolved,
        {"fault_symptom": ["电池鼓包"]},
        "aftersales",
        "现在电池鼓包了",
        "turn-new",
        intent="device_fault",
    )
    handed_off = merge_state(
        resolved,
        {},
        "aftersales",
        "请帮我转人工",
        "turn-human",
        intent="human_handoff",
    )

    assert fresh["task_id"] == "turn-new"
    assert fresh["entities"]["serial_number"] == []
    assert fresh["facts"]["diagnosis_status"] == "not_started"
    assert fresh["task_transition"]["reason"] == "new_fault_after_terminal_task"
    assert handed_off["task_id"] == "turn-old"
    assert handed_off["entities"]["serial_number"] == ["FBP-CN-A10001"]


def test_tool_failure_replans_only_failed_step_and_stops_at_budget():
    request = make_request()
    request.entities["order_id"] = ["RF-NOT-FOUND"]
    request.conversation_state["entities"] = request.entities
    missing = lookup_order(request, {})
    assert missing["success"] is False
    assert request.conversation_state["resolution_plan"]["next_step"] == "collect_order_id"
    assert request.conversation_state["tool_budget"]["replans_used"] == 1

    request.conversation_state["tool_budget"]["calls_used"] = 16
    exhausted = lookup_order(request, {})
    assert exhausted["success"] is False
    assert request.conversation_state["resolution_plan"]["status"] == "handoff_required"
    assert request.conversation_state["resolution_plan"]["next_step"] == "escalate"


def test_recovered_plan_clears_stale_blockers_and_last_failed_step():
    request = make_request()
    lookup_order(request, {})
    lookup_device(request, {})
    check_warranty(request, {})

    failed = recommend_resolution(request, {"desired_action": "exchange"})
    assert failed["success"] is False
    assert request.conversation_state["resolution_plan"]["last_failed_step"] == "decide"
    assert request.conversation_state["resolution_plan"]["blockers"]

    run_device_diagnostic(request, {})
    recovered = recommend_resolution(request, {"desired_action": "exchange"})

    assert recovered["success"] is True
    assert "blockers" not in request.conversation_state["resolution_plan"]
    assert "last_failed_step" not in request.conversation_state["resolution_plan"]


def test_completed_fallback_step_clears_obsolete_blocker_reason_before_retry():
    request = make_request()
    lookup_order(request, {})
    lookup_device(request, {})
    check_warranty(request, {})
    failed = recommend_resolution(request, {"desired_action": "exchange"})
    assert failed["success"] is False

    run_device_diagnostic(request, {})

    assert request.conversation_state["resolution_plan"]["next_step"] == "decide"
    assert "blockers" not in request.conversation_state["resolution_plan"]
    assert "last_failed_step" not in request.conversation_state["resolution_plan"]


def test_conditional_safety_text_does_not_mark_normal_case_escalated():
    agent = DiagnosticAgent.__new__(DiagnosticAgent)
    request = make_request()
    assert agent._needs_escalation("若出现鼓包，请直接转人工升级。", request) is False
    request.conversation_state["resolution_plan"]["status"] = "handoff_required"
    assert agent._needs_escalation("已记录信息。", request) is True


def test_multi_agent_reply_is_composed_from_structured_facts():
    import asyncio

    request = make_request()
    lookup_order(request, {})
    lookup_device(request, {})
    check_warranty(request, {})
    run_device_diagnostic(request, {})
    recommend_resolution(request, {"desired_action": "exchange"})
    composer = ResponseComposer(None, "unused")
    result = asyncio.run(composer.compose_result(request, [
        AgentResponse(AgentType.DIAGNOSTIC, "旧诊断长文本", True, status="partial"),
        AgentResponse(AgentType.RESOLUTION, "旧方案长文本", True, status="partial"),
    ]))

    assert result.status == "success"
    assert "订单 RF202610270001 已核验" in result.content
    assert "在保换货人工审核" in result.content
    assert "旧诊断长文本" not in result.content


def test_single_successful_aftersales_agent_still_uses_structured_composer():
    import asyncio

    request = make_request()
    lookup_order(request, {})
    lookup_device(request, {})
    check_warranty(request, {})
    run_device_diagnostic(request, {})
    composer = ResponseComposer(None, "unused")
    result = asyncio.run(composer.compose_result(request, [
        AgentResponse(AgentType.DIAGNOSTIC, "内部枚举=manual_exchange_review；原始提示词", True),
    ]))

    assert "内部枚举" not in result.content
    assert "manual_exchange_review" not in result.content
    assert "订单 RF202610270001 已核验" in result.content
