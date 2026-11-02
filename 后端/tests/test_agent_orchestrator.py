import asyncio

import pytest

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != BillingAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_fault_code" in TechnicalAgent.profile.tool_scope
    assert "check_warranty" in BillingAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in technical_packet
    assert "aftersales_fields" in billing_packet
    assert general_packet != technical_packet != billing_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")
    req = make_request(
        message="我想找人工处理登录问题",
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.HIGH,
        conversation_state={"facts": {"account_checked": True}},
    )

    result = asyncio.run(agent.handle(req))

    assert result.success is True
    assert result.escalate is True
    assert "停止自动处理" in result.content
    assert "未自动接入真实人工队列" in result.content
    assert "意图=" not in result.content
    assert "HIGH" not in result.content
    assert "{" not in result.content
    assert req.conversation_state["handoff_summary"]["reason_code"] == "user_requested_human"
    assert req.conversation_state["handoff_summary"]["queue_connected"] is False
    assert req.conversation_state["handoff_summary"]["facts"]["account_checked"] is True
    assert client.calls == []


def test_device_safety_handoff_keeps_internal_fields_out_of_customer_reply():
    req = make_request(
        message="FlowBuds Pro 充电盒电池鼓包，还有烧焦味",
        intent=IntentCategory.DEVICE_FAULT,
        urgency=UrgencyLevel.CRITICAL,
        entities={"product_model": ["FlowBuds Pro"], "fault_symptom": ["电池鼓包", "烧焦味"]},
        conversation_state={"resolution_plan": {"steps": [{"id": "collect", "status": "completed"}]}},
    )

    result = asyncio.run(EscalationAgent(FakeClient(), "test-model").handle(req))

    assert "立即停止使用和充电" in result.content
    assert "已停止自动处理" in result.content
    assert "未自动接入真实人工队列" in result.content
    assert "意图=" not in result.content
    assert "CRITICAL" not in result.content
    assert "product_model" not in result.content
    summary = req.conversation_state["handoff_summary"]
    assert summary["reason_code"] == "device_safety_risk"
    assert summary["urgency"] == "CRITICAL"
    assert summary["entities"]["product_model"] == ["FlowBuds Pro"]
    assert summary["completed_steps"] == ["collect"]


def test_information_readiness_gate_clarifies_vague_fault_progressively_without_tools():
    from collections import deque

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=20)

    first_req = Request(
        message="我耳机坏了",
        user_id="u1",
        conv_id="clarify-1",
        intent=IntentCategory.DEVICE_FAULT,
        intent_group="aftersales",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state={},
    )
    first = asyncio.run(orchestrator.run(first_req))
    assert first.agent_type is AgentType.INTAKE
    assert first.tools_used == []
    assert all(term in first.response for term in ("有", "没有", "不确定"))
    assert first.question_group["type"] == "single_select"
    assert [option["value"] for option in first.question_group["options"]] == ["有", "没有", "不确定"]
    assert first.question_group["allow_text"] is True
    assert first_req.conversation_state["support_flow"]["stage"] == "safety_check"

    second_req = Request(
        message="没有这些危险情况",
        user_id="u1",
        conv_id="clarify-1",
        intent=IntentCategory.OTHER,
        intent_group="other",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state=first_req.conversation_state,
    )
    second = asyncio.run(orchestrator.run(second_req))
    assert second.intent is IntentCategory.DEVICE_FAULT
    assert second.agent_type is AgentType.INTAKE
    assert second.tools_used == []
    assert "具体表现" in second.response
    assert second.question_group["type"] == "multi_select"
    assert "没声音" in [option["value"] for option in second.question_group["options"]]
    assert second_req.conversation_state["support_flow"]["safety_status"] == "clear"
    assert second_req.conversation_state["support_flow"]["stage"] == "symptom"

    third_req = Request(
        message="只有左耳没声音",
        user_id="u1",
        conv_id="clarify-1",
        intent=IntentCategory.DEVICE_FAULT,
        intent_group="aftersales",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state=second_req.conversation_state,
    )
    third = asyncio.run(orchestrator.run(third_req))
    assert third.agent_type is AgentType.INTAKE
    assert third.tools_used == []
    assert "型号" in third.response
    assert third.question_group["type"] == "text"
    assert third.question_group["options"] == []
    assert third_req.conversation_state["support_flow"]["stage"] == "device_identity"
    assert third_req.conversation_state["support_flow"]["symptom_category"] == "audio"
    assert third_req.conversation_state["entities"]["fault_symptom"] == ["只有左耳没声音"]


def test_information_readiness_gate_can_return_to_previous_question():
    from collections import deque

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=20)
    state = {}

    for message, intent in (
        ("我耳机坏了", IntentCategory.DEVICE_FAULT),
        ("没有", IntentCategory.OTHER),
        ("只有左耳没声音", IntentCategory.DEVICE_FAULT),
    ):
        req = Request(
            message=message, user_id="u1", conv_id="clarify-back",
            intent=intent, intent_group="aftersales", urgency=UrgencyLevel.LOW,
            entities={}, conversation_state=state,
        )
        result = asyncio.run(orchestrator.run(req))
        state = req.conversation_state

    assert result.question_group["stage"] == "device_identity"
    assert state["entities"]["fault_symptom"] == ["只有左耳没声音"]

    back_req = Request(
        message="返回上一问题", user_id="u1", conv_id="clarify-back",
        intent=IntentCategory.OTHER, intent_group="other", urgency=UrgencyLevel.LOW,
        entities={}, conversation_state=state,
    )
    back = asyncio.run(orchestrator.run(back_req))

    assert back.question_group["stage"] == "symptom"
    assert back_req.conversation_state["support_flow"]["stage"] == "symptom"
    assert not back_req.conversation_state["entities"].get("fault_symptom")
    assert back.tools_used == []


def test_information_readiness_gate_is_device_agnostic_and_does_not_expose_diagnostic_report():
    from collections import deque

    for message, intent in (
        ("我电脑坏了", IntentCategory.DEVICE_FAULT),
        ("这东西突然不能用了", IntentCategory.AFTERSALES),
    ):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._recent_tool_traces = deque(maxlen=20)
        req = Request(
            message=message,
            user_id="u1",
            conv_id=f"clarify-{message}",
            intent=intent,
            intent_group="aftersales",
            urgency=UrgencyLevel.LOW,
            entities={},
            conversation_state={},
        )

        result = asyncio.run(orchestrator.run(req))

        assert result.agent_type is AgentType.INTAKE
        assert result.tools_used == []
        assert all(term in result.response for term in ("有", "没有", "不确定"))
        assert not any(term in result.response for term in ("当前判断", "实际依据", "验证方式", "序列号", "故障码"))
        assert result.routing_reason.startswith("information_readiness_gate")


def test_explicit_ticket_rejection_records_a_structured_not_created_result():
    from collections import deque

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=20)
    orchestrator._composer = ResponseComposer(None, "unused")
    state = {
        "domain": "aftersales",
        "task_id": "exchange-task",
        "intent": "exchange_request",
        "entities": {"order_id": ["RF202610270001"]},
        "facts": {"recommended_resolution": "manual_exchange_review"},
        "task_progress": {"troubleshooting_completed": True, "issue_persists": True},
        "pending_action": {
            "type": "create_service_ticket",
            "ticket_type": "exchange",
            "action_id": "prepare-exchange-task",
            "status": "awaiting_user_confirmation",
        },
        "resolution_plan": {
            "goal": "prepare_exchange",
            "status": "awaiting_action",
            "steps": [],
            "next_step": "execute",
            "missing_fields": [],
        },
    }
    req = Request(
        message="先算了，保留方案但不要提交。",
        user_id="u1",
        conv_id="reject-ticket",
        request_id="reject-request",
        intent=IntentCategory.HUMAN_HANDOFF,
        intent_group="other",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state=state,
    )

    result = asyncio.run(orchestrator.run(req))

    assert result.intent is IntentCategory.EXCHANGE_REQUEST
    assert result.tools_used == []
    assert result.task_state["pending_action"]["status"] == "awaiting_user_confirmation"
    assert result.task_state["task_progress"]["ticket_created"] is False
    assert result.task_state["task_progress"]["ticket_submission_rejected"] is True
    assert result.task_state["last_action_decision"]["decision"] == "rejected"
    assert "不会执行建单" in result.response


def test_active_clarification_does_not_capture_an_explicit_new_business_intent():
    from core.support_flow import advance_device_support_flow

    state = {
        "domain": "aftersales",
        "entities": {},
        "support_flow": {
            "type": "device_fault_clarification",
            "status": "active",
            "stage": "safety_check",
        },
    }

    decision = advance_device_support_flow(
        state,
        message="我想查询工单 RF-T20261020-001",
        intent="ticket_status",
        urgency="LOW",
        request_id="new-task",
    )

    assert decision is None


def test_complete_device_and_symptom_bypass_readiness_gate():
    from core.support_flow import advance_device_support_flow

    state = {
        "domain": "aftersales",
        "entities": {
            "product_model": ["FlowBuds Pro"],
            "fault_symptom": ["左耳无声"],
        },
    }

    decision = advance_device_support_flow(
        state,
        message="FlowBuds Pro 左耳无声",
        intent="device_fault",
        urgency="LOW",
        request_id="ready",
    )

    assert decision is None
    assert "support_flow" not in state


@pytest.mark.parametrize(
    ("message", "entities"),
    [
        (
            "FlowBuds Pro 序列号 FBP-CN-A10001，充电失败并提示 E201。",
            {"product_model": ["FlowBuds Pro"], "serial_number": ["FBP-CN-A10001"], "fault_code": ["E201"]},
        ),
        (
            "FCS-CN-C30003 的 FlowClean S1 报 E301，滚刷不转。",
            {"product_model": ["FlowClean S1"], "serial_number": ["FCS-CN-C30003"], "fault_code": ["E301"]},
        ),
        (
            "FBP-CN-A10001 没有发热、冒烟或漏液，只是充不上电并报 E201。",
            {"product_model": [], "serial_number": ["FBP-CN-A10001"], "fault_code": ["E201"]},
        ),
    ],
)
def test_complete_device_identity_and_fault_code_bypass_readiness_gate(message, entities):
    from core.support_flow import advance_device_support_flow

    state = {"domain": "aftersales", "entities": entities}
    decision = advance_device_support_flow(
        state,
        message=message,
        intent="device_fault",
        urgency="LOW",
        request_id="ready-by-code",
    )

    assert decision is None
    assert "support_flow" not in state


def test_bluetooth_product_name_is_not_mistaken_for_connectivity_symptom():
    from core.support_flow import advance_device_support_flow

    state = {"domain": "aftersales", "entities": {}}
    first = advance_device_support_flow(
        state,
        message="我的蓝牙耳塞好像出毛病了",
        intent="device_fault",
        urgency="LOW",
        request_id="bluetooth-product",
    )
    second = advance_device_support_flow(
        state,
        message="都没有",
        intent="device_fault",
        urgency="LOW",
        request_id="bluetooth-safe",
    )
    third = advance_device_support_flow(
        state,
        message="充不进去电",
        intent="device_fault",
        urgency="LOW",
        request_id="bluetooth-symptom",
    )

    assert first.stage == "safety_check"
    assert second.stage == "symptom"
    assert third.stage == "device_identity"
    assert state["support_flow"]["symptom_category"] == "charging"
    assert state["support_flow"].get("product_model") is None


def test_safety_risk_detection_is_negation_aware():
    from core.intent_recognizer import IntentRecognizer
    from core.support_flow import detect_safety_risks

    assert detect_safety_risks("没有鼓包、没有冒烟，也没有烧焦味") == []
    assert detect_safety_risks("没有发烫、鼓包、冒烟、烧焦味、漏液或触电") == []
    assert detect_safety_risks("没有鼓包，但有烧焦味") == ["烧焦味"]
    assert detect_safety_risks("没有鼓包，设备后来开始冒烟") == ["冒烟"]
    assert detect_safety_risks("外壳变形而且发烫") == ["鼓包/变形", "异常发热"]

    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    assert recognizer._urgency(
        "设备没有鼓包、没有冒烟",
        IntentCategory.DEVICE_FAULT,
    ) is UrgencyLevel.LOW
    assert recognizer._urgency(
        "FlowBuds Pro 耳机没声音，没有发烫、鼓包、冒烟、烧焦味、漏液或触电",
        IntentCategory.DEVICE_FAULT,
    ) is UrgencyLevel.LOW
    assert recognizer._urgency(
        "没有鼓包，但有烧焦味",
        IntentCategory.DEVICE_FAULT,
    ) is UrgencyLevel.CRITICAL


def test_cross_product_display_symptom_advances_to_device_identity():
    from core.support_flow import advance_device_support_flow

    state = {}
    advance_device_support_flow(
        state,
        message="我家摄像头罢工了",
        intent="device_fault",
        urgency="LOW",
        request_id="camera-start",
    )
    advance_device_support_flow(
        state,
        message="没有这些危险情况",
        intent="device_fault",
        urgency="LOW",
        request_id="camera-safe",
    )
    decision = advance_device_support_flow(
        state,
        message="设备在线但是画面全黑",
        intent="device_fault",
        urgency="LOW",
        request_id="camera-symptom",
    )

    assert decision.stage == "device_identity"
    assert state["support_flow"]["symptom_category"] == "display"
    assert state["entities"]["fault_symptom"] == ["设备在线但是画面全黑"]


def test_quality_complaint_missing_fields_stop_before_agent_and_return_component():
    from collections import deque

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._composer = ResponseComposer.__new__(ResponseComposer)
    orchestrator._recent_tool_traces = deque(maxlen=20)
    req = Request(
        message="同一个故障已经维修三次仍然复发，请登记质量投诉，不是再开普通维修。",
        user_id="u1",
        conv_id="quality-gate",
        intent=IntentCategory.QUALITY_COMPLAINT,
        intent_group="aftersales",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state={},
    )

    result = asyncio.run(orchestrator.run(req))

    assert result.primary_agent is AgentType.INTAKE
    assert result.tools_used == []
    assert result.question_group["stage"] == "collect_order_id"
    assert "质量投诉" in result.response
    assert "普通维修" in result.response


def test_safety_entity_extraction_does_not_turn_negation_into_positive_fact():
    from core.intent_recognizer import IntentRecognizer

    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    entities = recognizer._extract_entities(
        "FlowBuds Pro 没有鼓包、没有冒烟，左耳没声音"
    )

    assert entities["fault_symptom"] == ["没声音"]


def test_direct_safety_risk_blocks_normal_flow_and_prepares_handoff():
    from collections import deque

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.ESCALATION: [EscalationAgent(FakeClient(), "test-model")],
    }
    orchestrator._recent_tool_traces = deque(maxlen=20)
    req = Request(
        message="FlowBuds Pro 充电盒电池鼓包，还有烧焦味",
        user_id="u1",
        conv_id="safety-direct",
        intent=IntentCategory.DEVICE_FAULT,
        intent_group="aftersales",
        urgency=UrgencyLevel.CRITICAL,
        entities={"product_model": ["FlowBuds Pro"], "fault_symptom": ["电池鼓包", "烧焦味"]},
        conversation_state={},
    )

    result = asyncio.run(orchestrator.run(req))

    assert result.agent_type is AgentType.ESCALATION
    assert result.escalated is True
    assert result.tools_used == []
    assert "立即停止使用和充电" in result.response
    assert "不要自行拆机" in result.response
    flow = req.conversation_state["support_flow"]
    assert flow["status"] == "blocked"
    assert flow["stage"] == "safety_blocked"
    assert flow["risk_signals"] == ["鼓包/变形", "烧焦味"]
    assert req.conversation_state["resolution_plan"]["status"] == "handoff_required"
    assert req.conversation_state["handoff_summary"]["reason_code"] == "device_safety_risk"


def test_selected_safety_risk_stops_progressive_flow_immediately():
    from collections import deque

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.ESCALATION: [EscalationAgent(FakeClient(), "test-model")],
    }
    orchestrator._recent_tool_traces = deque(maxlen=20)

    first_req = Request(
        message="我耳机坏了",
        user_id="u1",
        conv_id="safety-branch",
        intent=IntentCategory.DEVICE_FAULT,
        intent_group="aftersales",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state={},
    )
    first = asyncio.run(orchestrator.run(first_req))
    assert first.tools_used == []

    second_req = Request(
        message="有",
        user_id="u1",
        conv_id="safety-branch",
        intent=IntentCategory.OTHER,
        intent_group="other",
        urgency=UrgencyLevel.LOW,
        entities={},
        conversation_state=first_req.conversation_state,
    )
    second = asyncio.run(orchestrator.run(second_req))
    assert second.tools_used == []
    assert second_req.conversation_state["support_flow"]["stage"] == "safety_detail"
    assert "可以多选" in second.response
    assert second.question_group["type"] == "multi_select"
    assert {option["value"] for option in second.question_group["options"]} >= {"明显发烫", "冒烟", "烧焦味"}

    third_req = Request(
        message="电池鼓包",
        user_id="u1",
        conv_id="safety-branch",
        intent=IntentCategory.DEVICE_FAULT,
        intent_group="aftersales",
        urgency=UrgencyLevel.CRITICAL,
        entities={"fault_symptom": ["电池鼓包"]},
        conversation_state=second_req.conversation_state,
    )
    third = asyncio.run(orchestrator.run(third_req))

    assert third.agent_type is AgentType.ESCALATION
    assert third.escalated is True
    assert third.tools_used == []
    assert third.question_group is None
    assert third_req.conversation_state["support_flow"]["stage"] == "safety_blocked"
    assert third_req.conversation_state["support_flow"]["risk_signals"] == ["鼓包/变形"]
    assert "立即停止使用和充电" in third.response


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TECHNICAL, "先排查 Token 是否过期。", True),
        AgentResponse(AgentType.BILLING, "请提供两笔扣款的时间和金额。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 Token 是否过期。")
    assert "补充说明" in content
    assert "两笔扣款" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_resolution_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(FakeClient(), "test")],
        AgentType.TECHNICAL: [TechnicalAgent(FakeClient(), "test")],
        AgentType.BILLING: [BillingAgent(FakeClient(), "test")],
    }

    decision = orchestrator._route_decision(make_request(
        message="耳机左耳没声音，我想换货",
        intent=IntentCategory.EXCHANGE_REQUEST,
        intent_group="aftersales",
        entities={"fault_symptom": ["左耳无声"], "serial_number": ["FBP-CN-A10001"]},
    ))

    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
    assert decision.multi_agent is True


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields", "lookup_order", "get_service_ticket"}
    assert technical_tools == {"lookup_device", "lookup_fault_code", "run_device_diagnostic"}
    assert billing_tools == {"lookup_order", "lookup_device", "check_warranty", "recommend_resolution", "prepare_service_ticket", "create_service_ticket", "get_service_ticket"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & technical_tools
    assert "create_service_ticket" not in technical_tools
    assert "run_device_diagnostic" not in billing_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "7 天内可退款"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, technical, billing, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_fault_code"]

    try:
        agent._validate_tool_input(spec, {"fault_code": "E101", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_fault_code"
        input = {"fault_code": "E101"}

    class TextBlock:
        type = "text"
        text = "已根据 E101 故障码给出安全排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_fault_code"]
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool_name"] == "lookup_fault_code"
    assert response.tool_traces[0]["input"] == {"fault_code": "E101"}
    assert response.tool_traces[0]["success"] is True
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_device",
        "lookup_fault_code",
        "run_device_diagnostic",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_three_sequential_tools_still_leave_a_turn_for_the_final_answer():
    class ToolUseBlock:
        type = "tool_use"

        def __init__(self, index, name, inputs):
            self.id = f"toolu_{index}"
            self.name = name
            self.input = inputs

    class TextBlock:
        type = "text"
        text = "设备、故障码与低风险诊断均已完成。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock(1, "lookup_device", {"serial_number": "FBP-CN-A10001"})]})(),
                type("Response", (), {"content": [ToolUseBlock(2, "lookup_fault_code", {"fault_code": "E101"})]})(),
                type("Response", (), {"content": [ToolUseBlock(3, "run_device_diagnostic", {"symptom": "左耳无声"})]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request(
        message="SN FBP-CN-A10001 左耳无声，故障码 E101",
        entities={
            "serial_number": ["FBP-CN-A10001"],
            "fault_code": ["E101"],
            "fault_symptom": ["左耳无声"],
        },
    )))

    assert response.success is True
    assert response.tools_used == ["lookup_device", "lookup_fault_code", "run_device_diagnostic"]
    assert len(client.calls) == 4


def test_concurrent_requests_keep_tool_traces_isolated():
    class ToolUseBlock:
        type = "tool_use"

        def __init__(self, code):
            self.id = f"toolu_{code}"
            self.name = "lookup_error_code"
            self.input = {"error_code": code}

    class TextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class ConcurrentClient:
        class Messages:
            async def create(inner, **kwargs):
                messages = kwargs["messages"]
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    code = "500" if "500" in str(last_content) else "401"
                    await asyncio.sleep(0)
                    return type("Response", (), {"content": [TextBlock(f"{code} 排查完成")]})()
                code = "500" if "500" in str(last_content) else "401"
                await asyncio.sleep(0.01 if code == "401" else 0)
                return type("Response", (), {"content": [ToolUseBlock(code)]})()

        def __init__(self):
            self.messages = self.Messages()

    async def run_both():
        agent = TechnicalAgent(ConcurrentClient(), "test-model")
        return await asyncio.gather(
            agent.handle(make_request(message="登录报 401", entities={"error_code": ["401"]})),
            agent.handle(make_request(message="页面报 500", entities={"error_code": ["500"]})),
        )

    first, second = asyncio.run(run_both())
    assert first.tool_traces[0]["input"] == {"error_code": "401"}
    assert second.tool_traces[0]["input"] == {"error_code": "500"}


def test_monitor_penalty_falls_back_to_general_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    technical.stats.monitor_penalty = 0.5
    orchestrator._pool = {
        AgentType.GENERAL: [general],
        AgentType.TECHNICAL: [technical],
        AgentType.BILLING: [billing],
    }

    decision = orchestrator._route_decision(make_request(
        message="登录报 401",
        entities={"error_code": ["401"]},
    ))

    assert decision.primary_agent is AgentType.GENERAL
    assert decision.supporting_agents == []
    assert "monitor_fallback=[diagnostic]" in decision.reason


def test_recovery_probe_allows_only_one_request_and_recovers():
    import time
    from types import SimpleNamespace
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def reply(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(content=[{'type':'text','text':'恢复成功'}])
        agent = TechnicalAgent(SimpleNamespace(messages=SimpleNamespace(create=reply)), 'test')
        agent.stats.set_penalty(.8)
        assert not agent.stats.available()
        agent.stats.retry_at = time.monotonic() - 1
        first = asyncio.create_task(agent.handle(make_request()))
        await entered.wait()
        assert not agent.stats.available()
        second = await agent.handle(make_request())
        assert not second.success
        release.set()
        assert (await first).success
        assert agent.stats.available()
        assert agent.stats.monitor_penalty == 0
        assert agent.stats.recent()[0] == 1
    asyncio.run(run())


def test_failed_probe_reopens_and_window_expires():
    import time
    from agents.agent_orchestrator import AgentStats
    stats = AgentStats()
    stats.set_penalty(.8)
    stats.retry_at = time.monotonic() - 1
    stats.probe_running = True
    stats.record(False, 10)
    assert not stats.available()
    assert not stats.probe_running
    stats.samples.clear()
    stats.samples.append((time.monotonic()-301, False, 10000))
    assert stats.recent() == (0, 1.0, 0.0)


def test_cancelled_probe_releases_exclusive_slot():
    import time
    from types import SimpleNamespace
    async def run():
        entered = asyncio.Event()
        async def reply(**kwargs):
            entered.set()
            await asyncio.sleep(30)
        agent = TechnicalAgent(SimpleNamespace(messages=SimpleNamespace(create=reply)), 'test')
        agent.stats.set_penalty(.8)
        agent.stats.retry_at = time.monotonic()-1
        task = asyncio.create_task(agent.handle(make_request()))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not agent.stats.probe_running
        assert not agent.stats.available()
    asyncio.run(run())
