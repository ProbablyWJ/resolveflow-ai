"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 技术排障建议
  - 账单字段核验
  - 人工升级摘要
  - 共享知识库 RAG

订单查询、退款执行、账单修改等需要真实业务系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union, Tuple

from core.action_confirmation import is_explicit_ticket_confirmation

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler
    allowed_agents: Optional[Tuple[str, ...]] = None
    retry_safe: bool = True


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
    retry_safe: bool = True,
    allowed_agents: Optional[Tuple[str, ...]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
        retry_safe=retry_safe,
        allowed_agents=allowed_agents,
    )


_DEMO_DATA_PATH = Path(__file__).resolve().parents[1] / "config" / "aftersales_demo.json"
_DEMO_DATA = json.loads(_DEMO_DATA_PATH.read_text(encoding="utf-8"))
_TICKET_LOCK = threading.Lock()
_TICKETS: Dict[str, Dict[str, Any]] = dict(_DEMO_DATA.get("tickets", {}))


def _first(req: Request, field: str, fallback: str = "") -> str:
    values = (req.entities or {}).get(field, [])
    return str(values[0]).strip() if values else fallback


def _aftersales_state(req: Request) -> Dict[str, Any]:
    """取得可被业务工具更新的任务状态；工具结果仍需由前端/记忆层持久化。"""
    if not isinstance(req.conversation_state, dict):
        req.conversation_state = {}
    state = req.conversation_state
    state.setdefault("domain", "aftersales")
    state.setdefault("task_id", req.request_id)
    state.setdefault("entities", req.entities or {})
    state.setdefault("facts", {})
    state.setdefault("evidence_refs", [])
    state.setdefault("pending_action", None)
    state.setdefault("tool_budget", {"calls_used": 0, "calls_max": 16, "replans_used": 0, "replans_max": 2})
    return state


def _consume_tool_budget(req: Request, tool_name: str) -> Optional[Dict[str, Any]]:
    state = _aftersales_state(req)
    budget = state["tool_budget"]
    used, maximum = int(budget.get("calls_used", 0)), int(budget.get("calls_max", 16))
    if used >= maximum:
        from core.conversation_state import replan_after_failure
        replan_after_failure(state, tool_name, "工具调用预算已耗尽", "escalate", terminal=True)
        return {
            "success": False,
            "error": "当前任务已达到工具调用上限，需要人工核验或新建任务",
            "tool_name": tool_name,
            "calls_used": used,
            "calls_max": maximum,
        }
    budget["calls_used"] = used + 1
    return None


def _record_fact(req: Request, key: str, value: Any, evidence_ref: Optional[str] = None) -> None:
    state = _aftersales_state(req)
    state["facts"][key] = value
    if evidence_ref and evidence_ref not in state["evidence_refs"]:
        state["evidence_refs"].append(evidence_ref)


def _refresh_state(req: Request) -> None:
    from core.conversation_state import refresh_resolution_plan
    refresh_resolution_plan(req.conversation_state, intent=req.intent.value if req.intent else None)


def _replan(req: Request, failed_step: str, reason: str, fallback_step: str, *, terminal: bool = False) -> None:
    from core.conversation_state import replan_after_failure
    replan_after_failure(_aftersales_state(req), failed_step, reason, fallback_step, terminal=terminal)


def lookup_order(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "lookup_order"):
        return error
    order_id = str(args.get("order_id") or _first(req, "order_id")).upper()
    order = _DEMO_DATA["orders"].get(order_id)
    if not order:
        _replan(req, "verify_order", "演示订单不存在", "collect_order_id")
        return {"success": False, "found": False, "order_id": order_id, "error": "演示订单不存在"}
    _record_fact(req, "order_verified", True, f"business://orders/{order_id}")
    _record_fact(req, "verified_order", {"order_id": order_id, **order})
    state_entities = _aftersales_state(req)["entities"]
    state_entities["order_id"] = [order_id]
    state_entities["product_model"] = [order["product_model"]]
    state_entities["serial_number"] = [order["serial_number"]]
    req.entities = state_entities
    _refresh_state(req)
    return {"success": True, "found": True, "source": "demo_business_system", "dataset_version": _DEMO_DATA["dataset_version"], "order": {"order_id": order_id, **order}}


def lookup_device(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "lookup_device"):
        return error
    serial_number = str(args.get("serial_number") or _first(req, "serial_number")).upper()
    device = _DEMO_DATA["devices"].get(serial_number)
    if not device:
        _replan(req, "verify_device", "演示设备不存在", "collect_serial_number")
        return {"success": False, "found": False, "serial_number": serial_number, "error": "演示设备不存在"}
    _record_fact(req, "device_verified", True, f"business://devices/{serial_number}")
    _record_fact(req, "verified_device", {"serial_number": serial_number, **device})
    state_entities = _aftersales_state(req)["entities"]
    state_entities["serial_number"] = [serial_number]
    state_entities["product_model"] = [device["product_model"]]
    req.entities = state_entities
    _refresh_state(req)
    return {"success": True, "found": True, "source": "demo_business_system", "dataset_version": _DEMO_DATA["dataset_version"], "device": {"serial_number": serial_number, **device}}


def check_warranty(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "check_warranty"):
        return error
    order_id = str(args.get("order_id") or _first(req, "order_id")).upper()
    serial_number = str(args.get("serial_number") or _first(req, "serial_number")).upper()
    order = _DEMO_DATA["orders"].get(order_id)
    if not order and serial_number:
        order_id, order = next(((key, value) for key, value in _DEMO_DATA["orders"].items() if value["serial_number"] == serial_number), ("", None))
    if not order:
        _replan(req, "verify_warranty", "找不到可核验的演示订单", "verify_order")
        return {"success": False, "verified": False, "error": "找不到可核验的演示订单"}
    if serial_number and order["serial_number"] != serial_number:
        _replan(req, "verify_warranty", "订单与设备序列号不匹配", "collect_matching_device")
        return {"success": False, "verified": False, "error": "订单与设备序列号不匹配"}
    warranty_until = date.fromisoformat(order["warranty_until"])
    status = "in_warranty" if date.today() <= warranty_until else "out_of_warranty"
    _record_fact(req, "order_verified", True, f"business://orders/{order_id}")
    _record_fact(req, "device_verified", True, f"business://devices/{order['serial_number']}")
    _record_fact(req, "warranty_status", status, "policy://warranty/standard-v1")
    _record_fact(req, "warranty_until", order["warranty_until"])
    _refresh_state(req)
    return {"success": True, "verified": True, "source": "demo_business_system", "order_id": order_id, "serial_number": order["serial_number"], "warranty_status": status, "warranty_until": order["warranty_until"], "policy_ref": "policy://warranty/standard-v1"}


def lookup_fault_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "lookup_fault_code"):
        return error
    code = str(args.get("fault_code") or _first(req, "fault_code")).upper()
    fault = _DEMO_DATA["fault_codes"].get(code)
    if not fault:
        _replan(req, "retrieve_fault_code", "知识中没有该故障码", "collect_fault_symptom")
        return {"success": False, "found": False, "fault_code": code, "error": "未找到该演示故障码"}
    _record_fact(req, "fault_code_verified", code, f"manual://fault-codes/{code}")
    # 将用户真正需要的手册结论保存为结构化事实，回复层才能解释
    # “故障码代表什么、什么情况下必须停止”，而不只展示已核验状态。
    _record_fact(req, "verified_fault", {
        "fault_code": code,
        "product_model": fault.get("product_model"),
        "meaning": fault.get("meaning"),
        "stop_conditions": list(fault.get("stop_conditions", [])),
    })
    _refresh_state(req)
    return {"success": True, "found": True, "source": "demo_device_manual", "fault_code": code, **fault, "evidence_ref": f"manual://fault-codes/{code}"}


def _diagnostic_steps_for_context(
    *,
    product_category: str,
    product_model: str,
    symptom_category: str,
) -> List[str]:
    """在没有已核验故障码时，根据设备和现象给出可泛化的低风险步骤。

    这里只使用用户已提供的上下文，不推断故障码或硬件结论。
    型号明确匹配演示手册时优先使用手册步骤，其余情况使用设备类型无关的安全排查。
    """
    if product_model.lower() == "flowbuds pro" and symptom_category in {"audio", "connectivity"}:
        return list(_DEMO_DATA["fault_codes"]["E101"]["safe_steps"])
    if product_model.lower() == "flowbuds pro" and symptom_category == "charging":
        return list(_DEMO_DATA["fault_codes"]["E201"]["safe_steps"])

    if symptom_category == "audio":
        if product_category == "耳机":
            return [
                "确认耳机有电，且手机音量和音频输出设备选择正确",
                "将耳机放回充电盒 10 秒后取出，再重新连接",
                "在蓝牙设置中忽略该设备后重新配对",
            ]
        return [
            "确认系统未静音，并将音频输出切换到预期设备",
            "断开并重新连接耳机或扬声器，再播放系统测试音",
            "关闭音频应用后重启设备，再次测试",
        ]
    if symptom_category == "connectivity":
        return [
            "确认设备有电、连接功能已开启，并靠近要连接的设备",
            "断开当前连接，在系统设置中删除旧记录后重新配对",
            "重启两端设备后再连接一次",
        ]
    if symptom_category == "charging":
        return [
            "先断开电源，检查充电口和接触点是否有可见异物",
            "更换合规充电线和电源适配器后重试",
            "连接后观察指示灯或充电图标 30 秒",
        ]
    if symptom_category == "power":
        return [
            "使用合规电源连续充电 30 分钟",
            "断开外接设备后，按设备正常方式重新开机",
            "观察指示灯、声音或屏幕是否有反应",
        ]
    if symptom_category == "display":
        return [
            "断开非必要外接设备后重启",
            "调低亮度以外的显示设置保持默认，再测试一次",
            "若使用外接屏幕，重新插接线缆并切换显示输出",
        ]
    return [
        "确认设备电量与指示灯状态",
        "断开非必要外设后重启设备",
        "重现一次问题，并告诉我具体卡在哪一步",
    ]


def run_device_diagnostic(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "run_device_diagnostic"):
        return error
    from core.support_flow import detect_product_category, detect_safety_risks, detect_symptom_category
    # 安全升级只接受用户原句或已由确定性闸门保存的风险信号作为依据。
    # 结构化实体和工具参数可能由模型生成，不能把“没有冒烟”或 Prompt 示例
    # 误当成用户已经报告的真实安全事件。
    state = req.conversation_state if isinstance(req.conversation_state, dict) else {}
    flow = state.get("support_flow") if isinstance(state.get("support_flow"), dict) else {}
    saved_risks = list(flow.get("risk_signals") or [])
    reported_risks = saved_risks or detect_safety_risks(req.message)
    if reported_risks:
        _record_fact(req, "diagnosis_status", "stopped_for_safety", "safety://battery-and-electrical-risk-v1")
        _refresh_state(req)
        _replan(req, "diagnose", "检测到设备安全风险", "escalate", terminal=True)
        return {"success": False, "safe_to_continue": False, "requires_escalation": True, "reason": "检测到设备安全风险，停止普通诊断", "next_action": "停止使用和充电，远离热源与可燃物并转人工"}
    entity_symptoms = [
        str(value).strip()
        for value in (req.entities or {}).get("fault_symptom", [])
        if str(value).strip() and not detect_safety_risks(str(value))
    ]
    tool_symptom = str(args.get("symptom") or "").strip()
    symptom = str(
        (entity_symptoms[0] if entity_symptoms else "")
        or (tool_symptom if not detect_safety_risks(tool_symptom) else "")
    ).strip()
    code = str(_first(req, "fault_code") or args.get("fault_code") or "").upper()
    fault = _DEMO_DATA["fault_codes"].get(code) if code else None
    model = _first(req, "product_model") or str(flow.get("product_model") or "")
    product_category = str(
        flow.get("product_category")
        or ("耳机" if model.lower() == "flowbuds pro" else "")
        or detect_product_category(" ".join((req.message or "", model)))
        or ""
    )
    symptom_category = str(
        flow.get("symptom_category")
        or detect_symptom_category(symptom)
        or ""
    )
    steps = fault["safe_steps"] if fault else _diagnostic_steps_for_context(
        product_category=product_category,
        product_model=model,
        symptom_category=symptom_category,
    )
    diagnosis = {"symptom": symptom or "未提供", "fault_code": code or None, "safe_steps": steps[:3], "result": "low_risk_steps_provided"}
    _record_fact(req, "diagnosis_status", "completed", f"diagnostic://{req.request_id}")
    _record_fact(req, "diagnosis", diagnosis)
    _refresh_state(req)
    return {"success": True, "safe_to_continue": True, "source": "deterministic_diagnostic", "diagnosis": diagnosis, "requires_escalation": False}


def recommend_resolution(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "recommend_resolution"):
        return error
    state = _aftersales_state(req)
    facts = state["facts"]
    desired = str(args.get("desired_action", "repair"))
    if not (facts.get("order_verified") or facts.get("device_verified")):
        _replan(req, "decide", "缺少已核验订单或设备", "verify")
        return {"success": False, "error": "需要先核验订单或设备"}
    if facts.get("diagnosis_status") not in {"completed", "not_required"} and desired in {"repair", "exchange"}:
        _replan(req, "decide", "尚未完成安全诊断", "diagnose")
        return {"success": False, "error": "需要先完成安全诊断或明确无需诊断"}
    warranty = facts.get("warranty_status", "unknown")
    if desired == "exchange":
        recommended = "manual_exchange_review" if warranty == "in_warranty" else "paid_repair_or_manual_review"
    elif desired == "parts":
        recommended = "parts_service_review"
    else:
        recommended = "in_warranty_repair" if warranty == "in_warranty" else "paid_repair_quote"
    _record_fact(req, "recommended_resolution", recommended, "policy://aftersales/decision-v1")
    _refresh_state(req)
    return {"success": True, "source": "deterministic_policy_engine", "desired_action": desired, "warranty_status": warranty, "recommended_resolution": recommended, "requires_manual_approval": "manual" in recommended, "policy_ref": "policy://aftersales/decision-v1"}


def prepare_service_ticket(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "prepare_service_ticket"):
        return error
    state = _aftersales_state(req)
    ticket_type = str(args.get("ticket_type", "repair"))
    if not state["facts"].get("recommended_resolution"):
        _replan(req, "prepare_ticket", "尚未形成售后方案", "decide")
        return {"success": False, "error": "需要先形成售后方案"}
    action_id = f"prepare-{state['task_id']}"
    action = {
        "action_id": action_id,
        "type": "create_service_ticket",
        "ticket_type": ticket_type,
        "order_id": _first(req, "order_id"),
        "serial_number": _first(req, "serial_number"),
        "symptom": _first(req, "fault_symptom"),
        "resolution": state["facts"].get("recommended_resolution"),
        "requires_confirmation": True,
        "status": "awaiting_user_confirmation",
    }
    state["pending_action"] = action
    _refresh_state(req)
    return {"success": True, "prepared": True, "pending_action": action, "confirmation_prompt": "请核对以上信息；若同意创建演示工单，请回复“确认创建工单”。"}


def create_service_ticket(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    state = _aftersales_state(req)
    pending = state.get("pending_action") or {}
    action_id = str(args.get("action_id", ""))
    confirmed = is_explicit_ticket_confirmation(req.message)
    with _TICKET_LOCK:
        existing = next((ticket_id for ticket_id, ticket in _TICKETS.items() if ticket.get("action_id") == action_id), None)
    if existing and confirmed:
        return {"success": True, "created": False, "idempotent_replay": True, "ticket_id": existing, "ticket": _TICKETS[existing]}
    if error := _consume_tool_budget(req, "create_service_ticket"):
        return error
    if pending.get("type") != "create_service_ticket" or pending.get("action_id") != action_id:
        return {"success": False, "created": False, "error": "没有匹配的待确认工单操作"}
    if not confirmed:
        return {"success": False, "created": False, "error": "尚未收到用户对创建工单的明确确认"}
    with _TICKET_LOCK:
        ticket_id = f"RF-T{datetime.now().strftime('%Y%m%d')}-{len(_TICKETS) + 1:03d}"
        ticket = {
            "action_id": action_id,
            "user_id": req.user_id,
            "order_id": pending.get("order_id"),
            "serial_number": pending.get("serial_number"),
            "ticket_type": pending.get("ticket_type"),
            "symptom": pending.get("symptom"),
            "resolution": pending.get("resolution"),
            "status": "created",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "next_action": "等待售后人员审核演示工单",
        }
        _TICKETS[ticket_id] = ticket
    state["entities"]["ticket_id"] = [ticket_id]
    req.entities = state["entities"]
    state["pending_action"] = None
    _record_fact(req, "ticket_created", True, f"business://tickets/{ticket_id}")
    # 创建结果已包含权威状态，直接写入任务事实；无需为同一状态再做冗余查询。
    _record_fact(req, "ticket_status", ticket["status"])
    _refresh_state(req)
    return {"success": True, "created": True, "source": "demo_business_system", "ticket_id": ticket_id, "ticket": ticket}


def get_service_ticket(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    if error := _consume_tool_budget(req, "get_service_ticket"):
        return error
    ticket_id = str(args.get("ticket_id") or _first(req, "ticket_id")).upper()
    ticket = _TICKETS.get(ticket_id)
    if not ticket:
        return {"success": False, "found": False, "ticket_id": ticket_id, "error": "演示工单不存在"}
    _record_fact(req, "ticket_status", ticket["status"], f"business://tickets/{ticket_id}")
    _record_fact(req, "verified_ticket", {"ticket_id": ticket_id, **ticket})
    _refresh_state(req)
    return {"success": True, "found": True, "source": "demo_business_system", "ticket_id": ticket_id, "ticket": ticket}


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent == "device_fault":
        fields = ["设备型号或序列号", "具体故障现象"]
    elif intent in {"warranty_inquiry", "repair_request", "exchange_request"}:
        fields = ["订单号", "设备序列号", "故障现象"]
    elif intent == "parts_request":
        fields = ["设备型号或序列号", "需要的配件"]
    elif intent == "ticket_status":
        fields = ["工单号"]
    elif intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给人工客服的结构化摘要。"""
    state = req.conversation_state if isinstance(req.conversation_state, dict) else {}
    plan = state.get("resolution_plan") if isinstance(state.get("resolution_plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    return {
        "success": True,
        "status": "prepared",
        "queue_connected": False,
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工客服继续核验"))[:120],
        "reason_code": str(args.get("reason_code", "manual_review_required"))[:80],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "facts": state.get("facts", {}),
        "evidence_refs": state.get("evidence_refs", []),
        "pending_action": state.get("pending_action"),
        "completed_steps": [
            step.get("id")
            for step in steps
            if isinstance(step, dict) and step.get("status") == "completed" and step.get("id")
        ],
        "missing_fields": plan.get("missing_fields", []),
        "tool_attempts": list(req.tool_attempts),
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = args.get("top_k", 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            _replan(req, "retrieve", "RAG 工具未初始化", "escalate", terminal=True)
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            _replan(req, "retrieve", getattr(result, "error", "知识库检索失败"), "retry_retrieve")
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        response = {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
        }
        if isinstance(req.conversation_state, dict):
            refs = req.conversation_state.setdefault("evidence_refs", [])
            for item in result.data or []:
                ref = f"rag://{item.get('document_id', 'unknown')}@{item.get('version', 'unknown')}"
                if ref not in refs:
                    refs.append(ref)
            _refresh_state(req)
        return response

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
        "lookup_order": make_tool(
            "lookup_order",
            "在本地演示业务系统中核验订单，并返回对应设备、购买时间和保修截止日。",
            {"order_id": {"type": "string", "description": "演示订单号，如 RF202610270001"}},
            lookup_order,
        ),
        "get_service_ticket": make_tool(
            "get_service_ticket",
            "查询已存在的演示售后工单状态；不会查询真实品牌系统。",
            {"ticket_id": {"type": "string", "description": "演示工单号，如 RF-T20261020-001"}},
            get_service_ticket,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_device": make_tool(
            "lookup_device",
            "在本地演示业务系统中核验设备序列号、型号、激活状态和固件版本。",
            {"serial_number": {"type": "string", "description": "设备序列号，如 FBP-CN-A10001"}},
            lookup_device,
        ),
        "lookup_fault_code": make_tool(
            "lookup_fault_code",
            "查询消费电子设备演示故障码的含义、安全步骤和停止条件。",
            {"fault_code": {"type": "string", "description": "故障码，如 E101"}},
            lookup_fault_code,
        ),
        "run_device_diagnostic": make_tool(
            "run_device_diagnostic",
            "根据已知现象和故障码生成确定性的低风险诊断；检测到电池或电气风险会停止。",
            {
                "symptom": {"type": "string", "description": "故障现象"},
                "fault_code": {"type": "string", "description": "可选故障码"},
            },
            run_device_diagnostic,
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_order": make_tool(
            "lookup_order",
            "在本地演示业务系统中核验订单和对应设备。",
            {"order_id": {"type": "string", "description": "演示订单号"}},
            lookup_order,
        ),
        "lookup_device": make_tool(
            "lookup_device",
            "在本地演示业务系统中核验设备。",
            {"serial_number": {"type": "string", "description": "设备序列号"}},
            lookup_device,
        ),
        "check_warranty": make_tool(
            "check_warranty",
            "基于本地演示订单与设备数据确定性核验保修状态。",
            {
                "order_id": {"type": "string", "description": "演示订单号"},
                "serial_number": {"type": "string", "description": "设备序列号"},
            },
            check_warranty,
        ),
        "recommend_resolution": make_tool(
            "recommend_resolution",
            "根据已核验事实、诊断和保修状态形成维修、换货或配件处理建议。",
            {"desired_action": {"type": "string", "enum": ["repair", "exchange", "parts"], "description": "用户期望的处理方式"}},
            recommend_resolution,
            required=["desired_action"],
        ),
        "prepare_service_ticket": make_tool(
            "prepare_service_ticket",
            "整理待创建的演示售后工单并进入等待确认状态，不会直接创建。",
            {"ticket_type": {"type": "string", "enum": ["repair", "exchange", "parts", "complaint"], "description": "工单类型"}},
            prepare_service_ticket,
            required=["ticket_type"],
        ),
        "create_service_ticket": make_tool(
            "create_service_ticket",
            "仅在存在匹配的待确认动作且用户明确说确认创建时，创建幂等的演示工单。",
            {"action_id": {"type": "string", "description": "prepare_service_ticket 返回的 action_id"}},
            create_service_ticket,
            required=["action_id"],
            retry_safe=False,
        ),
        "get_service_ticket": make_tool(
            "get_service_ticket",
            "查询演示售后工单状态。",
            {"ticket_id": {"type": "string", "description": "演示工单号"}},
            get_service_ticket,
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
