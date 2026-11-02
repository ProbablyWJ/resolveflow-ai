"""设备故障的信息就绪门控。

这里只保存用户可观测、可验证的业务状态，不保存模型思考过程。
目标是在信息不足时阻止诊断 Agent 和工具过早执行。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


FLOW_TYPE = "device_fault_clarification"
ACTIVE_STAGES = {"safety_check", "safety_detail", "symptom", "device_identity"}

_GENERIC_FAILURE = re.compile(
    r"(?:坏了|坏掉|不能用|不能用了|用不了|出问题|有问题|故障|不正常|不对劲)"
)

_PRODUCT_CATEGORIES = (
    ("耳机", re.compile(r"耳机|耳塞|蓝牙耳机|充电盒")),
    ("电脑", re.compile(r"电脑|笔记本|台式机|macbook|laptop|pc", re.I)),
    ("手机", re.compile(r"手机|智能机|iphone|android", re.I)),
    ("平板", re.compile(r"平板|ipad", re.I)),
    ("相机", re.compile(r"相机|摄像头|camera", re.I)),
    ("清洁设备", re.compile(r"吸尘器|扫地机|洗地机|滚刷")),
)

_SYMPTOM_PATTERNS = (
    ("audio", re.compile(r"没声音|无声|单耳|左耳|右耳|杂音|异响|麦克风|听不到")),
    ("connectivity", re.compile(r"连不上|无法连接|连接失败|断连|蓝牙(?:搜不到|找不到|连接异常)|网络连接异常")),
    ("charging", re.compile(r"无法充电|充不进(?:去)?电|充不上电|充电失败|充电没反应|不充电|掉电|续航")),
    ("power", re.compile(r"开不了机|无法开机|不开机|开机没反应|自动关机|重启")),
    (
        "display",
        re.compile(r"黑屏|花屏|闪屏|屏幕|显示异常|画面(?:全黑|一片黑|显示不出来)|没有画面|无画面"),
    ),
    ("input", re.compile(r"键盘|触摸|按键|鼠标|控制失灵")),
    ("performance", re.compile(r"卡顿|发热|过热|异常发热|变慢|崩溃|闪退")),
)

_SAFETY_RISK_ALIASES = (
    ("鼓包/变形", ("电池鼓包", "鼓包", "外壳变形", "变形")),
    ("冒烟", ("冒烟",)),
    ("起火", ("起火", "着火")),
    ("烧焦味", ("烧焦味", "焦糊味", "糊味")),
    ("进水/漏液", ("进水", "漏液", "漏电池液")),
    ("触电", ("触电", "麻手")),
    ("异常发热", ("异常高温", "异常发热", "明显发烫", "发烫", "烫手")),
    ("人身伤害", ("烫伤", "人身伤害")),
)
_SAFETY_CLEAR = re.compile(
    r"^(?:没有|无|都没有|不存在|不发热|没冒烟|"
    r"没有这些(?:危险|安全)?(?:情况|异常|现象|风险)?|"
    r"上述(?:危险|安全)?(?:情况|异常|现象|风险)?都没有)"
    r"[。！！!\s]*$"
)
_SAFETY_UNCERTAIN = re.compile(r"不确定|不知道|看不出|不清楚")
_SAFETY_YES = re.compile(r"^(?:有|是|是的|有这种情况)[。！！!\s]*$")
_UNKNOWN_DEVICE = re.compile(r"不知道|看不到|不清楚|没有型号|找不到")
_BACK_COMMAND = re.compile(r"^(?:返回上一问题|返回上一步|上一步|修改上一项|重新回答上一题)[。！!\s]*$")


@dataclass(frozen=True)
class ClarificationDecision:
    response: str
    stage: str
    reason: str
    question_group: Optional[Dict[str, Any]] = None


def is_active_support_flow(state: Any) -> bool:
    flow = state.get("support_flow") if isinstance(state, dict) else None
    return bool(
        isinstance(flow, dict)
        and flow.get("type") == FLOW_TYPE
        and flow.get("status") == "active"
        and flow.get("stage") in ACTIVE_STAGES
    )


def detect_product_category(text: str) -> Optional[str]:
    for category, pattern in _PRODUCT_CATEGORIES:
        if pattern.search(text or ""):
            return category
    return None


def detect_symptom_category(text: str) -> Optional[str]:
    for category, pattern in _SYMPTOM_PATTERNS:
        if pattern.search(text or ""):
            return category
    return None


def detect_safety_risks(text: str) -> list[str]:
    """识别用户已明确报告的设备安全信号。

    按句子和转折词分段；逗号和顿号保留在段内，使“没有发烫、鼓包、
    冒烟”的成组否定能覆盖整个列表。转折后的“但有烧焦味”仍会识别。
    """
    segments = re.split(r"[。；;！!？?]|但是|但|不过|然而|可是|却", (text or "").lower())
    all_aliases = tuple(
        sorted(
            {alias for _, aliases in _SAFETY_RISK_ALIASES for alias in aliases},
            key=len,
            reverse=True,
        )
    )
    negative_pattern = re.compile(r"没有|并无|不存在|未出现|没发现|没|无|未|不是|不")
    found: list[str] = []
    for segment in segments:
        if not segment.strip():
            continue
        for label, aliases in _SAFETY_RISK_ALIASES:
            matched = False
            for alias in aliases:
                for occurrence in re.finditer(re.escape(alias), segment):
                    prefix = segment[:occurrence.start()]
                    immediate = prefix[-4:]
                    explicitly_positive = bool(
                        re.search(r"(?:有|出现|发生|感到|闻到|看到)$", immediate)
                        and not re.search(r"(?:没有|并无|无)$", immediate)
                    )
                    negated = bool(re.search(
                        r"(?:没有|没|无|未|不|不是|并无|不存在|未出现|没发现)"
                        r"(?:任何|明显|异常|出现|发生|发现|看到|闻到|感到|这些|这种)?"
                        r"[^\s]{0,8}$",
                        prefix,
                    ))
                    # 处理“没有发烫、鼓包、冒烟”：否定词后若只剩风险词、
                    # 列表符号和连接词，则否定范围持续到当前风险项。
                    negative_matches = list(negative_pattern.finditer(prefix))
                    if negative_matches:
                        tail = prefix[negative_matches[-1].end():]
                        for known_alias in all_aliases:
                            tail = tail.replace(known_alias, "")
                        tail = re.sub(
                            r"[、，,\s:：]*(?:也|都|和|或|以及|并且|与|任一|任何|"
                            r"明显|异常|上述|以下|这些|这种|风险|情况|现象|症状)*",
                            "",
                            tail,
                        )
                        negated = negated or not tail
                    if not negated or explicitly_positive:
                        matched = True
                        break
                if matched:
                    break
            if matched and label not in found:
                found.append(label)
    return found


def mark_support_flow_safety_risk(
    state: Dict[str, Any], *, message: str, request_id: str
) -> list[str]:
    """将可验证的安全信号写入当前任务，并终止普通诊断。"""
    risks = detect_safety_risks(message)
    if not isinstance(state, dict) or not risks:
        return risks
    flow = dict(state.get("support_flow") or {})
    flow.update({
        "type": FLOW_TYPE,
        "status": "blocked",
        "stage": "safety_blocked",
        "safety_status": "risk",
        "risk_signals": risks,
        "updated_request_id": request_id,
    })
    flow.setdefault("started_request_id", request_id)
    flow.setdefault("product_category", detect_product_category(message))
    state["support_flow"] = flow
    for risk in risks:
        existing = state.setdefault("entities", {}).setdefault("fault_symptom", [])
        if risk not in existing:
            existing.append(risk)
    state.setdefault("sources", {})["fault_symptom"] = request_id
    state.setdefault("facts", {})["diagnosis_status"] = "stopped_for_safety"
    refs = state.setdefault("evidence_refs", [])
    if "safety://battery-and-electrical-risk-v1" not in refs:
        refs.append("safety://battery-and-electrical-risk-v1")
    state.setdefault("resolution_plan", {}).update({
        "status": "handoff_required",
        "next_step": "escalate",
        "interaction_stage": "safety_blocked",
        "stop_reason": "检测到设备安全风险",
    })
    return risks


def _first(values: Any) -> Optional[str]:
    if isinstance(values, list):
        return next((str(value).strip() for value in values if str(value).strip()), None)
    return None


def _update_entity(state: Dict[str, Any], key: str, value: str, request_id: str) -> None:
    entities = state.setdefault("entities", {})
    if not entities.get(key):
        entities[key] = [value]
        state.setdefault("sources", {})[key] = request_id


def _set_stage(state: Dict[str, Any], flow: Dict[str, Any], stage: str, request_id: str) -> None:
    flow.update({
        "stage": stage,
        "status": "ready" if stage == "diagnosis_ready" else "active",
        "updated_request_id": request_id,
    })
    state["support_flow"] = flow
    plan = state.setdefault("resolution_plan", {})
    plan["interaction_stage"] = stage
    if stage != "diagnosis_ready":
        plan.update({"status": "collecting_information", "next_step": "collect"})


def _safety_question() -> str:
    return (
        "先确认安全情况：设备是否有发烫、鼓包、冒烟、烧焦味、漏液或触电中的任一情况？\n\n"
        "请选择：有 / 没有 / 不确定，也可以直接描述。"
    )


def _symptom_question(category: Optional[str]) -> str:
    prefix = f"这台{category}" if category else "设备"
    return (
        f"好的，安全情况已确认。{prefix}的具体表现是什么？\n\n"
        "例如：没声音、连不上、无法充电、开不了机、屏幕异常，也可以直接描述。"
    )


def _identity_question(category: Optional[str]) -> str:
    target = category or "设备"
    return f"请告诉我这台{target}的品牌和型号；如果看不到，也可以说明购买时间或设备类型。"


def build_question_group(
    stage: str,
    *,
    request_id: str,
    category: Optional[str] = None,
    uncertain: bool = False,
) -> Dict[str, Any]:
    """返回用户界面可直接渲染的问题组，不包含内部路由信息。"""
    base: Dict[str, Any] = {
        "id": f"{request_id}:{stage}",
        "stage": stage,
        "required": True,
        "allow_text": True,
        "submit_label": "提交",
    }
    if stage == "safety_check":
        base.update({
            "type": "single_select",
            "prompt": "请选择当前安全情况",
            "options": [
                {"id": "risk_yes", "label": "有", "value": "有"},
                {"id": "risk_no", "label": "没有", "value": "没有"},
                {
                    "id": "risk_unknown",
                    "label": "仍不确定" if uncertain else "不确定",
                    "value": "仍不确定" if uncertain else "不确定",
                },
            ],
            "text_placeholder": "也可以直接描述设备的安全情况",
        })
        return base
    if stage == "safety_detail":
        base.update({
            "type": "multi_select",
            "prompt": "请选择出现的安全异常（可多选）",
            "options": [
                {"id": "heat", "label": "明显发烫", "value": "明显发烫"},
                {"id": "swelling", "label": "鼓包/变形", "value": "电池鼓包或外壳变形"},
                {"id": "smoke", "label": "冒烟", "value": "冒烟"},
                {"id": "burning_smell", "label": "烧焦味", "value": "烧焦味"},
                {"id": "leak", "label": "漏液", "value": "漏液"},
                {"id": "shock", "label": "触电/麻手", "value": "触电或麻手"},
            ],
            "text_placeholder": "补充其他安全异常（选填）",
        })
        return base
    if stage == "symptom":
        target = category or "设备"
        base.update({
            "type": "multi_select",
            "prompt": f"请选择这台{target}的故障表现（可多选）",
            "options": [
                {"id": "audio", "label": "没声音", "value": "没声音"},
                {"id": "connection", "label": "连不上", "value": "连不上"},
                {"id": "charging", "label": "无法充电", "value": "无法充电"},
                {"id": "power", "label": "开不了机", "value": "开不了机"},
                {"id": "display", "label": "屏幕异常", "value": "屏幕异常"},
            ],
            "text_placeholder": "也可以直接描述具体表现",
        })
        return base
    if stage == "device_identity":
        target = category or "设备"
        base.update({
            "type": "text",
            "prompt": f"请输入这台{target}的品牌和型号",
            "options": [],
            "text_placeholder": "例如：FlowBuds Pro；不知道时可填写购买时间或设备类型",
        })
        return base
    return {}


def build_missing_field_question_group(
    state: Any, *, request_id: str
) -> Optional[Dict[str, Any]]:
    """将任务计划中的第一个缺失字段转成前端可渲染的单问组件。

    这里不会一次向用户展示全部内部字段；完成当前字段后，
    下一轮再根据新状态生成下一个问题。
    """
    if not isinstance(state, dict) or state.get("domain") != "aftersales":
        return None
    plan = state.get("resolution_plan") if isinstance(state.get("resolution_plan"), dict) else {}
    missing = plan.get("missing_fields") if isinstance(plan.get("missing_fields"), list) else []
    if plan.get("status") != "collecting_information" or not missing:
        return None
    field = str(missing[0])
    copy = {
        "order_id": ("请输入订单号", "例如：RF202610270001"),
        "serial_number": ("请输入设备序列号", "请查看设备本体或包装标签"),
        "product_model": ("请输入产品型号", "例如：FlowBuds Pro"),
        "fault_symptom": ("请描述具体故障现象", "例如：左耳没有声音"),
        "fault_code": ("请输入故障码", "例如：E101"),
        "ticket_id": ("请输入工单号", "例如：RF-T20261020-001"),
    }
    prompt, placeholder = copy.get(field, (f"请补充{field}", "请输入"))
    return {
        "id": f"{request_id}:collect_{field}",
        "stage": f"collect_{field}",
        "type": "text",
        "prompt": prompt,
        "options": [],
        "required": True,
        "allow_text": True,
        "text_placeholder": placeholder,
        "submit_label": "提交",
    }


def _clarification(
    response: str,
    stage: str,
    reason: str,
    *,
    request_id: str,
    category: Optional[str] = None,
    uncertain: bool = False,
) -> ClarificationDecision:
    return ClarificationDecision(
        response=response,
        stage=stage,
        reason=reason,
        question_group=build_question_group(
            stage,
            request_id=request_id,
            category=category,
            uncertain=uncertain,
        ),
    )


def _rewind_support_flow(
    state: Dict[str, Any], flow: Dict[str, Any], *, request_id: str
) -> ClarificationDecision:
    """撤回上一个用户问题组，同时清理其后产生的结构化答案。"""
    current = str(flow.get("stage") or "safety_check")
    category = flow.get("product_category")
    if current == "device_identity":
        flow.pop("symptom_category", None)
        flow.pop("symptom_detail", None)
        state.setdefault("entities", {}).pop("fault_symptom", None)
        state.setdefault("sources", {}).pop("fault_symptom", None)
        target = "symptom"
        response = _symptom_question(category)
        reason = "user_returned_to_symptom"
    elif current == "safety_detail":
        flow["safety_status"] = "unknown"
        flow.pop("risk_signals", None)
        target = "safety_check"
        response = _safety_question()
        reason = "user_returned_to_safety_check"
    elif current == "symptom":
        flow["safety_status"] = "unknown"
        target = "safety_check"
        response = _safety_question()
        reason = "user_returned_to_safety_check"
    else:
        target = "safety_check"
        response = _safety_question()
        reason = "already_at_first_question"
    _set_stage(state, flow, target, request_id)
    return _clarification(
        response, target, reason,
        request_id=request_id, category=category,
    )


def advance_device_support_flow(
    state: Dict[str, Any],
    *,
    message: str,
    intent: Optional[str],
    urgency: Optional[str],
    request_id: str,
) -> Optional[ClarificationDecision]:
    """推进设备故障澄清状态。

    返回 None 表示信息已就绪或不属于本流程，可继续正常路由。
    """
    if not isinstance(state, dict) or urgency == "CRITICAL":
        return None

    text = (message or "").strip()
    flow = dict(state.get("support_flow") or {})
    active = is_active_support_flow(state)
    if active and intent not in {"device_fault", "aftersales", "other", "query", "request", "greeting"}:
        return None
    if not active:
        candidate = intent == "device_fault" or (
            intent in {"aftersales", "other"} and bool(_GENERIC_FAILURE.search(text))
        )
        if not candidate:
            return None

        entities = state.setdefault("entities", {})
        symptom = detect_symptom_category(text) or detect_symptom_category(" ".join(entities.get("fault_symptom", [])))
        product_model = _first(entities.get("product_model"))
        serial_number = _first(entities.get("serial_number"))
        fault_code = _first(entities.get("fault_code"))
        # 型号或序列号都能作为设备身份；故障码本身也是可核验的诊断信号。
        # 信息完整时直接进入工具链，避免只对单个 E101 文案形成特例。
        if (symptom or fault_code) and (product_model or serial_number):
            return None
        flow = {
            "type": FLOW_TYPE,
            "status": "active",
            "stage": "safety_check",
            "safety_status": "unknown",
            "product_category": detect_product_category(text),
            "product_model": product_model,
            "symptom_category": symptom,
            "symptom_detail": _first(entities.get("fault_symptom")),
            "started_request_id": request_id,
        }
        _set_stage(state, flow, "safety_check", request_id)
        return _clarification(
            _safety_question(), "safety_check", "missing_safety_status",
            request_id=request_id, category=flow.get("product_category"),
        )

    category = detect_product_category(text) or flow.get("product_category")
    flow["product_category"] = category
    product_model = _first(state.get("entities", {}).get("product_model")) or flow.get("product_model")
    flow["product_model"] = product_model
    stage = str(flow.get("stage"))

    if _BACK_COMMAND.search(text):
        return _rewind_support_flow(state, flow, request_id=request_id)

    if stage == "safety_check":
        if _SAFETY_CLEAR.search(text):
            flow["safety_status"] = "clear"
            if flow.get("symptom_category"):
                next_stage = "diagnosis_ready" if product_model else "device_identity"
                _set_stage(state, flow, next_stage, request_id)
                if next_stage == "diagnosis_ready":
                    return None
                return _clarification(
                    _identity_question(category), next_stage, "missing_device_identity",
                    request_id=request_id, category=category,
                )
            _set_stage(state, flow, "symptom", request_id)
            return _clarification(
                _symptom_question(category), "symptom", "missing_symptom_category",
                request_id=request_id, category=category,
            )
        if _SAFETY_UNCERTAIN.search(text):
            flow["safety_status"] = "uncertain"
            _set_stage(state, flow, "safety_check", request_id)
            return _clarification(
                "可以先观察设备是否明显发烫、变形、冒烟、有烧焦味、漏液或麻手触电。\n\n"
                "请选择：有 / 没有 / 仍不确定。",
                "safety_check",
                "safety_status_uncertain",
                request_id=request_id,
                category=category,
                uncertain=True,
            )
        if _SAFETY_YES.search(text):
            flow["safety_status"] = "reported"
            _set_stage(state, flow, "safety_detail", request_id)
            return _clarification(
                "请告诉我出现了哪一种情况：明显发烫、鼓包/变形、冒烟、烧焦味、漏液或触电；可以多选或直接描述。",
                "safety_detail",
                "missing_safety_detail",
                request_id=request_id,
                category=category,
            )
        if detect_safety_risks(text):
            # 正常请求会被意图识别标记为 CRITICAL，并由升级路由处理。
            return None
        _set_stage(state, flow, "safety_check", request_id)
        return _clarification(
            _safety_question(), "safety_check", "invalid_safety_answer",
            request_id=request_id, category=category,
        )

    if stage == "safety_detail":
        if detect_safety_risks(text):
            return None
        return _clarification(
            "请描述具体安全异常，例如发烫、鼓包、冒烟、烧焦味、漏液或触电。",
            "safety_detail",
            "missing_safety_detail",
            request_id=request_id,
            category=category,
        )

    if stage == "symptom":
        symptom = detect_symptom_category(text)
        if not symptom:
            _set_stage(state, flow, "symptom", request_id)
            return _clarification(
                _symptom_question(category), "symptom", "missing_symptom_category",
                request_id=request_id, category=category,
            )
        flow.update({"symptom_category": symptom, "symptom_detail": text})
        _update_entity(state, "fault_symptom", text, request_id)
        product_model = _first(state.get("entities", {}).get("product_model")) or flow.get("product_model")
        if product_model:
            flow["product_model"] = product_model
            _set_stage(state, flow, "diagnosis_ready", request_id)
            return None
        _set_stage(state, flow, "device_identity", request_id)
        return _clarification(
            _identity_question(category), "device_identity", "missing_device_identity",
            request_id=request_id, category=category,
        )

    if stage == "device_identity":
        product_model = _first(state.get("entities", {}).get("product_model"))
        symptom = detect_symptom_category(text)
        if not product_model and symptom:
            flow.update({"symptom_category": symptom, "symptom_detail": text})
            _update_entity(state, "fault_symptom", text, request_id)
            _set_stage(state, flow, "device_identity", request_id)
            return _clarification(
                _identity_question(category), "device_identity", "symptom_received_before_identity",
                request_id=request_id, category=category,
            )
        if not product_model and text and not _UNKNOWN_DEVICE.search(text):
            product_model = text[:80]
            _update_entity(state, "product_model", product_model, request_id)
            _update_entity(state, "product", product_model, request_id)
        if product_model:
            flow["product_model"] = product_model
            _set_stage(state, flow, "diagnosis_ready", request_id)
            return None
        _set_stage(state, flow, "device_identity", request_id)
        return _clarification(
            _identity_question(category), "device_identity", "missing_device_identity",
            request_id=request_id, category=category,
        )

    return None
