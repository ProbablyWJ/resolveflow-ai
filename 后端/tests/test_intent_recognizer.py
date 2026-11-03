import asyncio

import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.BILLING, 0.3),       # weighted score 0.21
        (IntentCategory.TECHNICAL_LOGIN, 0.9),  # weighted score 0.18
        (IntentCategory.TECHNICAL, 0.4),     # weighted score 0.04
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "message",
    [
        "订单 RF202510010002 的设备是否过保？",
        "这台耳机还在保修期吗？",
        "怎么查质保截止时间？",
    ],
)
def test_explicit_warranty_question_has_deterministic_business_rule(message):
    sources = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        message,
        IntentCategory.REPAIR_REQUEST,
        0.78,
        sources,
    )

    assert intent is IntentCategory.WARRANTY_INQUIRY
    assert confidence == pytest.approx(0.95)
    assert sources["business_rule"] == pytest.approx(1.0)


def test_warranty_word_inside_explicit_repair_request_does_not_force_override():
    sources = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        "过保了，我现在要申请维修",
        IntentCategory.REPAIR_REQUEST,
        0.88,
        sources,
    )

    assert intent is IntentCategory.REPAIR_REQUEST
    assert confidence == pytest.approx(0.88)
    assert "business_rule" not in sources


@pytest.mark.parametrize(
    ("message", "initial", "expected"),
    [
        ("先别办换货，我只想知道为什么连不上", IntentCategory.BILLING, IntentCategory.DEVICE_FAULT),
        ("不要换新，优先给我维修", IntentCategory.EXCHANGE_REQUEST, IntentCategory.REPAIR_REQUEST),
        ("维修需要提交哪些信息", IntentCategory.TICKET_STATUS, IntentCategory.REPAIR_REQUEST),
        ("我想先了解换货条件，暂时不要创建工单", IntentCategory.PARTS_REQUEST, IntentCategory.EXCHANGE_REQUEST),
        ("这个東西用着不太对，我该选维修还是换货", IntentCategory.REPAIR_REQUEST, IntentCategory.AFTERSALES),
        ("我不知道这算故障还是配件问题", IntentCategory.LOGISTICS, IntentCategory.AFTERSALES),
        ("买的设备有点问题，但我还不确定怎么处理", IntentCategory.DEVICE_FAULT, IntentCategory.AFTERSALES),
    ],
)
def test_explicit_aftersales_goal_negation_and_ambiguity_are_stable(message, initial, expected):
    sources = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        message,
        initial,
        0.72,
        sources,
    )

    assert intent is expected
    assert confidence >= 0.90
    assert sources["business_rule"] == pytest.approx(1.0)


def test_device_safety_risk_overrides_model_and_history_noise():
    sources = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        "FlowBuds Pro 充电盒电池鼓包，还有烧焦味",
        IntentCategory.WARRANTY_INQUIRY,
        0.81,
        sources,
    )

    assert intent is IntentCategory.DEVICE_FAULT
    assert confidence == pytest.approx(0.99)
    assert sources["business_rule"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("我的电脑坏了", IntentCategory.DEVICE_FAULT),
        ("刚拆封就无法开机，可以直接换一台吗", IntentCategory.EXCHANGE_REQUEST),
        ("我的笔记本一直蓝屏怎么办", IntentCategory.TECHNICAL_CRASH),
        ("我不想再跟机器人说了，直接转人工", IntentCategory.HUMAN_HANDOFF),
        ("算了，我不修了", IntentCategory.AFTERSALES),
        ("返回上一问题", IntentCategory.AFTERSALES),
        ("我找不到工单号", IntentCategory.TICKET_STATUS),
        ("耳机好像有点热，我不确定这算不算异常", IntentCategory.DEVICE_FAULT),
        ("FlowBuds Pro 左耳没声音，没有发烫或鼓包", IntentCategory.DEVICE_FAULT),
        ("我只想换一个充电盒，不修整机", IntentCategory.PARTS_REQUEST),
        ("维修工单在哪里看", IntentCategory.TICKET_STATUS),
        ("耳机掉水里了，现在还能继续用吗", IntentCategory.DEVICE_FAULT),
        ("不要转人工，我想先自己排查", IntentCategory.AFTERSALES),
    ],
)
def test_product_high_precision_phrases_have_deterministic_routes(message, expected):
    sources = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        message, IntentCategory.OTHER, 0.0, sources,
    )

    assert intent is expected
    assert confidence >= 0.95
    assert sources["business_rule"] == pytest.approx(1.0)


def test_deterministic_route_skips_remote_llm_call():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._cache = {}
    recognizer._cache_expiries = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("确定性快路由不应调用远端模型")

    recognizer._llm_recognize = forbidden
    recognizer._embedding_recognize = forbidden

    result = asyncio.run(recognizer.recognize("我的电脑坏了"))

    assert result.intent is IntentCategory.DEVICE_FAULT
    assert result.embedding_source == "deterministic-rule"
    assert result.source_scores["business_rule"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("message", "initial", "expected"),
    [
        ("我想申请维修，不考虑换新", IntentCategory.EXCHANGE_REQUEST, IntentCategory.REPAIR_REQUEST),
        ("摄像头无法通电，在哪里报修", IntentCategory.INVOICE, IntentCategory.REPAIR_REQUEST),
        ("我只想了解换机规则，先不要建服务单", IntentCategory.REQUEST, IntentCategory.EXCHANGE_REQUEST),
        ("想查一下配件申请要什么资料", IntentCategory.TECHNICAL_LOGIN, IntentCategory.PARTS_REQUEST),
        ("我不要整机维修，只需要一个滚刷", IntentCategory.REPAIR_REQUEST, IntentCategory.PARTS_REQUEST),
        ("查询一下我的换货申请处理进度", IntentCategory.EXCHANGE_REQUEST, IntentCategory.TICKET_STATUS),
        ("我想查一下维修到哪了", IntentCategory.REPAIR_REQUEST, IntentCategory.TICKET_STATUS),
        ("同一个故障处理三次还没好，我要投诉", IntentCategory.ESCALATION, IntentCategory.QUALITY_COMPLAINT),
        ("不知道该报修还是申请其他服务", IntentCategory.REPAIR_REQUEST, IntentCategory.AFTERSALES),
    ],
)
def test_holdout_exposed_business_boundaries_follow_product_taxonomy(message, initial, expected):
    sources = {}
    intent, confidence = IntentRecognizer._apply_high_precision_business_rules(
        message, initial, 0.72, sources,
    )

    assert intent is expected
    assert confidence >= 0.90
    assert sources["business_rule"] == pytest.approx(1.0)
