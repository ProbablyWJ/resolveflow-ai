"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import math
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from core.support_flow import detect_safety_risks

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    ORDER_STATUS = "order_status"        # 订单状态
    LOGISTICS = "logistics"              # 物流配送
    REFUND = "refund"                    # 退款/退货
    INVOICE = "invoice"                  # 发票
    PAYMENT_ISSUE = "payment_issue"      # 支付/扣款异常
    ACCOUNT_SECURITY = "account_security" # 账户安全
    TECHNICAL_LOGIN = "technical_login"  # 登录认证故障
    TECHNICAL_CRASH = "technical_crash"  # 崩溃/错误码
    HUMAN_HANDOFF = "human_handoff"      # 转人工
    AFTERSALES = "aftersales"            # 售后问题大类
    DEVICE_FAULT = "device_fault"        # 设备故障诊断
    WARRANTY_INQUIRY = "warranty_inquiry" # 质保/保修查询
    REPAIR_REQUEST = "repair_request"    # 申请维修/寄修
    EXCHANGE_REQUEST = "exchange_request" # 申请换货/换新
    PARTS_REQUEST = "parts_request"      # 配件补发/申请
    TICKET_STATUS = "ticket_status"      # 售后工单创建/查询
    QUALITY_COMPLAINT = "quality_complaint" # 质量投诉
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    embedding_source: str = "local-ngram"


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.ORDER_STATUS: ["我的订单现在是什么状态？", "订单有没有发货？", "订单处理到哪一步了？"],
    IntentCategory.LOGISTICS: ["快递什么时候到？", "物流一直不更新", "配送要多久？"],
    IntentCategory.REFUND: ["我要申请退款", "退货退款怎么处理？", "退款多久到账？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么重复扣款？", "支付失败怎么办？", "这个月多扣了钱"],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请升级处理"],
    IntentCategory.AFTERSALES: ["我的设备有售后问题", "商品坏了怎么处理", "需要售后帮助"],
    IntentCategory.DEVICE_FAULT: ["耳机左边没有声音", "设备无法充电", "蓝牙一直连接失败", "电池鼓包了"],
    IntentCategory.WARRANTY_INQUIRY: ["这个设备还在保修期吗", "怎么查询质保", "保修包含电池问题吗"],
    IntentCategory.REPAIR_REQUEST: ["我想申请维修", "设备怎么寄修", "帮我报修"],
    IntentCategory.EXCHANGE_REQUEST: ["我想申请换货", "能不能给我换新", "质量问题怎么换机"],
    IntentCategory.PARTS_REQUEST: ["耳帽丢了能补发吗", "我需要新的充电盒", "怎么申请配件"],
    IntentCategory.TICKET_STATUS: ["查询售后工单进度", "我的维修单到哪一步了", "确认创建售后工单"],
    IntentCategory.QUALITY_COMPLAINT: ["这个产品反复坏我要投诉", "维修两次还是没解决", "我要投诉这个质量问题"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
    IntentCategory.DEVICE_FAULT,
    IntentCategory.WARRANTY_INQUIRY,
    IntentCategory.REPAIR_REQUEST,
    IntentCategory.EXCHANGE_REQUEST,
    IntentCategory.PARTS_REQUEST,
    IntentCategory.TICKET_STATUS,
    IntentCategory.QUALITY_COMPLAINT,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
    IntentCategory.ESCALATION,
    IntentCategory.AFTERSALES,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
    IntentCategory.DEVICE_FAULT: IntentCategory.AFTERSALES,
    IntentCategory.WARRANTY_INQUIRY: IntentCategory.AFTERSALES,
    IntentCategory.REPAIR_REQUEST: IntentCategory.AFTERSALES,
    IntentCategory.EXCHANGE_REQUEST: IntentCategory.AFTERSALES,
    IntentCategory.PARTS_REQUEST: IntentCategory.AFTERSALES,
    IntentCategory.TICKET_STATUS: IntentCategory.AFTERSALES,
    IntentCategory.QUALITY_COMPLAINT: IntentCategory.AFTERSALES,
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        from core.intent_embedding import IntentEmbedding
        self._embedding = IntentEmbedding()
        self._embedding_enabled = True
        self._cache: Dict[str, IntentResult] = {}
        self._cache_expiries: Dict[str, float] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache and time.monotonic() < self._cache_expiries.get(key, 0):
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # 高精度、可解释的业务表达无需等待远端模型。先走确定性快路由，
        # 只有开放语义再进入 LLM + Embedding + Pattern 三路融合。
        fast_scores: Dict[str, float] = {}
        if not history and re.fullmatch(
            r"(?:这个|那个|它|这件事).{0,6}(?:怎么办|怎么弄|怎么处理|要怎么搞)[啊呀吗呢？?。！!\s]*",
            (message or "").strip(),
        ):
            fast_intent, fast_confidence = IntentCategory.OTHER, 0.20
            fast_scores["business_rule"] = 1.0
        else:
            fast_intent, fast_confidence = self._apply_high_precision_business_rules(
                message, IntentCategory.OTHER, 0.0, fast_scores
            )
        if fast_scores.get("business_rule") == 1.0:
            entities = self._extract_entities(message)
            result = IntentResult(
                intent=fast_intent,
                confidence=fast_confidence,
                urgency=self._urgency(message, fast_intent),
                intent_group=self._intent_group(fast_intent),
                entities=entities,
                reasoning="deterministic_high_precision_rule",
                latency_ms=(time.monotonic() - t0) * 1000,
                source_scores=fast_scores,
                embedding_source="deterministic-rule",
            )
            self._cache[key] = result
            self._cache_expiries[key] = time.monotonic() + 300
            return result

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        intent, confidence = self._apply_high_precision_business_rules(
            message, intent, confidence, source_scores
        )
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
            embedding_source=emb.get("source", "local-ngram"),
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
                self._cache_expiries.pop(k, None)
        self._cache[key] = result
        self._cache_expiries[key] = time.monotonic() + (60 if result.embedding_source == "local-ngram" and self._embedding.enabled else 300)
        return result

    @staticmethod
    def _apply_high_precision_business_rules(
        message: str,
        intent: IntentCategory,
        confidence: float,
        source_scores: Dict[str, float],
    ) -> tuple[IntentCategory, float]:
        """对少量高精度业务表达做确定性校正。

        规则只处理可以直接从原句确定的目标、否定和二选一表达；
        开放语义仍交给 LLM / Embedding / Pattern 三路融合。
        """
        text = (message or "").lower()
        if detect_safety_risks(text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.DEVICE_FAULT, max(confidence, 0.99)

        if re.fullmatch(
            r"(?:先)?(?:算了|不用了|不弄了|不处理了|不修了|先不(?:修|弄|处理)了|到此为止)"
            r"(?:[，,。！!\s]*(?:我)?(?:不修了|不弄了|不处理了|不用处理了))?[。！!\s]*",
            text,
        ):
            source_scores["business_rule"] = 1.0
            return IntentCategory.AFTERSALES, max(confidence, 0.99)

        requests_human = bool(re.search(r"(?:直接|马上|现在)?(?:转|转接|找)(?:一下)?人工|人工客服|真人客服|找真人", text))
        rejects_human = bool(re.search(r"(?:不要|不用|先别|无需).{0,5}(?:转|转接|找)?人工", text))
        if rejects_human:
            source_scores["business_rule"] = 1.0
            return IntentCategory.AFTERSALES, max(confidence, 0.97)
        if requests_human and not rejects_human:
            source_scores["business_rule"] = 1.0
            return IntentCategory.HUMAN_HANDOFF, max(confidence, 0.99)

        # 明确的业务目标优先于弱关键词。这些表达不需要模型猜测，
        # 且能在 LLM 超时时稳定兜底。
        if re.fullmatch(r"(?:你好[，,。！!\s]*)?(?:在吗|有人在吗|客服在吗)[？?。！!\s]*", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.GREETING, max(confidence, 0.95)

        if re.fullmatch(r"(?:返回上一问题|返回上一步|上一步|修改上一项|重新回答上一题)[。！!\s]*", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.AFTERSALES, max(confidence, 0.99)

        if re.search(r"(?:找不到|没有|不知道|忘了).{0,6}(?:工单号|售后单号|维修单号)", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.TICKET_STATUS, max(confidence, 0.98)

        if re.search(r"蓝屏|系统崩溃|应用崩溃|软件崩溃|反复闪退|一直闪退", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.TECHNICAL_CRASH, max(confidence, 0.97)

        explicit_part_request = bool(
            re.search(
                r"(?:换|更换|补|补发|买|申请|需要|想要|只想).{0,10}"
                r"(?:充电盒|耳帽|电源线|数据线|充电线|滚刷|过滤网|底座|配件)",
                text,
            )
            or re.search(
                r"(?:充电盒|耳帽|电源线|数据线|充电线|滚刷|过滤网|底座|配件).{0,8}"
                r"(?:换|更换|补|补发|申请|需要)",
                text,
            )
        )
        part_request_negated = bool(re.search(
            r"(?:不是|不要|不想|不用|先别).{0,8}(?:补|换|更换|申请)?(?:一个)?配件",
            text,
        ))
        part_warranty_question = bool(re.search(
            r"(?:配件|充电盒|耳帽|电源线|数据线|滚刷|过滤网|底座).{0,12}"
            r"(?:保多久|保多长|质保多久|保修多久|保修期|质保期)",
            text,
        ))
        if explicit_part_request and not part_request_negated and not part_warranty_question:
            source_scores["business_rule"] = 1.0
            return IntentCategory.PARTS_REQUEST, max(confidence, 0.97)

        direct_exchange = bool(re.search(r"(?:直接|能否|可以|想|要)?(?:换|更换)(?:一台|一部|一副|一个)", text))
        exchange_denied = bool(re.search(r"(?:不要|不想|不用|先别|不考虑).{0,8}(?:换|更换)(?:一台|一部|一副|一个)", text))
        if direct_exchange and not exchange_denied:
            source_scores["business_rule"] = 1.0
            return IntentCategory.EXCHANGE_REQUEST, max(confidence, 0.97)

        product_mentioned = bool(re.search(
            r"耳机|耳塞|左耳|右耳|充电盒|电脑|笔记本|手机|平板|相机|摄像头|吸尘器|扫地机|洗地机|"
            r"flowbuds|flowcam|flowclean|设备|产品|东西|東西",
            text,
        ))
        device_problem = bool(re.search(
            r"坏了|坏掉|不能用|用不了|没声音|无声|连不上|无法连接|充不上电|无法充电|"
            r"开不了机|无法开机|黑屏|花屏|失灵|故障|有点热|发热|发烫|很热",
            text,
        ))
        explicit_service_goal = bool(re.search(
            r"维修|报修|送修|寄修|返修|换货|换新|换机|置换|配件|补发",
            text,
        ))
        if product_mentioned and device_problem and not explicit_service_goal:
            source_scores["business_rule"] = 1.0
            return IntentCategory.DEVICE_FAULT, max(confidence, 0.97)

        quality_feedback = bool(
            re.search(r"质量.{0,8}(?:反馈|投诉)|(?:反馈|投诉).{0,8}质量", text)
            or re.search(
                r"(?:重复|反复|多次|一直).{0,30}"
                r"(?:没解决|不解决|仍复发|还是复发).{0,16}(?:反馈|投诉)",
                text,
            )
        )
        if quality_feedback:
            source_scores["business_rule"] = 1.0
            return IntentCategory.QUALITY_COMPLAINT, max(confidence, 0.95)

        exchange_eligibility = bool(
            re.search(r"(?:换货|换新|换机|整机更换|置换).{0,10}(?:资格|条件|政策|能否|是否可以)", text)
            or re.search(r"(?:资格|条件|政策).{0,10}(?:换货|换新|换机|置换)", text)
        )
        if exchange_eligibility:
            source_scores["business_rule"] = 1.0
            return IntentCategory.EXCHANGE_REQUEST, max(confidence, 0.95)

        ticket_location = bool(
            re.search(r"(?:工单|售后单|维修单|服务单).{0,10}(?:在哪|哪里|怎么看|查看|查找|查询|找到)", text)
            or re.search(r"(?:在哪|哪里|怎么看|查看|查找|查询).{0,10}(?:工单|售后单|维修单|服务单)", text)
        )
        if ticket_location:
            source_scores["business_rule"] = 1.0
            return IntentCategory.TICKET_STATUS, max(confidence, 0.97)

        explicit_repair_service = bool(
            re.search(r"(?:付费|上门|预约|官方).{0,12}(?:维修|检修|修理|寄修|返修|送修)", text)
            or re.search(r"(?:维修|检修|修理|寄修|返修|送修).{0,12}(?:付费|上门|预约|地址|步骤|流程|在哪|哪里|官方)", text)
        )
        if explicit_repair_service:
            source_scores["business_rule"] = 1.0
            return IntentCategory.REPAIR_REQUEST, max(confidence, 0.95)

        specific_warranty_duration = re.search(
            r"(?:配件|电池|整机|主机|设备).{0,12}(?:保多久|保多长|质保多久|保修多久)",
            text,
        )
        if specific_warranty_duration:
            source_scores["business_rule"] = 1.0
            return IntentCategory.WARRANTY_INQUIRY, max(confidence, 0.95)

        # 明确售后工单号 + 进度语义是高精度业务键；当 LLM 超时时
        # 也应稳定落到查工单，而不是被“没更新”误判为设备故障。
        progress_language = re.search(r"进度|状态|更新|到哪|哪一步|查询|查下|看下|处理到", text)
        ticket_tracking = progress_language and (
            re.search(r"\bRF-T\d{8}-\d{3,6}\b", text, re.I)
            or re.search(r"工单|售后单|维修单|服务单", text)
        )
        if ticket_tracking:
            source_scores["business_rule"] = 1.0
            return IntentCategory.TICKET_STATUS, max(confidence, 0.99)

        service_progress = re.search(r"进展|进度|进行到|处理到|到哪(?:了|儿)?|哪个节点|哪一步", text)
        if service_progress and re.search(r"维修|换货|换机|售后|申请|流程", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.TICKET_STATUS, max(confidence, 0.95)

        quality_complaint = (
            "投诉" in text
            and bool(re.search(r"质量|品控|产品|故障|维修|反复|多次|没解决|没好", text))
        ) or bool(re.search(r"(?:反复|多次).{0,12}(?:故障|坏|维修|售后).{0,12}(?:没解决|没好|不满意)", text))
        if quality_complaint:
            source_scores["business_rule"] = 1.0
            return IntentCategory.QUALITY_COMPLAINT, max(confidence, 0.95)

        warranty_query = re.search(
            r"(是否|还|有没有|查询|怎么查|什么时候|几号).{0,8}(过保|在保|保修|质保)"
            r"|(过保|在保|保修|质保).{0,8}(吗|么|到期|截止|查询)",
            text,
        )
        if warranty_query:
            source_scores["business_rule"] = 1.0
            return IntentCategory.WARRANTY_INQUIRY, max(confidence, 0.95)

        repair_terms = ("维修", "报修", "寄修", "返修", "送修", "检修", "修理")
        exchange_terms = ("换货", "换新", "换机", "更换设备", "整机更换", "置换")
        parts_terms = ("配件", "补发", "耳帽", "电源线", "数据线", "滚刷", "过滤网", "底座", "充电盒")
        fault_terms = ("连不上", "连接失败", "无声", "没声音", "无法充电", "故障")
        has_repair = any(term in text for term in repair_terms)
        has_exchange = any(term in text for term in exchange_terms)
        has_parts = any(term in text for term in parts_terms)
        has_fault = any(term in text for term in fault_terms)
        explicit_ambiguity = ("还是" in text or "不知道这算" in text or "不确定怎么处理" in text)
        category_count = sum((has_repair, has_exchange, has_parts, has_fault))
        if explicit_ambiguity and category_count >= 2:
            source_scores["business_rule"] = 1.0
            return IntentCategory.AFTERSALES, max(confidence, 0.90)

        # 否定范围不跨越逗号/句号，避免把“不要维修，我要换货”
        # 错解为“不要换货”。
        exchange_negated = bool(re.search(
            r"(?:不是|不想|不要|先别|暂不|不考虑)"
            r"[^，,。；;！!？?]{0,8}(?:办|申请|做)?(?:换货|换新|换机|置换)",
            text,
        ))
        repair_negated = bool(re.search(
            r"(?:不是|不想|不要|先别|暂不|不考虑)"
            r"[^，,。；;！!？?]{0,8}(?:办|申请|做)?(?:维修|报修|寄修|整机维修)",
            text,
        ))
        parts_negated = bool(re.search(
            r"(?:不是|不要|先别|暂不|只不)"
            r"[^，,。；;！!？?]{0,10}(?:补|申请|更换)?(?:配件|耳帽|滚刷|过滤网|底座|数据线|充电盒)",
            text,
        ))
        if has_fault and exchange_negated:
            source_scores["business_rule"] = 1.0
            return IntentCategory.DEVICE_FAULT, max(confidence, 0.95)

        if has_repair and exchange_negated:
            source_scores["business_rule"] = 1.0
            return IntentCategory.REPAIR_REQUEST, max(confidence, 0.95)

        if has_repair and not repair_negated and re.search(r"(?:维修|报修|寄修|返修).{0,10}(?:需要|提交|信息|材料|流程|怎么)", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.REPAIR_REQUEST, max(confidence, 0.95)

        if has_repair and not repair_negated and re.search(r"(?:哪里|在哪|如何|怎么).{0,10}(?:维修|报修|寄修|返修|送修)", text):
            source_scores["business_rule"] = 1.0
            return IntentCategory.REPAIR_REQUEST, max(confidence, 0.95)

        if has_parts and not parts_negated and (
            repair_negated
            or re.search(r"(?:申请|补|配|换|需要|只要|只需要|想要|入口|资料|流程).{0,12}(?:配件|耳帽|滚刷|过滤网|底座|数据线|充电盒)", text)
            or re.search(r"(?:配件|耳帽|滚刷|过滤网|底座|数据线|充电盒).{0,12}(?:申请|补|配|换|需要|入口|资料|流程|吗)", text)
        ):
            source_scores["business_rule"] = 1.0
            return IntentCategory.PARTS_REQUEST, max(confidence, 0.95)

        if (intent != IntentCategory.REPAIR_REQUEST and has_repair
                and re.search(r"(?:我要|我想|想要|请|申请|帮我).{0,8}(?:维修|报修|寄修|返修|送修)", text)):
            source_scores["business_rule"] = 1.0
            return IntentCategory.REPAIR_REQUEST, max(confidence, 0.95)

        if has_exchange and not exchange_negated and not (has_repair and explicit_ambiguity):
            source_scores["business_rule"] = 1.0
            return IntentCategory.EXCHANGE_REQUEST, max(confidence, 0.95)

        broad_aftersales = re.search(
            r"(?:设备|商品|东西|東西).{0,10}(?:有点问题|不太对|怎么处理)",
            text,
        )
        post_purchase_question = re.search(
            r"(?:设备|商品|产品).{0,12}(?:使用后|购买后|售后).{0,12}(?:事项|问题|想问|咨询)",
            text,
        )
        ambiguous_service = re.search(
            r"(?:不知道|不确定).{0,12}(?:报修|维修|换货|配件).{0,12}(?:还是|或|或者|其他)",
            text,
        )
        if broad_aftersales or post_purchase_question or ambiguous_service:
            source_scores["business_rule"] = 1.0
            return IntentCategory.AFTERSALES, max(confidence, 0.90)
        return intent, confidence

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._embedding.remote_templates.clear()
            self._embedding.local_templates.clear()
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            self._cache_expiries.clear()
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。

示例:
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await asyncio.wait_for(self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            ), timeout=max(0.1, float(os.getenv("RESOLVEFLOW_INTENT_TIMEOUT_S", "8"))))
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            confidence = data.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("意图 confidence 必须是 0 到 1 的有限数字")
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            templates, msg_vec, source = await self._embedding.vectors(
                self._clean_text(message), _TEMPLATES, self._local_embedding)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in templates.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": min(1.0, max(0.0, best_score)), "source": source}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工"],
            IntentCategory.TICKET_STATUS: ["工单进度", "维修单", "售后单", "服务单", "创建工单", "提交工单", "确认创建"],
            IntentCategory.EXCHANGE_REQUEST: ["换货", "换新", "换一台", "更换设备"],
            IntentCategory.REPAIR_REQUEST: ["维修", "寄修", "报修", "返修", "送修"],
            IntentCategory.WARRANTY_INQUIRY: ["保修", "质保", "在保", "过保"],
            IntentCategory.PARTS_REQUEST: ["配件", "补发", "充电盒", "耳帽", "电源线"],
            IntentCategory.QUALITY_COMPLAINT: ["质量投诉", "反复坏", "多次维修", "投诉质量"],
            IntentCategory.DEVICE_FAULT: ["没声音", "无声", "无法充电", "充不进电", "连不上", "连接失败", "断连", "续航", "故障", "坏了", "异常发热", "错误码", "鼓包", "冒烟", "起火", "漏液", "触电", "烧焦味"],
            IntentCategory.ORDER_STATUS: ["订单状态", "发货了吗", "处理到哪", "order status"],
            IntentCategory.LOGISTICS: ["物流", "快递", "配送", "运单", "delivery", "shipping"],
            IntentCategory.REFUND: ["退款", "退货", "refund", "return"],
            IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
            IntentCategory.PAYMENT_ISSUE: ["重复扣款", "多扣", "支付失败", "扣费", "payment failed"],
            IntentCategory.ACCOUNT_SECURITY: ["被盗", "异常登录", "重置密码", "两步验证", "安全"],
            IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "401", "验证码"],
            IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "500", "报错", "crash"],
        }
        generic_patterns = {
            IntentCategory.AFTERSALES: ["售后", "设备问题", "商品问题"],
            IntentCategory.ESCALATION: ["投诉", "经理", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        cleaned = []
        for source in (llm, emb, pat):
            value = source.get("confidence")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                source = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
            cleaned.append(source)
        llm, emb, pat = cleaned
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        group_scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            weighted_score = w * conf
            scores[cat] = scores.get(cat, 0.0) + weighted_score
            group = _INTENT_GROUPS.get(cat, cat)
            group_scores[group] = group_scores.get(group, 0.0) + weighted_score

        best_group = max(group_scores, key=group_scores.get)
        best_score = group_scores[best_group]
        best = max(
            (cat for cat in scores if _INTENT_GROUPS.get(cat, cat) == best_group),
            key=scores.get,
        )
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if (best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS
                and _INTENT_GROUPS.get(pat_intent, pat_intent) == best_group
                and pat_conf >= 0.5 and best_score < 0.8):
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        dates = self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message))
        order_ids = self._unique(
            re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)
            + re.findall(r"\b(RF\d{8,20})\b", message, re.I)
        )
        product_models = self._unique(re.findall(
            r"\b(FlowBuds\s*Pro|FlowCam\s*Mini|FlowClean\s*S1)\b", message, re.I
        ))
        serial_numbers = self._unique(
            re.findall(r"(?:序列号|设备编号|SN)\s*[:：#]?\s*([A-Za-z0-9-]{6,32})", message, re.I)
            + re.findall(r"\b(FBP-[A-Za-z0-9-]{5,20}|FCM-[A-Za-z0-9-]{5,20}|FCS-[A-Za-z0-9-]{5,20})\b", message, re.I)
        )
        fault_codes = self._unique(re.findall(r"(?:故障码|错误码|code)?\s*[:：#]?\s*\b(E\d{3})\b", message, re.I))
        # 普通故障现象可直接按规则提取；安全风险必须经过统一的否定语义
        # 检测，避免把“没有冒烟”保存成用户已报告的正向风险事实。
        symptoms = self._unique(
            re.findall(
                r"(左耳无声|右耳无声|左耳没声音|右耳没声音|单侧无声|没有声音|没声音|无法充电|充不进电|连接失败|连不上|频繁断连|续航异常)",
                message,
            )
            + detect_safety_risks(message)
        )
        ticket_ids = self._unique(
            re.findall(r"(?:工单号?|服务单号?)\s*[:：#]?\s*([A-Za-z0-9-]{6,32})", message, re.I)
            + re.findall(r"\b(RF-T\d{8}-\d{3,6})\b", message, re.I)
        )
        evidence = self._unique(re.findall(r"(购买凭证|订单截图|故障视频|故障照片|设备照片|序列号照片)", message))
        return {
            "order_id": order_ids,
            "product": product_models,
            "product_model": product_models,
            "serial_number": serial_numbers,
            "purchase_date": dates,
            "fault_code": fault_codes,
            "fault_symptom": symptoms,
            "evidence": evidence,
            "ticket_id": ticket_ids,
            "date": dates,
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, re.I)
                + re.findall(r"\b([45]\d{2})\b", message)
            ),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        msg = re.sub(r"(?:不(?:是|用|需要)?|无需|无须|非|并不|没那么|暂不)\s*(?:紧急|立刻|马上|尽快|快点)", "", msg)
        msg = re.sub(r"\b(?:not|isn't|is not|no)\s+(?:an?\s+)?(?:emergency|urgent|asap|now)\b", "", msg)
        if detect_safety_risks(msg):
            return UrgencyLevel.CRITICAL
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", msg) if kw.isascii() else kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
