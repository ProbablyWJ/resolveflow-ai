"""副作用动作的明确确认判断；否定与疑问必须优先于肯定表达。"""
import re
import unicodedata


_TICKET_ACTION = (
    r"(?:创建|提交|办理)"
    r"(?:这张|该|当前|这)?"
    r"(?:售后|服务|维修|换货|配件)?"
    r"工单"
)
_PENDING_ACTION_REJECTION = re.compile(
    r"(?:算了|取消|撤销|不要|不用|别|停止|不再|先不).{0,8}(?:提交|办理|创建)"
    r"|(?:不要提交|别提交|不提交|取消提交)"
)
_NEGATIVE_BEFORE_ACTION = re.compile(
    rf"(?:不同意|不确认|不允许|不需要|不想|不要|暂不|先不|不用|无需|别|拒绝|取消|撤销|停止|不)"
    rf".{{0,12}}{_TICKET_ACTION}"
)
_NEGATIVE_AFTER_ACTION = re.compile(
    rf"{_TICKET_ACTION}.{{0,12}}(?:取消|撤销|不要|不用|不需要|不同意|不确认|先等等|稍后再说)"
)
_QUESTION = re.compile(
    rf"(?:如何|怎么|怎样|能否|能不能|是否|可不可以).{{0,12}}{_TICKET_ACTION}|"
    rf"{_TICKET_ACTION}.{{0,6}}(?:吗|么|呢|\?|？)"
)
_EXPLICIT_POSITIVE = re.compile(
    rf"(?:我\s*)?(?:确认|同意)[：:，,\s]*(?:立即\s*)?{_TICKET_ACTION}|"
    rf"(?:请|现在|立即)[：:，,\s]*(?:立即\s*)?{_TICKET_ACTION}"
)


def is_explicit_ticket_confirmation(message: str) -> bool:
    """仅接受无否定、非疑问的明确建单指令；无法确定时一律返回 False。"""
    text = unicodedata.normalize("NFKC", str(message or "")).strip()
    if not text:
        return False
    if _NEGATIVE_BEFORE_ACTION.search(text) or _NEGATIVE_AFTER_ACTION.search(text):
        return False
    if _QUESTION.search(text):
        return False
    return bool(_EXPLICIT_POSITIVE.search(text))


def is_explicit_ticket_rejection(message: str) -> bool:
    """识别用户对当前建单动作的明确拒绝；疑问和无关否定不算拒绝。"""
    text = unicodedata.normalize("NFKC", str(message or "")).strip()
    if not text or _QUESTION.search(text):
        return False
    return bool(
        _NEGATIVE_BEFORE_ACTION.search(text)
        or _NEGATIVE_AFTER_ACTION.search(text)
        or _PENDING_ACTION_REJECTION.search(text)
    )
