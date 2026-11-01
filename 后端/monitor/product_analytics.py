"""售后澄清流的轻量产品指标。

只记录事件类型、流程阶段和匿名会话键，不记录用户原文。
当前为单进程演示实现；生产环境可将同样的事件契约接入数仓。
"""
from __future__ import annotations

import hashlib
import threading
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ProductAnalytics:
    MAX_SESSIONS = 10_000

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._counters: Counter[str] = Counter()
        self._exit_points: Counter[str] = Counter()

    @staticmethod
    def _key(user_id: str, conv_id: str) -> str:
        raw = f"{user_id}:{conv_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _session(self, user_id: str, conv_id: str) -> Dict[str, Any]:
        key = self._key(user_id, conv_id)
        session = self._sessions.get(key)
        if session is None:
            session = {
                "started": False,
                "ended": False,
                "handoff": False,
                "presented": set(),
                "answered": set(),
                "back_count": 0,
                "last_stage": "unknown",
            }
            self._sessions[key] = session
            self._counters["conversations_observed"] += 1
            if len(self._sessions) > self.MAX_SESSIONS:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(key)
        return session

    def observe_response(
        self,
        *,
        user_id: str,
        conv_id: str,
        question_group: Optional[Dict[str, Any]],
        escalated: bool,
        task_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            session = self._session(user_id, conv_id)
            if question_group:
                if not session["started"]:
                    session["started"] = True
                    self._counters["flows_started"] += 1
                group_id = str(question_group.get("id") or "")
                stage = str(question_group.get("stage") or "unknown")
                session["last_stage"] = stage
                if group_id and group_id not in session["presented"]:
                    session["presented"].add(group_id)
                    self._counters["questions_presented"] += 1
            elif session["started"] and not session["ended"]:
                session["ended"] = True
                transition = (task_state or {}).get("task_transition", {})
                if transition.get("type") == "new_task":
                    self._counters["flows_abandoned"] += 1
                    self._exit_points[session["last_stage"] or "unknown"] += 1
                else:
                    self._counters["flows_completed"] += 1

            if escalated and not session["handoff"]:
                session["handoff"] = True
                self._counters["handoffs"] += 1

    def record_event(
        self,
        *,
        event_type: str,
        user_id: str,
        conv_id: str,
        question_group_id: str = "",
        stage: str = "",
    ) -> None:
        with self._lock:
            session = self._session(user_id, conv_id)
            if stage:
                session["last_stage"] = stage
            if event_type == "question_answered":
                if question_group_id and question_group_id not in session["answered"]:
                    session["answered"].add(question_group_id)
                    self._counters["questions_answered"] += 1
            elif event_type == "question_back":
                session["back_count"] += 1
                self._counters["back_navigations"] += 1
            elif event_type == "flow_abandoned" and session["started"] and not session["ended"]:
                session["ended"] = True
                self._counters["flows_abandoned"] += 1
                self._exit_points[session["last_stage"] or "unknown"] += 1

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            started = int(self._counters["flows_started"])
            completed = int(self._counters["flows_completed"])
            abandoned = int(self._counters["flows_abandoned"])
            answered = int(self._counters["questions_answered"])
            handoffs = int(self._counters["handoffs"])
            conversations = int(self._counters["conversations_observed"])
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "storage_scope": "process_memory_demo",
                "flows_started": started,
                "flows_completed": completed,
                "flows_abandoned": abandoned,
                "conversations_observed": conversations,
                "completion_rate": round(completed / started, 4) if started else 0.0,
                "abandonment_rate": round(abandoned / started, 4) if started else 0.0,
                "handoffs": handoffs,
                "handoff_rate": round(handoffs / conversations, 4) if conversations else 0.0,
                "questions_presented": int(self._counters["questions_presented"]),
                "questions_answered": answered,
                "avg_clarification_rounds": round(answered / started, 2) if started else 0.0,
                "back_navigations": int(self._counters["back_navigations"]),
                "exit_points": dict(self._exit_points.most_common()),
            }
