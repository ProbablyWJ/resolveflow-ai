from monitor.product_analytics import ProductAnalytics


def test_product_analytics_tracks_funnel_without_storing_user_text():
    analytics = ProductAnalytics()
    first = {"id": "req-1:safety_check", "stage": "safety_check"}
    second = {"id": "req-2:symptom", "stage": "symptom"}

    analytics.observe_response(
        user_id="u1", conv_id="c1", question_group=first, escalated=False,
    )
    # 幂等：同一回执或重试不重复计数。
    analytics.observe_response(
        user_id="u1", conv_id="c1", question_group=first, escalated=False,
    )
    analytics.record_event(
        event_type="question_answered", user_id="u1", conv_id="c1",
        question_group_id=first["id"], stage=first["stage"],
    )
    analytics.observe_response(
        user_id="u1", conv_id="c1", question_group=second, escalated=False,
    )
    analytics.record_event(
        event_type="question_back", user_id="u1", conv_id="c1",
        question_group_id=second["id"], stage=second["stage"],
    )
    analytics.record_event(
        event_type="flow_abandoned", user_id="u1", conv_id="c1",
        question_group_id=second["id"], stage=second["stage"],
    )

    summary = analytics.summary()
    assert summary["flows_started"] == 1
    assert summary["questions_presented"] == 2
    assert summary["questions_answered"] == 1
    assert summary["flows_abandoned"] == 1
    assert summary["abandonment_rate"] == 1.0
    assert summary["back_navigations"] == 1
    assert summary["exit_points"] == {"symptom": 1}
    assert "message" not in repr(analytics._sessions)


def test_product_analytics_tracks_completion_and_handoff_rates():
    analytics = ProductAnalytics()
    group = {"id": "req:safety_check", "stage": "safety_check"}
    analytics.observe_response(
        user_id="u1", conv_id="complete", question_group=group, escalated=False,
    )
    analytics.observe_response(
        user_id="u1", conv_id="complete", question_group=None, escalated=False,
    )
    analytics.observe_response(
        user_id="u2", conv_id="handoff", question_group=group, escalated=False,
    )
    analytics.observe_response(
        user_id="u2", conv_id="handoff", question_group=None, escalated=True,
    )

    summary = analytics.summary()
    assert summary["flows_started"] == 2
    assert summary["flows_completed"] == 2
    assert summary["completion_rate"] == 1.0
    assert summary["handoffs"] == 1
    assert summary["handoff_rate"] == 0.5


def test_task_switch_counts_as_abandonment_and_direct_handoff_has_a_denominator():
    analytics = ProductAnalytics()
    group = {"id": "req:symptom", "stage": "symptom"}
    analytics.observe_response(
        user_id="u1", conv_id="switched", question_group=group, escalated=False,
    )
    analytics.observe_response(
        user_id="u1", conv_id="switched", question_group=None, escalated=False,
        task_state={"task_transition": {"type": "new_task"}},
    )
    analytics.observe_response(
        user_id="u2", conv_id="direct-risk", question_group=None, escalated=True,
    )

    summary = analytics.summary()
    assert summary["conversations_observed"] == 2
    assert summary["flows_completed"] == 0
    assert summary["flows_abandoned"] == 1
    assert summary["exit_points"] == {"symptom": 1}
    assert summary["handoff_rate"] == 0.5
