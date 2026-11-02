from types import SimpleNamespace

import pytest

from api import main


def test_demo_blocks_admin_and_expensive_surfaces():
    assert main._demo_blocked("/eval/run", "POST")
    assert main._demo_blocked("/knowledge/upload", "POST")
    assert main._demo_blocked("/trace/recent", "GET")
    assert main._demo_blocked("/metrics", "GET")
    assert not main._demo_blocked("/chat", "POST")
    assert not main._demo_blocked("/health", "GET")


@pytest.mark.asyncio
async def test_demo_chat_rate_limit_is_per_client(monkeypatch):
    monkeypatch.setenv("RESOLVEFLOW_DEMO_CHAT_LIMIT", "2")
    monkeypatch.setenv("RESOLVEFLOW_DEMO_GLOBAL_LIMIT", "10")
    monkeypatch.setenv("RESOLVEFLOW_DEMO_WINDOW_S", "3600")
    main._demo_rate_hits.clear()
    main._demo_global_hits = []
    request = SimpleNamespace(
        headers={"x-forwarded-for": "203.0.113.8, 10.0.0.1"},
        client=SimpleNamespace(host="10.0.0.1"),
    )

    assert not await main._demo_rate_limited(request)
    assert not await main._demo_rate_limited(request)
    assert await main._demo_rate_limited(request)

    other = SimpleNamespace(headers={}, client=SimpleNamespace(host="203.0.113.9"))
    assert not await main._demo_rate_limited(other)
