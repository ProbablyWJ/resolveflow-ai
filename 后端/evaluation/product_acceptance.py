"""ResolveFlow 用户视角产品验收。

这套检查直接请求 /chat，验证用户目标、问题组件、路由、工具、状态和
用户可见话术。它不替代意图评测或 LLM-as-Judge，也不把通过率当线上指标。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


GLOBAL_FORBIDDEN_TERMS = (
    "当前判断",
    "实际依据",
    "routing_reason",
    "intent=",
    "product_model",
    "基础信息仍待核验",
)


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return exc.code, parsed


def _one_of(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return value in expected
    return value == expected


def _check(response: dict[str, Any], expected: dict[str, Any], http_status: int) -> dict[str, bool]:
    text = str(response.get("response") or "")
    group = response.get("question_group") if isinstance(response.get("question_group"), dict) else None
    tools = set(response.get("tools_used") or [])
    checks: dict[str, bool] = {
        "http_200": http_status == 200,
        "status_success": response.get("status") == "success",
        "no_internal_terms": not any(term in text for term in GLOBAL_FORBIDDEN_TERMS),
    }
    if "intent" in expected:
        checks["intent"] = _one_of(response.get("intent"), expected["intent"])
    if "primary_agent" in expected:
        checks["primary_agent"] = _one_of(response.get("primary_agent"), expected["primary_agent"])
    if "escalated" in expected:
        checks["escalated"] = bool(response.get("escalated")) is bool(expected["escalated"])
    if "question_present" in expected:
        checks["question_present"] = bool(group) is bool(expected["question_present"])
    if "question_stage" in expected:
        checks["question_stage"] = bool(group) and _one_of(group.get("stage"), expected["question_stage"])
    if "question_type" in expected:
        checks["question_type"] = bool(group) and _one_of(group.get("type"), expected["question_type"])
    if expected.get("required_response_terms"):
        checks["required_response_terms"] = all(
            term in text for term in expected["required_response_terms"]
        )
    if expected.get("required_any_response_terms"):
        checks["required_any_response_terms"] = any(
            term in text for term in expected["required_any_response_terms"]
        )
    if expected.get("forbidden_response_terms"):
        checks["forbidden_response_terms"] = not any(
            term in text for term in expected["forbidden_response_terms"]
        )
    if expected.get("required_tools"):
        checks["required_tools"] = set(expected["required_tools"]).issubset(tools)
    if expected.get("forbidden_tools"):
        checks["forbidden_tools"] = not bool(set(expected["forbidden_tools"]) & tools)
    if expected.get("max_response_chars") is not None:
        checks["response_length"] = len(text) <= int(expected["max_response_chars"])
    if expected.get("max_latency_ms") is not None:
        checks["latency"] = float(response.get("latency_ms") or 0) <= float(expected["max_latency_ms"])
    return checks


def run_suite(dataset: pathlib.Path, base_url: str, timeout: float) -> dict[str, Any]:
    data = json.loads(dataset.read_text(encoding="utf-8"))
    run_id = uuid.uuid4().hex[:8]
    results = []
    for case in data["cases"]:
        case_id = str(case["case_id"])
        conv_id = f"product-acceptance-{run_id}-{case_id}"
        user_id = f"product-acceptance-{run_id}-{case_id}"
        turn_results = []
        for index, turn in enumerate(case["turns"], 1):
            payload = {
                "message": turn["message"],
                "client_request_id": f"{run_id}-{case_id}-{index}",
                "user_id": user_id,
                "conv_id": conv_id,
            }
            started = time.monotonic()
            try:
                http_status, response = _post(f"{base_url.rstrip('/')}/chat", payload, timeout)
                wall_latency_ms = round((time.monotonic() - started) * 1000, 1)
                checks = _check(response, turn.get("expected", {}), http_status)
                passed = bool(checks) and all(checks.values())
            except Exception as exc:  # noqa: BLE001 - 报告网络/解析失败，不中断整套验收
                http_status, response, wall_latency_ms = 0, {"error": str(exc)}, round((time.monotonic() - started) * 1000, 1)
                checks, passed = {"request_completed": False}, False
            turn_results.append({
                "turn": index,
                "message": turn["message"],
                "passed": passed,
                "checks": checks,
                "http_status": http_status,
                "wall_latency_ms": wall_latency_ms,
                "response": response,
            })
        results.append({
            "case_id": case_id,
            "category": case.get("category", ""),
            "passed": all(turn["passed"] for turn in turn_results),
            "turns": turn_results,
        })
    total_cases = len(results)
    passed_cases = sum(case["passed"] for case in results)
    total_turns = sum(len(case["turns"]) for case in results)
    passed_turns = sum(turn["passed"] for case in results for turn in case["turns"])
    return {
        "suite_version": data.get("suite_version"),
        "run_id": run_id,
        "dataset": str(dataset),
        "base_url": base_url,
        "case_total": total_cases,
        "case_passed": passed_cases,
        "case_pass_rate": round(passed_cases / total_cases, 4) if total_cases else 0,
        "turn_total": total_turns,
        "turn_passed": passed_turns,
        "turn_pass_rate": round(passed_turns / total_turns, 4) if total_turns else 0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--timeout", type=float, default=55)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_suite(pathlib.Path(args.dataset), args.base_url, args.timeout)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "suite_version", "case_total", "case_passed", "case_pass_rate",
        "turn_total", "turn_passed", "turn_pass_rate",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
