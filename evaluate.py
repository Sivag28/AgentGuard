
import json
import re
import statistics
import sys
import time
from pathlib import Path
from agent import run_agent


def main():
    golden_set_path = Path(__file__).with_name("golden_set.json")
    with golden_set_path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("golden_set.json must contain a list of cases")

    case_ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(not case_id for case_id in case_ids):
        raise ValueError("Every golden-set case must be an object with a non-empty id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Golden-set case ids must be unique")
    for case in cases:
        if not isinstance(case.get("task"), str) or not case["task"].strip():
            raise ValueError(f'Case {case["id"]} must have a non-empty task')
        if not case.get("expected_answer_contains") and not case.get("expected_answer_regex") and not case.get("expected_tools"):
            raise ValueError(f'Case {case["id"]} has no assertions')
        for pattern in case.get("expected_answer_regex", []):
            re.compile(pattern)

    results = []
    durations = []

    for case in cases:
        started = time.monotonic()
        result = run_agent(case["task"])
        elapsed = time.monotonic() - started
        durations.append(elapsed)
        answer = result["answer"].lower()
        tools_used = {event["tool"] for event in result["trace"]}

        required_text = case.get("expected_answer_contains", [])
        required_patterns = case.get("expected_answer_regex", [])
        answer_ok = (
            all(expected.casefold() in answer for expected in required_text)
            and all(re.search(pattern, answer, re.IGNORECASE) for pattern in required_patterns)
        )
        required_tools = set(case.get("expected_tools", []))
        forbidden_tools = set(case.get("forbidden_tools", []))
        tools_ok = required_tools.issubset(tools_used) and not (forbidden_tools & tools_used)
        status_ok = result["status"] == case.get("expected_status", "completed")
        passed = answer_ok and tools_ok and status_ok

        results.append({
            "id": case["id"],
            "passed": passed,
            "answer_ok": answer_ok,
            "tools_ok": tools_ok,
            "status_ok": status_ok,
            "status": result["status"],
            "tools_used": sorted(tools_used),
            "elapsed_seconds": round(elapsed, 3),
        })

    passed_count = sum(item["passed"] for item in results)
    total = len(results)
    rate = 100 * passed_count / total if total else 0

    print(json.dumps(results, indent=2))
    print(f"\nGolden-set score: {passed_count}/{total} ({rate:.1f}%)")
    if durations:
        print(
            "Latency: "
            f"mean={statistics.mean(durations):.2f}s, "
            f"median={statistics.median(durations):.2f}s, "
            f"max={max(durations):.2f}s"
        )
    if passed_count != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
