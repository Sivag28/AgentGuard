import json

from agent import _render_synthesis, _synthesis_schema


SOURCE = "uploaded log 'complex_enterprise_ci_log _1_.txt' (temporary)"


def _trace(lines, counts=None):
    trace = [{
        "tool": "extract_failure_evidence",
        "result": {"result": [f"L{number}: {text}" for number, text in lines]},
    }]
    if counts is not None:
        trace.append({"tool": "get_test_summary", "result": {"result": counts}})
    return trace


def test_first_failure_report_renders_exact_headings_and_adjacent_evidence():
    trace = _trace([
        (30, "[18:42:23.041] PASS UT-004 - Inventory reservation decrements available stock."),
        (31, "[18:42:23.409] FAIL UT-005 - Concurrent checkout may reserve the same inventory unit."),
        (32, "[18:42:23.410] Expected invariant: reserved_quantity <= available_quantity"),
        (33, "[18:42:23.410] Actual: reserved_quantity=12, available_quantity=10"),
    ])

    result = _render_synthesis(
        "Find the first failure and explain its likely cause.",
        json.dumps({"likely_explanation": "Concurrent checkout likely over-reserved inventory."}),
        trace,
        SOURCE,
        "logs.txt",
    )

    assert result.startswith("**First failure:** L31\n\n**Evidence:**")
    assert "L31: [18:42:23.409] FAIL UT-005" in result
    assert "L32: [18:42:23.410] Expected invariant" in result
    assert "L33: [18:42:23.410] Actual: reserved_quantity=12, available_quantity=10" in result
    assert "**Likely explanation:** Concurrent checkout likely over-reserved inventory." in result
    assert "L30:" not in result


def test_summary_report_uses_only_retrieved_citations_and_exact_sections():
    trace = _trace([
        (35, "[18:42:24.008] FAIL IT-001 - Payment provider sandbox returned HTTP 504."),
        (39, "[18:42:24.901] PASS IT-001 - Payment provider responded successfully on retry 2."),
        (57, "[18:42:27.105] FAIL SEC-003 - Staging configuration contains a default JWT signing secret."),
    ])

    result = _render_synthesis(
        "Summarize the strongest evidence in this log.",
        json.dumps({
            "findings": [
                {
                    "label": "IT-001 (Payment Provider Issue)",
                    "explanation": "The initial 504 was recovered on retry 2.",
                    "source_lines": [35, 39, 999],
                },
                {
                    "label": "SEC-003 (Default JWT Secret)",
                    "explanation": "Staging used a default signing secret.",
                    "source_lines": [57],
                },
            ]
        }),
        trace,
        SOURCE,
        "logs.txt",
    )

    assert result.startswith("**Summary of STRONGEST EVIDENCE:**")
    assert "**IT-001 (Payment Provider Issue):**" in result
    assert "**SEC-003 (Default JWT Secret):**" in result
    assert "**Source Logs:**\nL35, L39, L57" in result
    assert "Citation check" in result and "L999" in result
    assert "L999" not in result.split("**Source Logs:**", 1)[1].split("**Citation check:**", 1)[0]


def test_counts_report_uses_tool_counts_and_separate_evidence_lines():
    trace = _trace([
        (31, "[18:42:23.409] FAIL UT-005 - Concurrent checkout may reserve the same inventory unit."),
        (57, "[18:42:27.105] FAIL SEC-003 - Staging configuration contains a default JWT signing secret."),
    ], {
        "available": True,
        "total": 10,
        "passed": 6,
        "failed": 4,
        "failure_rate_percent": 40.0,
    })

    result = _render_synthesis(
        "Check whether test counts or failure summaries are available.",
        json.dumps({
            "failure_summaries": [
                {
                    "label": "UT-005",
                    "explanation": "Concurrent checkout failed.",
                    "source_lines": [31],
                },
                {
                    "label": "SEC-003",
                    "explanation": "The staging JWT secret violated policy.",
                    "source_lines": [57],
                },
            ]
        }),
        trace,
        SOURCE,
        "logs.txt",
    )

    assert result.startswith("**Counts:**\nThe test-count tool reported 10 total tests, 6 passed, and 4 failed")
    assert "**Failure Summaries:**" in result
    assert "**Verified log evidence:**" in result
    assert f"`{SOURCE}:L31`" in result
    assert f"`{SOURCE}:L57`" in result


def test_supported_report_requests_use_json_schema():
    assert _synthesis_schema("Find the first failure.")["required"] == ["likely_explanation"]
    assert "findings" in _synthesis_schema("Summarize the strongest evidence.")["properties"]
    assert "failure_summaries" in _synthesis_schema("Check test counts.")["properties"]
