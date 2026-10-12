
from tools import calculate_metric, run_tool, read_file


def test_failure_rate():
    assert calculate_metric(2, 4) == 50.0


def test_zero_total_rejected():
    result = run_tool(
        "calculate_metric",
        {"failed": 0, "total": 0}
    )
    assert "error" in result


def test_forbidden_tool_rejected():
    result = run_tool("delete_file", {"filename": "logs.txt"})
    assert result == {"error": "Tool not allowed"}


def test_invalid_arguments_rejected():
    result = run_tool(
        "calculate_metric",
        {"failed": 2, "total": 4, "extra": "unexpected"}
    )
    assert "error" in result


def test_logs_can_be_searched():
    result = run_tool("search_logs", {"keyword": "FAIL"})
    assert "result" in result
    assert len(result["result"]) > 0

def test_cannot_read_file_outside_data_folder():
    try:
        read_file("../agent.py")
        assert False, "Access should have been rejected"
    except ValueError:
        pass

def test_cannot_execute_unknown_tool():
    result = run_tool("delete_file", {"filename": "logs.txt"})
    assert result == {"error": "Tool not allowed"}


def test_missing_file_is_rejected():
    result = run_tool("read_file", {"filename": "missing_file.md"})
    assert "error" in result


def test_timeout_search_finds_timed_out_log():
    result = run_tool("search_logs", {"keyword": "Timeout"})
    assert "result" in result
    assert any(
        "timed out" in line.lower()
        for line in result["result"]
    )


def test_documentation_search_finds_timeout_guidance():
    result = run_tool("search_docs", {"keyword": "Timeout"})
    assert "result" in result
    assert any(
        "service availability" in section.lower()
        for section in result["result"]
    )