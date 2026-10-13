"""Bounded, validated tools exposed to the AgentGuard model."""

from pathlib import Path
import re
from typing import Any


DATA_DIR = (Path(__file__).parent / "data").resolve()
MAX_FILE_BYTES = 1_000_000
MAX_KEYWORD_LENGTH = 200
DEFAULT_DOC_RESULTS = 3

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_allowed_files",
            "description": "List approved data files.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read an approved data file.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": "Search approved log lines by keyword.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_metric",
            "description": "Calculate the test failure percentage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "failed": {"type": "integer", "minimum": 0},
                    "total": {"type": "integer", "minimum": 1},
                },
                "required": ["failed", "total"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Find and rank relevant troubleshooting document sections.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_test_summary",
            "description": "Read exact test counts from the log, or report when the log contains no test summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_failure_evidence",
            "description": "Extract prioritized failure, blocker, exception, retry, and nearby context lines from the selected CI log.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

ALLOWED_TOOLS = {item["function"]["name"] for item in TOOL_DEFINITIONS}


def list_allowed_files() -> list[str]:
    """List regular files within the approved data directory."""
    if not DATA_DIR.is_dir():
        raise OSError("Approved data directory is unavailable")
    return sorted(
        path.name for path in DATA_DIR.iterdir()
        if path.is_file() and path.resolve().parent == DATA_DIR
    )


def safe_data_file(filename: str) -> Path:
    """Resolve a single approved file and reject traversal and symlinks out."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Filename must be a non-empty string")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Only filenames inside the approved data folder are allowed")

    path = (DATA_DIR / filename).resolve()
    if path.parent != DATA_DIR or not path.is_file():
        raise ValueError("File is not allowed or does not exist")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds the maximum readable size")
    return path


def read_file(filename: str) -> str:
    """Read a bounded approved project data file."""
    return safe_data_file(filename).read_text(encoding="utf-8")


def validate_log_text(log_text: str) -> str:
    """Validate an in-memory UTF-8 log without persisting it."""
    if not isinstance(log_text, str) or not log_text.strip():
        raise ValueError("Uploaded log must contain text")
    try:
        encoded_size = len(log_text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("Uploaded log contains invalid Unicode text") from error
    if encoded_size > MAX_FILE_BYTES:
        raise ValueError("Log exceeds the 1 MB limit")
    if "\x00" in log_text:
        raise ValueError("Uploaded file does not appear to be a text log")
    return log_text


def _log_text(filename: str, log_text: str | None = None) -> str:
    return validate_log_text(log_text) if log_text is not None else read_file(filename)


def _validate_keyword(keyword: str) -> str:
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("Keyword must be a non-empty string")
    keyword = keyword.strip()
    if len(keyword) > MAX_KEYWORD_LENGTH:
        raise ValueError("Keyword is too long")
    return keyword


def search_logs(
    keyword: str, filename: str = "logs.txt", log_text: str | None = None
) -> list[str]:
    """Return matching log lines with one-based source line numbers."""
    keyword = _validate_keyword(keyword).casefold()
    terms = [keyword]
    if keyword in {"timeout", "timed out", "timedout"}:
        terms = ["timeout", "timed out", "timedout"]

    matches = [
        f"L{line_number}: {line}"
        for line_number, line in enumerate(_log_text(filename, log_text).splitlines(), start=1)
        if any(term in line.casefold() for term in terms)
    ]
    return matches[:100] or ["No matching log lines found."]


def get_log_context(
    filename: str, line_number: int, radius: int = 3, log_text: str | None = None
) -> list[dict[str, Any]]:
    """Read a small, numbered context window around a matched source line."""
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1:
        raise ValueError("Line number must be a positive integer")
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 10:
        raise ValueError("Context radius must be an integer from zero to ten")
    lines = _log_text(filename, log_text).splitlines()
    if line_number > len(lines):
        raise ValueError("Line number is outside the selected log")
    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    return [
        {"line": number, "text": lines[number - 1], "matched": number == line_number}
        for number in range(start, end + 1)
    ]


def extract_failure_evidence(
    filename: str = "logs.txt", log_text: str | None = None
) -> list[str]:
    """Return prioritized, numbered failure evidence plus one-line context."""
    lines = _log_text(filename, log_text).splitlines()
    scored: list[tuple[int, int]] = []
    for number, line in enumerate(lines, start=1):
        low = line.casefold()
        score = 0
        if re.search(r"\b(?:TypeError|ValueError|ReferenceError|\w+Error|\w+Exception)\b|\bTraceback\b", line):
            score += 100
        if re.search(r"(?:^|\]\s*)FAIL(?:\s|:)", line, re.IGNORECASE):
            score += 80
        if re.search(r"\bERROR\b|\bFATAL\b", line, re.IGNORECASE):
            score += 75
        if re.search(r"\b(?:blocked|cancelled|canceled)\b", low):
            score += 70
        if re.search(
            r"\b(?:expected|actual|observed|found|detected)\b\s*[:=]|"
            r"\b(?:retry|attempt)\b.{0,40}\b(?:succeed|success|pass)\w*|"
            r"\b(?:succeed|success|pass)\w*.{0,40}\b(?:retry|attempt)\b",
            low,
        ):
            score += 55
        if re.search(r"\b(?:retry|attempt)\b.*\b(?:succeed|success|pass)\w*|\b(?:succeed|success|pass)\w*.*\b(?:retry|attempt)\b", low):
            score += 55
        if re.search(r"\bWARN(?:ING)?\b", line, re.IGNORECASE):
            score += 45
        if re.search(r"\bPASS\b", line, re.IGNORECASE):
            score -= 25
        if score >= 45:
            scored.append((score, number))

    # Prioritize actionable failures but restore source order for readable context.
    selected = {number for _, number in sorted(scored, key=lambda item: (-item[0], item[1]))[:22]}
    selected_ids = {
        match.group(0).casefold()
        for number in selected
        if (match := re.search(r"\b[A-Z]{2,6}-\d+\b", lines[number - 1], re.IGNORECASE))
    }
    if selected_ids:
        selected.update(
            number for number, line in enumerate(lines, start=1)
            if any(identifier in line.casefold() for identifier in selected_ids)
        )
    context_numbers = {
        neighbor
        for number in selected
        for neighbor in (number - 1, number, number + 1)
        if 1 <= neighbor <= len(lines)
    }
    return [f"L{number}: {lines[number - 1]}" for number in sorted(context_numbers)][:50]


def calculate_metric(failed: int, total: int) -> float:
    """Calculate a test failure percentage from validated integer counts."""
    if isinstance(failed, bool) or not isinstance(failed, int):
        raise ValueError("Failed count must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("Total count must be an integer")
    if total <= 0:
        raise ValueError("Total must be greater than zero")
    if failed < 0 or failed > total:
        raise ValueError("Failed count must be between zero and total")
    return round((failed / total) * 100, 2)


def _document_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if body:
                sections.append((heading, "\n".join(body).strip()))
            heading = line.lstrip("# ").strip() or "Untitled"
            body = []
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body).strip()))
    return [(title, body) for title, body in sections if body]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def search_docs(keyword: str) -> list[str]:
    """Rank document sections by lexical relevance using a small BM25 scorer."""
    keyword = _validate_keyword(keyword)
    query_tokens = _tokens(keyword)
    if not query_tokens:
        raise ValueError("Keyword must contain searchable text")

    sections = _document_sections(read_file("docs.md"))
    if not sections:
        return ["No documentation is available."]

    tokenized = [_tokens(title + " " + body) for title, body in sections]
    average_length = sum(map(len, tokenized)) / len(tokenized) or 1
    scores: list[tuple[float, int]] = []
    for index, words in enumerate(tokenized):
        score = 0.0
        for term in set(query_tokens):
            frequency = words.count(term)
            if not frequency:
                continue
            document_frequency = sum(term in other for other in tokenized)
            inverse_frequency = 1 + (len(tokenized) - document_frequency + 0.5) / (document_frequency + 0.5)
            denominator = frequency + 1.2 * (0.25 + 0.75 * len(words) / average_length)
            score += inverse_frequency * frequency * 2.2 / denominator
        if score:
            scores.append((score, index))

    scores.sort(key=lambda item: (-item[0], item[1]))
    return [
        f"## {sections[index][0]}\n{sections[index][1]}"
        for _, index in scores[:DEFAULT_DOC_RESULTS]
    ] or ["No matching documentation found."]


def get_test_summary(filename: str = "logs.txt", log_text: str | None = None) -> dict[str, Any]:
    """Parse exact test counts, or explicitly report that they are unavailable."""
    values: dict[str, int] = {}
    for line in _log_text(filename, log_text).splitlines():
        for match in re.finditer(
            r"\b(TOTAL(?:\s+TESTS?)?|PASSED|FAILED)\s*[:=]\s*(\d+)\b",
            line,
            re.IGNORECASE,
        ):
            key, value = match.group(1).upper(), match.group(2)
            if key.startswith("TOTAL"):
                key = "TOTAL"
            if key in values:
                raise ValueError(f"Duplicate {key} test count in log")
            values[key] = int(value)

    if not values:
        return {
            "available": False,
            "reason": "The log contains no TOTAL, PASSED, and FAILED test-count summary.",
        }
    if set(values) != {"TOTAL", "PASSED", "FAILED"}:
        raise ValueError("Log must contain TOTAL, PASSED, and FAILED counts")
    total, passed, failed = values["TOTAL"], values["PASSED"], values["FAILED"]
    if total <= 0 or min(passed, failed) < 0 or passed + failed != total:
        raise ValueError("Test counts are inconsistent or invalid")
    return {
        "available": True,
        "total": total,
        "passed": passed,
        "failed": failed,
        "failure_rate_percent": round(failed / total * 100, 2),
    }


_FUNCTIONS = {
    "list_allowed_files": list_allowed_files,
    "read_file": read_file,
    "search_logs": search_logs,
    "calculate_metric": calculate_metric,
    "search_docs": search_docs,
    "get_test_summary": get_test_summary,
    "extract_failure_evidence": extract_failure_evidence,
}
_PARAMETERS = {
    item["function"]["name"]: item["function"]["parameters"]["properties"]
    for item in TOOL_DEFINITIONS
}


def run_tool(
    tool_name: str, arguments: Any, log_filename: str = "logs.txt", log_text: str | None = None
) -> dict[str, Any]:
    """Validate a model tool request and return a JSON-serializable result."""
    if tool_name not in _FUNCTIONS:
        return {"error": "Tool not allowed"}
    if not isinstance(arguments, dict):
        return {"error": "Arguments must be an object"}
    if set(arguments) != set(_PARAMETERS[tool_name]):
        return {"error": "Invalid tool arguments"}
    try:
        if tool_name == "read_file" and log_text is not None:
            if arguments["filename"] in {"logs.txt", log_filename}:
                return {"result": validate_log_text(log_text)}
        if tool_name == "search_logs":
            return {"result": search_logs(**arguments, filename=log_filename, log_text=log_text)}
        if tool_name == "get_test_summary":
            return {"result": get_test_summary(filename=log_filename, log_text=log_text)}
        if tool_name == "extract_failure_evidence":
            return {"result": extract_failure_evidence(filename=log_filename, log_text=log_text)}
        return {"result": _FUNCTIONS[tool_name](**arguments)}
    except (ValueError, OSError, TypeError) as error:
        return {"error": str(error)}
