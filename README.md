# AgentGuard

AgentGuard is a local AI-assisted CI failure investigator. It uses Ollama for language-model reasoning, a bounded Python tool harness for evidence gathering, local troubleshooting documents, a Streamlit interface, and a model-based golden-set evaluator. Its active log is now a real, historically captured CI build from the LogChunks research dataset; source and attribution details are in `data/source.md`.

## Run locally

Requirements: Python 3.11 or newer, Ollama, and a model that supports tool calling. The default model is `qwen2.5:3b`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
streamlit run app.py
```

Run the golden set with `python evaluate.py`. The evaluator exits with a nonzero code if a case fails and reports per-case tool use and latency. Model-based results can vary across model versions and hardware; treat changes in the score as a regression signal and inspect failed cases.

## Configuration

Set these environment variables before starting the app or evaluator:

| Variable | Default | Purpose |
|---|---:|---|
| `AGENTGUARD_MODEL` | `qwen2.5:3b` | Ollama model name |
| `AGENTGUARD_MAX_STEPS` | `8` | Maximum model turns in one run |
| `AGENTGUARD_MAX_TOOL_CALLS` | `12` | Maximum tool calls in one run |
| `AGENTGUARD_MAX_TASK_CHARS` | `8000` | Maximum submitted task length |
| `AGENTGUARD_MAX_TOOL_RESULT_CHARS` | `20000` | Maximum serialized tool result passed to the model |
| `AGENTGUARD_MAX_CONTEXT_CHARS` | `100000` | Maximum accumulated conversation size |
| `AGENTGUARD_MODEL_TIMEOUT` | `90` | Ollama request timeout, in seconds |
| `AGENTGUARD_MODEL_RETRIES` | `1` | Retries after a model request error |

## Design and controls

- **Harness and tool loop:** `agent.py` creates a task-specific checklist, updates goal status from tool evidence, enforces step and call budgets, blocks repeated searches including case/whitespace variants, bounds tool output, retries failed model requests, and records elapsed time and trace metadata. Bundled known workflows use a fast evidence-first report; uploaded logs receive a model-written synthesis over the gathered log and documentation evidence. Goals without supporting results are reported as unresolved; these controls support focused investigations but do not make the model a general-purpose autonomous agent.
- **Tool boundary:** `tools.py` is the allowlisted tool registry. Bundled tools read only files inside `data/`, reject traversal, cap readable file sizes, and expose no write or shell operations. Uploaded logs are scanned for prioritized failure evidence with nearby context; UTF-8 `.txt` uploads are limited to 1 MB and processed in session memory without saving to the project. The content is sent to the configured Ollama model (local by default).
- **Prompt injection resistance:** retrieved content is treated as untrusted evidence in the system instructions. The prompt is not a security boundary; tool code must continue to enforce access rules independently.
- **RAG:** `search_docs` splits Markdown into heading sections and ranks matching sections with a small BM25-style lexical scorer. It requires no external embedding service. It is not semantic/vector retrieval; add a separately evaluated embedding index if paraphrase retrieval becomes a requirement.
- **Real data:** `data/logs.txt` is a historical failed scikit-learn Travis CI build (ID 562398767) with a published failure annotation. It has no test-count summary, so the dashboard reports that those metrics are unavailable rather than inventing counts. The former demo log is kept in `data/sample_logs.txt`; the downloaded dataset archive is in `datasets/LogChunks.zip`.
- **Evaluation:** `golden_set.json` holds representative tasks and expected answer evidence/tool use. `evaluate.py` checks expected substrings or regexes, required/forbidden tools, completion state, and latency. Keep cases tied to real failure modes, include adversarial and no-evidence examples, and review false passes as well as failures.
- **UI and observability:** the Streamlit app lets the user select a bundled log or upload a temporary CI text log, shows source metadata and test counts, prioritizes explicit blocking conditions and concrete failure lines in evidence-first reports, and provides live investigation progress, model-step and tool-call counts, and an inspectable trace. Numbered citations open a small surrounding excerpt from the exact log used for that report. The harness logs model errors without writing prompts or source contents to application logs.

## Production deployment limits

This repository provides a stronger local baseline, not a complete production deployment. The bundled public log is real but historical, not current data from your own CI environment. The Streamlit app has no user authentication, tenant isolation, request quotas, centralized telemetry, or deployment health checks. Keep it bound to a trusted machine/network until those controls are added. For a shared deployment, define the identity and data-access model, protect the Ollama endpoint, add authentication and authorization, redact sensitive trace data, configure monitoring and alerting, and run security and evaluation reviews against the actual threat model. Pin and audit dependencies and model artifacts as part of release management.
