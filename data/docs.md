# General CI Troubleshooting Notes

These notes are general investigation guidance. Use them to decide what evidence to check; the CI log is the source of facts about a particular build. Do not assume a cause or claim a fix worked unless the log shows it.

## Dependency or environment setup failures

Look for the first package-manager or environment-creation error, not only the final build failure. Record the requested language and package versions, package-manager command, configured package sources, and the exact error. Check whether the requested versions are available from those sources and compatible with one another. Treat an unavailable version as a possibility until the log confirms it.

## Test failures

Find the first failing test and capture its name, assertion or exception, and nearby setup output. Check whether later lines show a retry, a subsequent pass, or a separate failure. A retry that passes is useful context, but it does not erase the original failure. Report test totals only when the log explicitly provides them.

## HTTP 5xx responses

An HTTP 5xx response indicates that a server reported an error. Find the request, timestamp, response body, and related server-side or dependency errors in the available logs. The status code alone does not identify the root cause. If only client-side output is available, state that server details are missing.

## Timeouts

Find which operation timed out, how long it waited, and whether it was retried. Check nearby evidence for slow dependencies, connection failures, rate limits, or resource pressure. Do not conclude that a service was unavailable unless the log supports that conclusion.

## Build blocked or cancelled

Search earlier in the log for the condition that blocked or cancelled the build. Separate the explicit blocking reason from warnings and unrelated failures. If the log only reports that the build was blocked, say the underlying reason was not established.

## Security and configuration failures

Quote the specific policy or configuration finding and identify the affected component when the log names it. Do not repeat secret values in a report. A scanner finding is evidence that a check reported an issue; use the scanner details to describe severity and impact, and avoid inferring exploitability.

## Test counts and failure rates

Use only explicit total, passed, and failed counts from the log. If the log omits a count, report it as unavailable. When counts are present, calculate failure rate as failed tests divided by total tests, multiplied by 100; do not calculate when the total is missing or zero.

## Evidence-based reporting

Prefer the earliest specific error and include nearby lines needed to understand it. Keep observed facts separate from likely explanations. Cite line numbers when available, mention conflicting or missing evidence, and describe a cause as uncertain when the log does not prove it.
