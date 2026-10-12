
# Troubleshooting Guide

## Real CI build: scikit-learn build 562398767
This historical Travis CI build log is from the LogChunks research dataset. The annotated failure chunk reports `PackagesNotFoundError`: the configured Conda channels could not provide the requested Python 3.4 environment, including scipy 0.16.1, numpy 1.10.4, cython 0.25.2, and pillow 4.0.0. The evidence points to an unavailable legacy dependency environment during setup; this log does not include test-count totals.

## HTTP 500 errors
An HTTP 500 error indicates that a server encountered an unexpected
condition. Inspect the server logs and service dependencies.

## Timeouts
A timeout can occur when a service takes too long to respond.
Check service availability and response times.

## Test failure rates
Failure rate = failed tests / total tests * 100.
Always calculate from the recorded test counts.
