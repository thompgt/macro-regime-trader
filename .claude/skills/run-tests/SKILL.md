---
name: run-tests
description: Run the macro_regime_trader test suite and report only failures, not full pytest output. Use whenever you need to check whether tests pass after a code change in this repo.
---

Run:

```
cd C:\Users\thoma\macro_regime_trader && python -m pytest -q 2>&1 | tail -20
```

If all tests pass, report just the pass count (e.g. "36 passed"). Do not paste
full pytest output back to the user.

If any test fails, re-run just the failing file with `-v` to get the specific
assertion/traceback, e.g.:

```
cd C:\Users\thoma\macro_regime_trader && python -m pytest tests/test_<module>.py -v 2>&1 | tail -60
```

Report only the failing test name(s) and the relevant assertion line — not
the full traceback or passing-test output.
