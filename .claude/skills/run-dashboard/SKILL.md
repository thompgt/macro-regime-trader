---
name: run-dashboard
description: Launch the macro_regime_trader Streamlit dashboard in the background for manual/visual verification. Use when the user wants to see or check the dashboard UI, not for automated tests.
---

Launch in the background (it's a long-running server, not a one-shot command):

```
cd C:\Users\thoma\macro_regime_trader && python -m macro_regime_trader.cli dashboard
```

Run this with a background-capable tool call so it doesn't block. Report the
local URL (default `http://localhost:8501`) back to the user rather than
polling its output — Streamlit serves indefinitely until stopped.
