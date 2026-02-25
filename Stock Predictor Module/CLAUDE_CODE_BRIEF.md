# A.V.I.S. Stock Predictor v6 — Claude Code Integration Brief

## What You're Working With
`stock_predictor.jsx` is a self-contained React component that predicts stock prices using a 25-method quantitative ensemble. It was built and debugged over an extensive session. The math has been verified, but you should independently verify it yourself before integrating.

## Your Tasks

### 1. Review the Code and Math
Read `stock_predictor.jsx` end-to-end. Independently verify:
- All mathematical formulas (MJD, James-Stein shrinkage, bootstrap CI, Kelly criterion, prediction intervals) are implemented correctly
- No NaN propagation, no undefined references, no dead code
- Weight arrays are the correct length and sum to 1.0
- The vol-adaptive signal scaling logic is sound (this is the core v6 feature — signals scale by `max(σ, 0.15) / 0.25` so high-vol stocks like IREN get meaningful signal estimates)
- The bootstrap CI runs on pre-shrinkage (raw) estimates, not post-shrinkage
- Prediction intervals use per-tail quadrature, not a single scale factor

If you find any issues, fix them before integrating.

### 2. Integrate into the Meridian Desktop Application
Add the stock predictor as a feature within the existing Meridian app. Explore the Meridian codebase first to understand its architecture, routing, styling conventions, component patterns, and state management before making any changes. The predictor should feel native to Meridian, not bolted on. Adapt the component's styling, layout, and patterns to match what already exists in the app.

### 3. Review the Data Fetching Approach
The component currently calls the Anthropic API (Haiku model with the `web_search` tool) directly from the client to fetch live market data. Review whether this is the best approach given:
- Meridian's existing architecture and any backend it already has
- Whether there's a better model to use than `claude-haiku-4-5-20251001` for structured data retrieval
- Whether a direct financial data API (Alpha Vantage, Polygon, Yahoo Finance, etc.) would be faster, cheaper, or more reliable than having an LLM search the web and format JSON
- API key security — the current implementation exposes the Anthropic API call client-side, which only works in the Claude.ai artifact sandbox
- Latency — the current approach takes 10-15 seconds per prediction

Recommend and implement the best data fetching strategy for Meridian's context. If you change the data source, ensure the JSON schema returned to `runModel()` remains identical (all field names and types must match).

## File
- `stock_predictor.jsx` — 1,046 lines, single-file React component, default export `StockPredictor`
- Only dependency: React (`useState`, `useCallback`)
- Fonts: Google Fonts (JetBrains Mono, DM Sans) loaded via `<link>` tag
