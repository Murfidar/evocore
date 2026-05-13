# Optimizer Telemetry

`OptimizationTelemetry` tracks the true breadth and cost of optimizer search.

Telemetry includes proposed candidates, screened candidates, partial evaluations, full
evaluations, promoted and eliminated counts by rung, and cost by rung. Trading systems can
use this evidence for anti-overfitting workflows such as Deflated Sharpe Ratio, White's
Reality Check, Hansen SPA, and Model Confidence Set.
