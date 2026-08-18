# Architecture

```text
External sources
  -> providers / fetch layer
  -> extraction + normalization
  -> structured signals
  -> evidence weighting
  -> baseline + analyst arena
  -> structured prediction
  -> checkpoint comparison
  -> frozen pre-event snapshot
  -> post-event evaluation
  -> calibration / bounded learning
  -> JSON output contract
```

## Main modules

- `providers.py`: external evidence retrieval.
- `deepseek_extractor.py`: structured evidence extraction.
- `structured_data.py`: shared domain structures.
- `prediction.py`: probabilistic forecasting logic.
- `arena.py`: senior-analyst model orchestration.
- `evaluator.py`: prediction-versus-result evaluation.
- `learning.py`: bounded calibration and feedback.
- `race_calendar.py`: race/session context and checkpoints.
- `output_contract.py`: client-facing JSON contract.
- `runner.py`: orchestration.

## Principles

**Evidence before narrative.** Store observation separately from interpretation.

**Recency with quality controls.** Representative weekend evidence may override stale priors when justified.

**Immutable scoring snapshots.** Predictions used for evaluation are frozen before results are known.

**Bounded learning.** Historical calibration adjusts future forecasts without overpowering fresh evidence.

**Client independence.** Engine output is JSON; clients do not need to embed engine logic.
