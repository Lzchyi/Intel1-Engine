# INTEL1 Engine

**Open-source Formula 1 race intelligence and prediction engine.**

INTEL1 turns race-weekend evidence into structured forecasts, then measures those forecasts against what actually happened. It is designed around a simple principle: **predict what happens next, explain why, and keep score.**

## What it does

- Collects and structures race-weekend evidence from configured sources
- Produces session-aware qualifying and race forecasts
- Tracks probability movement between analysis checkpoints
- Scores evidence quality and uncertainty
- Freezes pre-event predictions for honest post-event evaluation
- Measures calibration, Brier score, winner/podium/Top-10 performance
- Carries bounded learning forward without letting old races overpower fresh evidence
- Publishes a stable JSON contract for apps, dashboards, or other clients

## Architecture

```text
Sources
  ↓
Evidence extraction
  ↓
Quality + corroboration
  ↓
Analyst / prediction engine
  ↓
Structured probabilities + uncertainty
  ↓
Frozen prediction snapshot
  ↓
Event result
  ↓
Evaluation + calibration + bounded learning
  ↓
JSON output contract
```

The engine does the expensive analysis in the backend. Clients only need to consume the generated JSON.

## Quick start

```bash
git clone https://github.com/Lzchyi/Intel1-Engine.git
cd Intel1-Engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest
python -m intel1_pipeline --help
```

AI providers are optional. Add your own credentials to `.env` or GitHub Actions secrets when enabling provider-backed analysis. Never commit credentials.

## Repository

- `intel1_pipeline/` — extraction, prediction, evaluation and publishing engine
- `config/` — source configuration
- `tests/` — automated pipeline tests
- `Docs/AI_BEHAVIOUR_CONTRACT.md` — analyst behaviour and evidence rules
- `ARCHITECTURE.md` — system design and data flow
- `.github/workflows/` — scheduled automation

## Philosophy

INTEL1 is not a live-timing clone or a generic F1 news feed. It focuses on one question:

> **Given everything known right now, what is most likely to happen next?**

Forecasts are probabilities, not certainties. Conflicting evidence and uncertainty should remain visible rather than being hidden behind a confident narrative.

## License

Engine source is released under the **Apache License 2.0**. See `LICENSE` and `NOTICE.md`.

Formula 1, FIA, team names, event names and related marks belong to their respective owners. This repository does not grant rights to third-party trademarks, logos, photography, broadcast material, or other protected assets.

---

**INTEL1 Engine** · evidence → forecast → evaluation → learning
