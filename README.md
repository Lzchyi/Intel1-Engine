# INTEL1 Engine

INTEL1 Engine is an open-source Formula 1 race-intelligence and probabilistic forecasting pipeline. It collects structured evidence, separates observation from inference, produces checkpoint-based forecasts, and evaluates its own predictions after events.

> This repository contains the **engine**, not the proprietary INTEL1 iOS client.

## What it does

- Builds a structured F1 weekend context from official and trusted sources.
- Extracts and ranks evidence by source quality, recency, and corroboration.
- Produces Friday/Saturday/race forecasts with confidence and uncertainty.
- Tracks prediction history and compares each checkpoint with the previous one.
- Freezes pre-event predictions before evaluation to avoid hindsight contamination.
- Scores forecasts and maintains bounded calibration/learning state.
- Publishes machine-readable JSON suitable for any client.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m pytest
```

Run:

```bash
python -m intel1_pipeline --output-dir output/live --source-registry config/source_registry.json
```

See `ARCHITECTURE.md` and `Docs/AI_BEHAVIOUR_CONTRACT.md`.

## GitHub Actions

`.github/workflows/intel1.yml` is the reference automation. Configure API credentials through GitHub Actions **Secrets**, never committed files.

## Public/private boundary

The official INTEL1 iOS application, UI source, App Store assets, product artwork, and third-party visual assets are maintained separately and are not licensed by this repository.

## Trademarks and data

INTEL1 Engine is independent and is not affiliated with or endorsed by Formula 1, FIA, Formula One Management, any constructor, promoter, circuit, or driver. Third-party names, marks, content and datasets remain the property of their respective owners. See `NOTICE.md`.

## Licence

Apache License 2.0 for code and documentation owned by the project author, unless otherwise noted. Third-party intellectual property is excluded.
