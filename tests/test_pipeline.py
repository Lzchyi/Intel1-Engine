from __future__ import annotations

import json
import re
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

from intel1_pipeline.ai import build_extraction_cache, extract_signals, openai_extract_model_name, parse_json_object
from intel1_pipeline.arena import build_prediction_arena, prediction_prompt_payload, validate_prediction_prompt
from intel1_pipeline.cli import build_parser
from intel1_pipeline.config import SourceConfig
from intel1_pipeline.deepseek_extractor import extract_hybrid_signals
from intel1_pipeline.evaluator import evaluate_prediction_arena
from intel1_pipeline.learning import update_learning_state
from intel1_pipeline.output_contract import REQUIRED_FILES, source_log_payload
from intel1_pipeline.prediction import build_prediction, evidence_quality, weekend_phase
from intel1_pipeline.providers import OpenAIProvider, ProviderJSONResponse, deepseek_model_name, openai_model_supports_temperature
from intel1_pipeline.race_calendar import RACES_2026, load_static_current_weekend, prediction_checkpoints, session_end, weekend_sessions
from intel1_pipeline.reference_standings import reference_driver_standings
from intel1_pipeline.runner import RunOptions, enrich_prediction_with_previous, race_result_prediction_arena, run
from intel1_pipeline.session_result_extractor import supplement_session_results_with_deepseek
from intel1_pipeline.signal_store import signals_to_stored_signals
from intel1_pipeline.signals import ExtractedSignal, deterministic_extract_signals
from intel1_pipeline.source_items import fetch_items_for_source, make_item
from intel1_pipeline.standings_updater import apply_pending_session_results_to_standings, apply_session_results_to_standings
from intel1_pipeline.structured_data import WeekendContext, load_current_weekend
from intel1_pipeline.summary import build_summary, openai_summary_model_name


class PipelineTests(unittest.TestCase):

    def test_weekend_phase_progresses_from_friday_to_race_review(self) -> None:
        results = empty_session_results()
        results["fp2"] = [{"position": 1, "driver": "Driver One"}]
        self.assertEqual(weekend_phase("after_fp2", results), "friday_intelligence")
        results["qualifying"] = [{"position": 1, "driver": "Driver One"}]
        self.assertEqual(weekend_phase("after_qualifying", results), "race_forecast")
        results["race"] = [{"position": 1, "driver": "Driver One"}]
        self.assertEqual(weekend_phase("post_race", results), "review")

    def test_evidence_quality_rewards_official_confirmed_signal(self) -> None:
        quality = evidence_quality([sample_signal("pace", ["Driver One"], ["Team A"], "Official session evidence")])
        self.assertEqual(quality["signal_count"], 1)
        self.assertEqual(quality["official_signal_share"], 1.0)
        self.assertEqual(quality["confirmed_signal_share"], 1.0)
        self.assertGreaterEqual(quality["score"], 0.8)

    def test_change_digest_compares_same_weekend_only(self) -> None:
        current = {
            "run_id": "new", "weekend_id": "2026-test-gp",
            "race": {"driver_win_probabilities": [
                {"driver": "Driver One", "probability": 0.40},
                {"driver": "Driver Two", "probability": 0.30},
            ]},
            "prediction_delta_vs_previous": [], "change_digest": [],
        }
        previous = {
            "run_id": "old", "weekend_id": "2026-test-gp",
            "race": {"driver_win_probabilities": [
                {"driver": "Driver One", "probability": 0.31},
                {"driver": "Driver Two", "probability": 0.36},
            ]},
        }
        enrich_prediction_with_previous(current, previous)
        self.assertEqual(current["comparison_baseline_run_id"], "old")
        self.assertEqual(current["change_digest"][0]["target_name"], "Driver One")
        self.assertAlmostEqual(current["change_digest"][0]["delta"], 0.09)
        self.assertAlmostEqual(current["race"]["driver_win_probabilities"][0]["delta_vs_previous"], 0.09)

    def test_dry_run_writes_app_contract_files(self) -> None:
        weekend = WeekendContext(
            weekend_id="2026-test-gp",
            grand_prix_name="Test Grand Prix",
            circuit_name="Test Circuit",
            country="Testland",
            year=2026,
            round_number=1,
            race_date="2026-05-24",
            is_sprint_weekend=False,
            stage="pre_weekend",
            next_relevant_session="fp1",
            session_schedule=[{"session": "fp1", "label": "Practice 1"}],
        )
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 80.0, "wins": 1},
        ]
        constructors = [
            {"team": "Team A", "team_id": "team-a", "position": 1, "points": 140.0, "wins": 4},
            {"team": "Team B", "team_id": "team-b", "position": 2, "points": 110.0, "wins": 1},
        ]
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            with patch("intel1_pipeline.runner.load_current_weekend", return_value=weekend), patch(
                "intel1_pipeline.runner.load_driver_standings", return_value=drivers
            ), patch("intel1_pipeline.runner.load_constructor_standings", return_value=constructors), patch(
                "intel1_pipeline.runner.load_historical_race_data", return_value={"season": 2026, "races": []}
            ), patch(
                "intel1_pipeline.runner.load_session_results", return_value=empty_session_results()
            ), patch(
                "intel1_pipeline.runner.fetch_source_items", return_value=([], [])
            ):
                result = run(
                    RunOptions(
                        output_dir=output_dir,
                        source_registry=Path("config/source_registry.json"),
                        force_weekend_id="2026-test-gp",
                        force_stage="pre_weekend",
                        scheduled=False,
                        dry_run=True,
                        skip_ai=True,
                        skip_drive_upload=True,
                        max_items_per_source=2,
                        public_base_url="https://example.com/intel1",
                    )
                )

            self.assertFalse(result["skipped"])
            for name in REQUIRED_FILES + ["app_manifest.json"]:
                self.assertTrue(output_dir.joinpath(name).exists(), name)

            prediction = json.loads(output_dir.joinpath("latest_prediction.json").read_text(encoding="utf-8"))
            manifest = json.loads(output_dir.joinpath("app_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(prediction["race"]["predicted_winner"]["driver"], "Driver One")
            self.assertIn("session_results", prediction)
            self.assertIn("performance_comparison", prediction["race"])
            self.assertGreaterEqual(prediction["race"]["performance_comparison"][0]["speed_index"], 0)
            self.assertEqual(manifest["files"][0]["url"], "https://example.com/intel1/current_weekend.json")

    def test_win_probability_groups_sum_to_one_and_winner_is_top(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="after_sprint_qualifying")
        prediction = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )

        self.assertAlmostEqual(sum(item["probability"] for item in prediction["race"]["driver_win_probabilities"]), 1.0, places=4)
        self.assertAlmostEqual(sum(item["probability"] for item in prediction["sprint"]["driver_win_probabilities"]), 1.0, places=4)
        self.assertEqual(prediction["race"]["predicted_winner"]["driver"], prediction["race"]["driver_win_probabilities"][0]["driver"])

    def test_grid_matching_handles_openf1_uppercase_driver_names(self) -> None:
        drivers = [
            {"driver": "Andrea Kimi Antonelli", "team": "Mercedes", "position": 1, "points": 112.0, "wins": 3},
            {"driver": "George Russell", "team": "Mercedes", "position": 2, "points": 96.0, "wins": 2},
            {"driver": "Lando Norris", "team": "McLaren", "position": 3, "points": 90.0, "wins": 1},
            {"driver": "Oscar Piastri", "team": "McLaren", "position": 4, "points": 84.0, "wins": 1},
        ]
        results = empty_session_results()
        results["qualifying"] = [
            session_row(1, "George RUSSELL", "Mercedes", "1:12.578"),
            session_row(2, "Kimi ANTONELLI", "Mercedes", "+0.068s"),
            session_row(3, "Lando NORRIS", "McLaren", "+0.151s"),
            session_row(4, "Oscar PIASTRI", "McLaren", "+0.203s"),
        ]

        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(stage="after_qualifying"),
            drivers=drivers,
            signals=[],
            source_count=3,
            session_results=results,
        )

        self.assertEqual(prediction["race"]["predicted_winner"]["driver"], "George Russell")
        self.assertEqual(prediction["race"]["predicted_winner"]["starting_position"], 1)

    def test_sprint_probability_uses_sprint_grid_not_race_prior_only(self) -> None:
        results = empty_session_results()
        results["sprint_qualifying"] = [
            session_row(1, "Driver Two", "Team B", "1:10.000"),
            session_row(2, "Driver Three", "Team C", "+0.100"),
            session_row(3, "Driver Four", "Team D", "+0.200"),
            session_row(4, "Driver One", "Team A", "+0.300"),
        ]
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(is_sprint_weekend=True, stage="after_sprint_qualifying"),
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=results,
        )

        self.assertEqual(prediction["sprint"]["predicted_winner"]["driver"], "Driver Two")
        self.assertEqual(prediction["race"]["predicted_winner"]["driver"], "Driver One")

    def test_constructor_probabilities_sum_to_one(self) -> None:
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(),
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=empty_session_results(),
        )

        self.assertAlmostEqual(sum(item["probability"] for item in prediction["race"]["constructor_win_probabilities"]), 1.0, places=4)

    def test_aggressive_podium_call_has_reasoning(self) -> None:
        signal = sample_signal(
            signal_type="race_pace_positive",
            drivers=["Driver Three"],
            teams=["Team C"],
            evidence_summary="Long-run pace was materially stronger than the surrounding group.",
        )
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(stage="after_qualifying"),
            drivers=sample_drivers(),
            signals=[signal],
            source_count=4,
            session_results=sample_session_results(driver_three_start=6),
        )

        for entry in prediction["race"]["predicted_podium"]:
            if entry.get("positions_to_gain", 0) >= 3:
                self.assertTrue(entry["reasoning_factors"], entry)

    def test_session_results_decode_shape(self) -> None:
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(stage="after_qualifying"),
            drivers=sample_drivers(),
            signals=[],
            source_count=2,
            session_results=sample_session_results(),
        )

        results = prediction["session_results"]
        self.assertEqual(sorted(results.keys()), ["fp1", "fp2", "fp3", "qualifying", "race", "sprint", "sprint_qualifying"])
        row = results["qualifying"][0]
        self.assertEqual(row["position"], 1)
        self.assertEqual(row["source"], "Jolpica")
        self.assertTrue(row["is_official"])

    def test_metric_definitions_have_labels_and_explanations(self) -> None:
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(),
            drivers=sample_drivers(),
            signals=[],
            source_count=2,
            session_results=empty_session_results(),
        )

        self.assertGreaterEqual(len(prediction["metric_definitions"]), 4)
        for item in prediction["metric_definitions"]:
            self.assertTrue(item["title"])
            self.assertTrue(item["unit"])
            self.assertTrue(item["explanation"])
            self.assertTrue(item["source"])

    def test_parse_json_object_accepts_fenced_content(self) -> None:
        payload = parse_json_object('```json\n{"signals": []}\n```')
        self.assertEqual(payload, {"signals": []})

    def test_openai_model_env_is_split_by_pipeline_stage(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(openai_extract_model_name(), "gpt-5.5")
            self.assertEqual(openai_summary_model_name(), "gpt-5.5")
        with patch.dict("os.environ", {"OPENAI_MODEL": "fallback-model"}, clear=True):
            self.assertEqual(openai_extract_model_name(), "fallback-model")
            self.assertEqual(openai_summary_model_name(), "fallback-model")
        with patch.dict(
            "os.environ",
            {"OPENAI_MODEL": "fallback-model", "OPENAI_EXTRACT_MODEL": "cheap-model", "OPENAI_SUMMARY_MODEL": "smart-model"},
            clear=True,
        ):
            self.assertEqual(openai_extract_model_name(), "cheap-model")
            self.assertEqual(openai_summary_model_name(), "smart-model")

    def test_cli_default_uses_broader_source_scan(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.max_items_per_source, 16)

    def test_weak_rumour_does_not_create_deterministic_fallback_signal(self) -> None:
        source = SourceConfig(
            id="rumour_blog",
            name="Rumour Blog",
            tier="D",
            enabled=True,
            connector_type="html_index_page",
            reliability_weight=0.3,
        )
        item = make_item(
            source=source,
            title="Rumour says a driver may have a grid penalty",
            url="https://example.com/rumour",
            raw_excerpt="Unconfirmed rumour of a grid penalty with no FIA document or team confirmation.",
            fetch_status="success",
        )

        signals = deterministic_extract_signals([item], "2026-test-gp", "run-1", "pre_weekend")

        self.assertEqual(signals, [])

    def test_official_fia_document_can_create_high_impact_material_signal(self) -> None:
        source = SourceConfig(
            id="fia_f1_2026_documents",
            name="FIA Documents",
            tier="A",
            enabled=True,
            connector_type="html_document_index",
            reliability_weight=1.0,
        )
        item = make_item(
            source=source,
            title="FIA decision confirms grid penalty for Car 16",
            url="https://www.fia.com/documents/example",
            raw_excerpt="Official FIA stewards decision confirms a grid penalty.",
            fetch_status="success",
        )

        signals = deterministic_extract_signals([item], "2026-test-gp", "run-1", "final_pre_race")

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].impact_level, "high")
        self.assertEqual(signals[0].evidence_type, "official_fact")
        self.assertEqual(signals[0].corroboration_status, "officially_confirmed")
        self.assertTrue(signals[0].can_shift_probability)
        self.assertTrue(signals[0].material_change)

    def test_api_availability_sentinel_does_not_create_signal(self) -> None:
        source = SourceConfig(
            id="openf1",
            name="OpenF1",
            tier="A",
            enabled=True,
            connector_type="api",
            reliability_weight=1.0,
            base_url="https://api.openf1.org/v1/",
            role="session_timing_weather_race_control",
        )
        item = make_item(
            source=source,
            title="OpenF1 API available",
            url="https://api.openf1.org/v1/",
            raw_excerpt="Structured source role: session_timing_weather_race_control",
            fetch_status="success",
        )

        self.assertEqual(deterministic_extract_signals([item], "2026-test-gp", "run-1", "after_fp2"), [])

    def test_html_index_fetches_strategy_article_body(self) -> None:
        source = SourceConfig(
            id="formula1_official_latest",
            name="Formula 1 Official Latest",
            tier="A",
            enabled=True,
            connector_type="html_index_page",
            url="https://www.formula1.com/en/latest",
            reliability_weight=1.0,
            source_type="official_f1",
        )
        index_html = """
        <html><head><title>Latest F1</title><meta name="description" content="Formula 1 latest"></head>
        <body><a href="/en/latest/article/strategy-guide-canadian-grand-prix">
        Strategy Guide - what are the tactical options for Sunday's Canadian Grand Prix
        </a></body></html>
        """
        article_html = """
        <html><head><title>Strategy Guide</title></head>
        <body><article>Safety Car probability is high, Mercedes has track position, and Russell starts from pole.</article></body></html>
        """

        def fake_fetch(url: str, timeout: int = 15) -> str:
            return article_html if "strategy-guide" in url else index_html

        with patch("intel1_pipeline.source_items.fetch_text", side_effect=fake_fetch):
            items = fetch_items_for_source(source, max_items=4)

        self.assertEqual(len(items), 1)
        self.assertIn("Safety Car probability", items[0].raw_excerpt)
        self.assertIn("Russell starts from pole", items[0].raw_content or "")

    def test_rss_fetches_weekend_article_body(self) -> None:
        source = SourceConfig(
            id="motorsport_f1",
            name="Motorsport.com Formula 1",
            tier="B",
            enabled=True,
            connector_type="rss",
            url="https://www.motorsport.com/rss/f1/news/",
            reliability_weight=0.8,
            source_type="trusted_news",
        )
        rss = """
        <rss><channel><item>
        <title>Canadian GP qualifying results: Russell takes pole</title>
        <link>https://example.com/canada-qualifying</link>
        <description>Short RSS summary.</description>
        </item></channel></rss>
        """
        article_html = """
        <html><body><article>Russell takes pole, Antonelli starts second, and Norris lines up third.</article></body></html>
        """

        def fake_fetch(url: str, timeout: int = 15) -> str:
            return article_html if "canada-qualifying" in url else rss

        with patch("intel1_pipeline.source_items.fetch_text", side_effect=fake_fetch):
            items = fetch_items_for_source(source, max_items=4)

        self.assertEqual(len(items), 1)
        self.assertIn("Russell takes pole", items[0].raw_content or "")
        self.assertIn("Antonelli starts second", items[0].raw_excerpt)

    def test_html_article_source_creates_single_article_item(self) -> None:
        source = SourceConfig(
            id="reuters_formula1_canada",
            name="Reuters Formula 1 Canada",
            tier="B",
            enabled=True,
            connector_type="html_article",
            url="https://www.reuters.com/sports/formula1/example/",
            reliability_weight=0.86,
            source_type="trusted_news",
        )
        article_html = """
        <html><head><title>Norris encouraged by McLaren pace</title></head>
        <body><article>Norris says McLaren race pace is closer than expected despite the Mercedes front-row lockout.</article></body></html>
        """

        with patch("intel1_pipeline.source_items.fetch_text", return_value=article_html):
            items = fetch_items_for_source(source, max_items=4)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, source.url)
        self.assertIn("Mercedes front-row lockout", items[0].raw_content or "")

    def test_unchanged_source_hash_skips_ai_reprocessing(self) -> None:
        source = SourceConfig(
            id="formula1_official_latest",
            name="Formula 1 Official Latest",
            tier="A",
            enabled=True,
            connector_type="html_index_page",
            reliability_weight=1.0,
            source_type="official_f1",
        )
        item = make_item(
            source=source,
            title="Team finds race pace on long run",
            url="https://example.com/race-pace",
            raw_excerpt="Analysis says the long run pace looked stronger, while fuel loads remain unknown.",
            fetch_status="success",
        )
        cached_signals = deterministic_extract_signals([item], "2026-test-gp", "previous-run", "after_fp2")
        cache = build_extraction_cache([item], cached_signals)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True), patch("intel1_pipeline.ai.openai_extract") as openai_extract:
            signals = extract_signals(
                items=[item],
                weekend_id="2026-test-gp",
                run_id="run-2",
                stage="after_fp2",
                skip_ai=False,
                cached_signals_by_hash=cache,
            )

        openai_extract.assert_not_called()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].run_id, "run-2")
        self.assertEqual(signals[0].source_content_hash, item.content_hash)

    def test_non_material_update_skips_openai_summary_regeneration(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="after_fp1")
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 80.0, "wins": 1},
        ]
        source = SourceConfig(
            id="rumour_blog",
            name="Rumour Blog",
            tier="D",
            enabled=True,
            connector_type="html_index_page",
            reliability_weight=0.3,
        )
        item = make_item(
            source=source,
            title="Rumour says a driver may have a grid penalty",
            url="https://example.com/rumour",
            raw_excerpt="Unconfirmed rumour of a grid penalty with no FIA document or team confirmation.",
            fetch_status="success",
        )
        signals = deterministic_extract_signals([item], weekend.weekend_id, "run-1", weekend.stage)
        prediction = build_prediction(run_id="run-1", weekend=weekend, drivers=drivers, signals=signals, source_count=1)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True), patch("intel1_pipeline.summary.openai_summary") as openai_summary:
            summary = build_summary("run-1", weekend, prediction, signals, skip_ai=False)

        openai_summary.assert_not_called()
        self.assertFalse(prediction["material_change_detected"])
        self.assertIsNotNone(summary["sprint_outlook"])

    def test_openai_summary_can_be_disabled_even_for_material_update(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="after_qualifying")
        prediction = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[sample_signal("pace", ["Driver One"], ["Team A"], "Driver One has a confirmed session pace gain.")],
            source_count=1,
            session_results=sample_session_results(),
        )
        prediction["material_change_detected"] = True

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "INTEL1_OPENAI_SUMMARY_ENABLED": "false"}, clear=True), patch("intel1_pipeline.summary.openai_summary") as openai_summary:
            summary = build_summary("run-1", weekend, prediction, [], skip_ai=False)

        openai_summary.assert_not_called()
        self.assertEqual(summary["schema_version"], "1.1")

    def test_static_2026_calendar_detects_canada_after_sq(self) -> None:
        weekend = load_static_current_weekend(now=datetime(2026, 5, 23, 1, 4, tzinfo=UTC))

        self.assertIsNotNone(weekend)
        assert weekend is not None
        self.assertEqual(weekend.grand_prix_name, "Canadian Grand Prix")
        self.assertEqual(weekend.stage, "after_sprint_qualifying")
        self.assertEqual(weekend.next_relevant_session, "sprint")
        self.assertTrue(weekend.is_sprint_weekend)

    def test_static_2026_calendar_advances_after_canada_sprint(self) -> None:
        weekend = load_static_current_weekend(now=datetime(2026, 5, 23, 17, 45, tzinfo=UTC))

        self.assertIsNotNone(weekend)
        assert weekend is not None
        self.assertEqual(weekend.grand_prix_name, "Canadian Grand Prix")
        self.assertEqual(weekend.stage, "after_sprint")
        self.assertEqual(weekend.next_relevant_session, "qualifying")

    def test_static_2026_calendar_keeps_canada_for_post_race_result_window(self) -> None:
        weekend = load_static_current_weekend(now=datetime(2026, 5, 25, 1, 30, tzinfo=UTC))

        self.assertIsNotNone(weekend)
        assert weekend is not None
        self.assertEqual(weekend.grand_prix_name, "Canadian Grand Prix")
        self.assertEqual(weekend.stage, "post_race")
        self.assertIsNone(weekend.next_relevant_session)

    def test_current_weekend_uses_static_calendar_before_live_api(self) -> None:
        with patch("intel1_pipeline.race_calendar.utc_now", return_value=datetime(2026, 5, 23, 1, 4, tzinfo=UTC)):
            weekend = load_current_weekend(None, None)

        self.assertEqual(weekend.weekend_id, "2026-r5-canadian-grand-prix")
        self.assertEqual(weekend.stage, "after_sprint_qualifying")

    def test_static_2026_calendar_keeps_monaco_on_race_day(self) -> None:
        weekend = load_static_current_weekend(now=datetime(2026, 6, 7, 12, 0, tzinfo=UTC))

        self.assertIsNotNone(weekend)
        assert weekend is not None
        self.assertEqual(weekend.weekend_id, "2026-r6-monaco-grand-prix")
        self.assertEqual(weekend.grand_prix_name, "Monaco Grand Prix")
        self.assertEqual(weekend.stage, "after_qualifying")
        self.assertEqual(weekend.next_relevant_session, "race")

    def test_static_2026_calendar_advances_after_monaco_post_race_window(self) -> None:
        weekend = load_static_current_weekend(now=datetime(2026, 6, 7, 22, 1, tzinfo=UTC))

        self.assertIsNotNone(weekend)
        assert weekend is not None
        self.assertEqual(weekend.weekend_id, "2026-r7-barcelona-catalunya-grand-prix")
        self.assertEqual(weekend.grand_prix_name, "Barcelona-Catalunya Grand Prix")

    def test_github_action_schedule_covers_high_value_prediction_checkpoints(self) -> None:
        workflow = Path(".github/workflows/intel1.yml").read_text()
        actual = re.findall(r'cron: "([^"]+)"', workflow)
        expected = []
        for race in RACES_2026:
            for checkpoint in prediction_checkpoints(race):
                run_at = checkpoint["run_at"]
                expected.append(f"{run_at.minute:02d} {run_at.hour:02d} {run_at.day:02d} {run_at.month:02d} *")

        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), set(expected))

    def test_2026_calendar_includes_confirmed_bahrain_gp_in_malaysia_session_times(self) -> None:
        sepang = next(race for race in RACES_2026 if race["round"] == 16)
        self.assertEqual(sepang["name"], "Bahrain Grand Prix in Malaysia")
        self.assertEqual(sepang["circuit"], "Petronas Sepang International Circuit")
        self.assertEqual(sepang["date"], "2026-10-04")
        self.assertEqual(sepang["time"], "19:00")
        self.assertTrue(sepang["session_times_confirmed"])

        sessions = weekend_sessions(sepang)
        local = ZoneInfo("Asia/Kuala_Lumpur")
        local_times = {
            item["session"]: item["starts_at"].astimezone(local).strftime("%Y-%m-%d %H:%M")
            for item in sessions
        }
        self.assertEqual(local_times["fp1"], "2026-10-02 16:30")
        self.assertEqual(local_times["fp2"], "2026-10-02 20:00")
        self.assertEqual(local_times["fp3"], "2026-10-03 16:30")
        self.assertEqual(local_times["qualifying"], "2026-10-03 20:00")
        self.assertEqual(local_times["race"], "2026-10-04 19:00")

        checkpoints = prediction_checkpoints(sepang)
        self.assertEqual(
            [item["run_at"].strftime("%Y-%m-%d %H:%M") for item in checkpoints],
            ["2026-10-03 02:30", "2026-10-04 05:00", "2026-10-04 16:00"],
        )

    def test_rounds_after_sepang_are_renumbered_through_abu_dhabi(self) -> None:
        self.assertEqual(next(r for r in RACES_2026 if r["name"] == "Singapore Grand Prix")["round"], 17)
        self.assertEqual(next(r for r in RACES_2026 if r["name"] == "Abu Dhabi Grand Prix")["round"], 23)

    def test_deepseek_model_defaults_to_v4_pro(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(deepseek_model_name(), "deepseek-v4-pro")

    def test_openai_default_only_models_omit_temperature(self) -> None:
        self.assertFalse(openai_model_supports_temperature("gpt-5.5"))
        self.assertFalse(openai_model_supports_temperature("o3"))
        self.assertTrue(openai_model_supports_temperature("gpt-4.1"))

    def test_deepseek_extraction_valid_json_creates_stored_signal(self) -> None:
        source = SourceConfig(
            id="trusted_news",
            name="Trusted News",
            tier="B",
            enabled=True,
            connector_type="rss",
            reliability_weight=0.8,
            source_type="trusted_news",
        )
        item = make_item(
            source=source,
            title="Team shows stronger long run pace",
            url="https://example.com/pace",
            raw_excerpt="Analysis says the team looked stronger over a long run.",
            fetch_status="success",
        )
        provider = FakeProvider(
            {
                "signals": [
                    {
                        "source_item_id": item.source_item_id,
                        "signalType": "race_pace_positive",
                        "target": "Driver One",
                        "drivers": ["Driver One"],
                        "teams": ["Team A"],
                        "summary": "Long-run pace signal improved for Driver One.",
                        "strength": 0.7,
                        "confidence": 0.65,
                        "sourceQuality": 0.8,
                        "evidenceType": "reported",
                        "canShiftProbability": True,
                    }
                ]
            },
            model_name="deepseek-v4-pro",
            request_id="deepseek-request-1",
        )

        result = extract_hybrid_signals(
            items=[item],
            weekend_id="2026-test-gp",
            run_id="run-1",
            stage="after_fp2",
            skip_ai=False,
            cached_signals_by_hash={},
            provider=provider,
        )

        self.assertEqual(len(result.signals), 1)
        self.assertEqual(result.extraction_errors, [])
        stored = result.stored_signals[0]
        self.assertEqual(stored.sourceQuality, 0.8)
        self.assertEqual(stored.evidenceType, "reported")
        self.assertEqual(stored.inputHash, item.content_hash)
        self.assertTrue(stored.sourceBatchId)
        self.assertEqual(stored.providerRequestId, "deepseek-request-1")
        self.assertEqual(stored.modelTemperature, 0.0)

    def test_deepseek_validation_failure_returns_no_ai_signal_and_error(self) -> None:
        source = SourceConfig(
            id="trusted_news",
            name="Trusted News",
            tier="B",
            enabled=True,
            connector_type="rss",
            reliability_weight=0.8,
            source_type="trusted_news",
        )
        item = make_item(
            source=source,
            title="Rumour says a driver may have a grid penalty",
            url="https://example.com/rumour",
            raw_excerpt="Unconfirmed rumour with no official document.",
            fetch_status="success",
        )
        provider = FakeProvider({"signals": [{"source_item_id": "missing", "signalType": "rumour", "target": "Driver One"}]})

        result = extract_hybrid_signals(
            items=[item],
            weekend_id="2026-test-gp",
            run_id="run-1",
            stage="pre_weekend",
            skip_ai=False,
            cached_signals_by_hash={},
            provider=provider,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(len(result.extraction_errors), 1)
        self.assertEqual(result.extraction_errors[0].errorType, "validation_or_provider_error")

    def test_prediction_arena_normalizes_percentages_and_records_disagreement(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )
        chatgpt_provider = FakeProvider(
            prediction_payload("Driver One", "Driver Two"),
            model_name="gpt-test",
            request_id="chatgpt-request-1",
        )
        deepseek_provider = FakeProvider(
            prediction_payload("Driver Two", "Driver One"),
            model_name="deepseek-v4-pro",
            request_id="deepseek-request-1",
        )

        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=chatgpt_provider,
            deepseek_provider=deepseek_provider,
        )

        chatgpt = arena["predictions"]["chatgpt"]
        consensus = arena["predictions"]["intel1_consensus"]
        self.assertEqual(round(sum(item["probability"] for item in chatgpt["win_probabilities"]), 2), 100.0)
        self.assertEqual(round(sum(item["probability"] for item in chatgpt["constructor_win_probabilities"]), 2), 100.0)
        self.assertEqual(chatgpt["predicted_winner"], chatgpt["win_probabilities"][0]["driver"])
        self.assertEqual(chatgpt["analyst_report"]["final_call"]["winner_driver"], "Driver One")
        self.assertTrue(consensus["disagreement_notes"])

    def test_prediction_provider_accepts_percent_string_values(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )
        payload = prediction_payload("Driver One", "Driver Two")
        for group in ["win_probabilities", "constructor_win_probabilities", "podium_probabilities", "top10_probabilities", "dnf_risk"]:
            for entry in payload[group]:
                entry["probability"] = f"{entry['probability']}%"
        payload["confidence"] = "0.72"
        payload["safety_car_probability"] = "75-85%"

        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=FakeProvider(payload),
            deepseek_provider=FakeProvider(prediction_payload("Driver Two", "Driver One"), model_name="deepseek-v4-pro"),
        )

        chatgpt = arena["predictions"]["chatgpt"]
        self.assertEqual(chatgpt["providerStatus"], "ok")
        self.assertEqual(chatgpt["predicted_winner"], "Driver One")
        self.assertEqual(chatgpt["safety_car_probability"], 75.0)

    def test_consensus_merges_kimi_antonelli_aliases(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )

        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=FakeProvider(prediction_payload("George Russell", "Andrea Kimi Antonelli")),
            deepseek_provider=FakeProvider(prediction_payload("George Russell", "Kimi Antonelli"), model_name="deepseek-v4-pro"),
        )

        names = [entry["driver"] for entry in arena["predictions"]["intel1_consensus"]["win_probabilities"]]
        self.assertIn("Andrea Kimi Antonelli", names)
        self.assertNotIn("Kimi Antonelli", names)
        self.assertEqual(names.count("Andrea Kimi Antonelli"), 1)

    def test_prediction_arena_preserves_provider_analyst_report(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )
        payload = prediction_payload("Driver One", "Driver Two")
        payload["analyst_report"] = {
            "title": "Test GP prediction",
            "assumption": "After Qualifying data only.",
            "final_call": {
                "winner_driver": "Driver One",
                "winner_constructor": "Team A",
                "podium": ["Driver One", "Driver Two", "Driver Three"],
                "highest_scoring_team": "Team A",
                "safety_car_risk": "High - around 75%",
                "rain_impact": "Medium",
                "chaos_level": "High",
                "most_likely_upset_winner": "Driver Two",
                "dark_horse_podium": ["Driver Four"],
            },
            "narrative": [{"title": "Why this pick", "body": "Qualifying and structured pace signals favour Driver One."}],
            "strategy": {"dry": "Likely one-stop.", "wet_mixed": "Safety Car timing matters more."},
            "biggest_risks": [{"risk": "Lap 1 fight", "benefits": ["Driver Two"]}],
            "final_answer": "Winner: Driver One. Constructor: Team A.",
        }

        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=FakeProvider(payload),
            deepseek_provider=FakeProvider(prediction_payload("Driver Two", "Driver One"), model_name="deepseek-v4-pro"),
        )

        report = arena["predictions"]["chatgpt"]["analyst_report"]
        self.assertEqual(report["title"], "Test GP prediction")
        self.assertEqual(report["final_call"]["podium"], ["Driver One", "Driver Two", "Driver Three"])
        self.assertEqual(report["strategy"]["wet_mixed"], "Safety Car timing matters more.")

    def test_reference_market_damps_backmarker_win_and_podium_probabilities(self) -> None:
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(is_sprint_weekend=True, stage="after_sprint_qualifying"),
            drivers=reference_driver_standings(2026),
            signals=[],
            source_count=3,
            session_results=empty_session_results(),
        )

        win_entries = prediction["race"]["driver_win_probabilities"]
        podium_entries = prediction["race"]["driver_podium_probabilities"]
        lance_win = next(item for item in win_entries if item["driver"] == "Lance Stroll")
        lance_podium = next(item for item in podium_entries if item["driver"] == "Lance Stroll")
        self.assertGreater(win_entries[0]["probability"], 0.2)
        self.assertLess(lance_win["probability"], 0.01)
        self.assertLess(lance_podium["probability"], 0.03)

    def test_prediction_provider_failure_uses_deterministic_baseline(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )

        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=FailingProvider(),
            deepseek_provider=FailingProvider(model_name="deepseek-v4-pro"),
        )

        self.assertEqual(arena["predictions"]["chatgpt"]["providerStatus"], "deterministic_baseline")
        self.assertEqual(arena["predictions"]["deepseek"]["providerStatus"], "deterministic_baseline")

    def test_consensus_does_not_rewrite_prediction_to_official_winner(self) -> None:
        weekend = sample_weekend(stage="post_race")
        session_results = sample_session_results()
        session_results["race"] = [session_row(1, "Driver Two", "Team B", "1:32:00.000")]
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=session_results,
        )

        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=session_results,
            chatgpt_provider=FakeProvider(prediction_payload("Driver One", "Driver Two")),
            deepseek_provider=FakeProvider(prediction_payload("Driver One", "Driver Two"), model_name="deepseek-v4-pro"),
        )

        consensus = arena["predictions"]["intel1_consensus"]
        self.assertEqual(consensus["predicted_winner"], "Driver One")
        self.assertAlmostEqual(sum(item["probability"] for item in consensus["win_probabilities"]), 100.0)

    def test_race_result_uses_frozen_history_prediction_when_previous_publish_has_result(self) -> None:
        session_results = sample_session_results()
        session_results["race"] = [session_row(1, "Driver Two", "Team B", "1:30:00")]
        frozen = race_result_prediction_arena(
            run_id="run-2",
            weekend_id="2026-test-gp",
            stage="post_race",
            session_results=session_results,
            previous_prediction={"weekend_id": "2026-test-gp", "session_results": {"race": [session_row(1, "Driver Two", "Team B", "1:30:00")]}},
            previous_prediction_arena={},
            previous_manifest={"run_id": "cheating-run"},
            previous_prediction_history=[
                {
                    "run_id": "clean-run",
                    "weekend_id": "2026-test-gp",
                    "timestamp": "2026-05-24T12:00:00Z",
                    "stage": "after_qualifying",
                    "top_driver_win_probabilities": [{"driver": "Driver One", "probability": 0.46}],
                    "top_constructor_win_probabilities": [{"team": "Team A", "probability": 0.52}],
                    "safety_car": {"probability_at_least_one": 0.4},
                },
                {
                    "run_id": "cheating-run",
                    "weekend_id": "2026-test-gp",
                    "timestamp": "2026-05-25T03:00:00Z",
                    "stage": "post_race",
                    "top_driver_win_probabilities": [{"driver": "Driver Two", "probability": 1.0}],
                    "top_constructor_win_probabilities": [{"team": "Team B", "probability": 1.0}],
                    "safety_car": {"probability_at_least_one": 0.4},
                },
            ],
        )

        consensus = frozen["predictions"]["intel1_consensus"]
        self.assertEqual(frozen["stage"], "after_qualifying")
        self.assertEqual(consensus["predicted_winner"], "Driver One")
        self.assertEqual(consensus["providerStatus"], "history_snapshot")

        evaluation = evaluate_prediction_arena(
            run_id="run-2",
            weekend_id="2026-test-gp",
            arena_payload=frozen,
            session_results=session_results,
        )
        self.assertEqual(evaluation["evaluations"]["intel1_consensus"]["actual_winner"], "Driver Two")
        self.assertNotIn("chatgpt", evaluation["evaluations"])

    def test_openai_prediction_provider_can_be_disabled_without_skipping_deepseek(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "INTEL1_OPENAI_PREDICTION_ENABLED": "false"}, clear=True):
            arena = build_prediction_arena(
                run_id="run-1",
                weekend=weekend,
                baseline_prediction=baseline,
                stored_signals=[],
                session_results=sample_session_results(),
                chatgpt_provider=OpenAIProvider(model_name="gpt-test"),
                deepseek_provider=FakeProvider(prediction_payload("Driver Two", "Driver One"), model_name="deepseek-v4-pro"),
            )

        self.assertEqual(arena["predictions"]["chatgpt"]["providerStatus"], "deterministic_baseline")
        self.assertIn("disabled", arena["predictions"]["chatgpt"]["validation_errors"][0])
        self.assertEqual(arena["predictions"]["deepseek"]["providerStatus"], "ok")

    def test_deepseek_session_result_fallback_fills_missing_qualifying_rows(self) -> None:
        source = SourceConfig(
            id="formula1_official_latest",
            name="Formula 1 Official Latest",
            tier="A",
            enabled=True,
            connector_type="html_index_page",
            reliability_weight=1.0,
            source_type="official_f1",
        )
        item = make_item(
            source=source,
            title="Canadian Grand Prix qualifying results and classification",
            url="https://example.com/canada-qualifying-results",
            raw_excerpt="Qualifying results classification: P1 Driver Two 1:10.000, P2 Driver One +0.100.",
            fetch_status="success",
        )
        provider = FakeProvider(
            {
                "session_results": {
                    "qualifying": [
                        {
                            "position": 1,
                            "driver": "Driver Two",
                            "constructor": "Team B",
                            "time_or_gap": "1:10.000",
                            "status": "classified",
                            "source_item_id": item.source_item_id,
                        },
                        {
                            "position": 2,
                            "driver": "Driver One",
                            "constructor": "Team A",
                            "time_or_gap": "+0.100",
                            "status": "classified",
                            "source_item_id": item.source_item_id,
                        },
                    ]
                }
            },
            model_name="deepseek-v4-pro",
        )

        patched = supplement_session_results_with_deepseek(
            items=[item],
            weekend=sample_weekend(stage="after_qualifying"),
            session_results=empty_session_results(),
            skip_ai=False,
            provider=provider,
        )

        self.assertEqual(patched["qualifying"][0]["driver"], "Driver Two")
        self.assertEqual(patched["qualifying"][0]["time_or_gap"], "1:10.000")
        self.assertTrue(patched["qualifying"][0]["is_official"])

    def test_session_results_overlay_updates_stale_championship_standings(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="post_race")
        weekend.round_number = 2
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 97.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        constructors = [
            {"team": "Team A", "team_id": "team-a", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"team": "Team B", "team_id": "team-b", "position": 2, "points": 95.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        results = empty_session_results()
        results["sprint"] = [
            session_row(1, "Driver Two", "Team B", "29:10.000"),
            session_row(2, "Driver One", "Team A", "+1.000"),
        ]
        results["race"] = [
            session_row(1, "Driver Two", "Team B", "1:30:00"),
            session_row(2, "Driver One", "Team A", "+2.0"),
        ]

        updated_drivers, updated_constructors = apply_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=results,
        )

        self.assertEqual(updated_drivers[0]["driver"], "Driver Two")
        self.assertEqual(updated_drivers[0]["points"], 130.0)
        self.assertEqual(updated_drivers[0]["wins"], 2)
        team_b = next(item for item in updated_constructors if item["team"] == "Team B")
        self.assertEqual(team_b["points"], 128.0)
        self.assertEqual(team_b["wins"], 2)

    def test_session_results_overlay_skips_current_round_standings(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="post_race")
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 97.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        constructors = [
            {"team": "Team A", "team_id": "team-a", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"team": "Team B", "team_id": "team-b", "position": 2, "points": 95.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        results = empty_session_results()
        results["race"] = [session_row(1, "Driver Two", "Team B", "1:30:00")]

        updated_drivers, updated_constructors = apply_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=results,
        )

        self.assertEqual(updated_drivers[1]["points"], 97.0)
        self.assertEqual(updated_constructors[1]["points"], 95.0)

    def test_pending_standings_overlay_runs_each_session_once(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="post_race")
        weekend.round_number = 2
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 97.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        constructors = [
            {"team": "Team A", "team_id": "team-a", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"team": "Team B", "team_id": "team-b", "position": 2, "points": 95.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        results = empty_session_results()
        results["sprint"] = [session_row(1, "Driver Two", "Team B", "29:10.000")]

        first_drivers, _, state = apply_pending_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=results,
            standings_update_state={},
            previous_standings_payload=None,
        )
        previous_payload = standings_payload_from_rows(first_drivers, constructors)
        second_drivers, _, second_state = apply_pending_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=results,
            standings_update_state=state,
            previous_standings_payload=previous_payload,
        )

        self.assertEqual(first_drivers[0]["driver"], "Driver Two")
        self.assertEqual(first_drivers[0]["points"], 105.0)
        self.assertEqual(second_drivers[0]["points"], 105.0)
        self.assertEqual(len(second_state["applied_events"]), 1)

    def test_pending_standings_overlay_applies_race_after_sprint_once(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="post_race")
        weekend.round_number = 2
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 97.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        constructors = [
            {"team": "Team A", "team_id": "team-a", "position": 1, "points": 100.0, "wins": 3, "_source": "jolpica", "_round": 1},
            {"team": "Team B", "team_id": "team-b", "position": 2, "points": 95.0, "wins": 1, "_source": "jolpica", "_round": 1},
        ]
        sprint_results = empty_session_results()
        sprint_results["sprint"] = [session_row(1, "Driver Two", "Team B", "29:10.000")]
        sprint_drivers, sprint_constructors, state = apply_pending_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=sprint_results,
            standings_update_state={},
            previous_standings_payload=None,
        )
        race_results = dict(sprint_results)
        race_results["race"] = [session_row(1, "Driver Two", "Team B", "1:30:00")]

        race_drivers, _, state = apply_pending_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=race_results,
            standings_update_state=state,
            previous_standings_payload=standings_payload_from_rows(sprint_drivers, sprint_constructors),
        )

        self.assertEqual(race_drivers[0]["driver"], "Driver Two")
        self.assertEqual(race_drivers[0]["points"], 130.0)
        self.assertEqual(race_drivers[0]["wins"], 2)
        self.assertEqual([event["session"] for event in state["applied_events"]], ["sprint", "race"])

    def test_session_results_overlay_matches_antonelli_given_name_variant(self) -> None:
        weekend = sample_weekend(stage="post_race")
        drivers = [
            {"driver": "Kimi Antonelli", "team": "Mercedes", "position": 1, "points": 100.0, "wins": 3, "_source": "reference"},
        ]
        constructors = [
            {"team": "Mercedes", "team_id": "mercedes", "position": 1, "points": 180.0, "wins": 4, "_source": "reference"},
        ]
        results = empty_session_results()
        results["race"] = [session_row(1, "Andrea Kimi Antonelli", "Mercedes", "1:30:00")]

        updated_drivers, updated_constructors = apply_session_results_to_standings(
            weekend=weekend,
            driver_standings=drivers,
            constructor_standings=constructors,
            session_results=results,
        )

        self.assertEqual(len(updated_drivers), 1)
        self.assertEqual(updated_drivers[0]["driver"], "Kimi Antonelli")
        self.assertEqual(updated_drivers[0]["points"], 125.0)
        self.assertEqual(updated_constructors[0]["points"], 205.0)

    def test_prediction_prompt_does_not_include_raw_social_text_fields(self) -> None:
        prediction = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(),
            drivers=sample_drivers(),
            signals=[],
            source_count=2,
            session_results=empty_session_results(),
        )
        payload = prediction_prompt_payload(
            weekend=sample_weekend(),
            baseline_prediction=prediction,
            stored_signals=[],
            session_results=empty_session_results(),
        )

        validate_prediction_prompt(payload)
        serialized = json.dumps(payload)
        self.assertNotIn("raw_excerpt", serialized)
        self.assertNotIn("rawText", serialized)
        self.assertNotIn("raw_content", serialized)

    def test_prediction_prompt_includes_stage_aware_evidence_board(self) -> None:
        weekend = sample_weekend(is_sprint_weekend=True, stage="final_pre_race")
        results = sample_session_results()
        signal = sample_signal(
            signal_type="race_pace_positive",
            drivers=["Driver Two"],
            teams=["Team B"],
            evidence_summary="Sprint and long-run evidence point to stronger race pace.",
        )
        prediction = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[signal],
            source_count=5,
            session_results=results,
        )
        stored_signals = signals_to_stored_signals([signal], run_id="run-1")

        payload = prediction_prompt_payload(
            weekend=weekend,
            baseline_prediction=prediction,
            stored_signals=stored_signals,
            session_results=results,
        )

        validate_prediction_prompt(payload)
        self.assertIn("evidenceBoard", payload)
        self.assertIn("official final grid and qualifying order", payload["stageWeightingGuide"]["priority"])
        self.assertIn("Driver Two", payload["evidenceBoard"]["targetEvidence"])
        self.assertEqual(payload["evidenceBoard"]["sessionSnapshot"]["qualifying"]["leader"]["driver"], "Driver One")
        self.assertTrue(any("Still choose" in rule for rule in payload["decisionRules"]))

    def test_prediction_arena_surfaces_evidence_board(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        results = sample_session_results()
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=results,
        )
        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=results,
            chatgpt_provider=FakeProvider(prediction_payload("Driver One", "Driver Two")),
            deepseek_provider=FakeProvider(prediction_payload("Driver Two", "Driver One"), model_name="deepseek-v4-pro"),
        )

        self.assertEqual(arena["evidenceBoard"]["sessionSnapshot"]["qualifying"]["leader"]["driver"], "Driver One")
        self.assertEqual(arena["evidenceBoard"]["baselineTop"]["predictedWinner"]["driver"], "Driver One")

    def test_source_log_redacts_social_raw_excerpt(self) -> None:
        source = SourceConfig(
            id="reddit_f1",
            name="Reddit F1",
            tier="D",
            enabled=True,
            connector_type="rss",
            reliability_weight=0.3,
            source_type="reddit",
        )
        item = make_item(
            source=source,
            title="Fan thread",
            url="https://reddit.example/thread",
            raw_excerpt="Raw social post text should not persist.",
            fetch_status="success",
        )

        payload = source_log_payload("run-1", sample_weekend(), [item], [], {})

        self.assertEqual(payload["fetched_sources"][0]["raw_excerpt"], "")
        self.assertEqual(payload["fetched_sources"][0]["title"], "Reddit F1 social item")

    def test_result_evaluator_scores_models_against_actual_result(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )
        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=FakeProvider(prediction_payload("Driver One", "Driver Two")),
            deepseek_provider=FakeProvider(prediction_payload("Driver Two", "Driver One"), model_name="deepseek-v4-pro"),
        )
        results = sample_session_results()
        results["race"] = [
            session_row(1, "Driver One", "Team A", "1:30:00"),
            session_row(2, "Driver Two", "Team B", "+2.0"),
            session_row(3, "Driver Three", "Team C", "+5.0"),
        ]

        evaluation = evaluate_prediction_arena(
            run_id="run-1",
            weekend_id=weekend.weekend_id,
            arena_payload=arena,
            session_results=results,
        )

        self.assertEqual(evaluation["status"], "evaluated")
        self.assertEqual(evaluation["latest_classified_session"], "race")
        self.assertIn("podium_hit_rate", evaluation["evaluations"]["chatgpt"])

    def test_result_evaluator_waits_for_race_result_after_qualifying(self) -> None:
        evaluation = evaluate_prediction_arena(
            run_id="run-1",
            weekend_id="2026-test-gp",
            arena_payload={
                "predictions": {
                    "intel1_consensus": {
                        "predicted_winner": "Driver One",
                        "win_probabilities": [{"driver": "Driver One", "probability": 52.0}],
                    }
                }
            },
            session_results=sample_session_results(),
        )

        self.assertEqual(evaluation["status"], "awaiting_race_result")
        self.assertEqual(evaluation["latest_classified_session"], "qualifying")
        self.assertEqual(evaluation["evaluations"], {})

    def test_result_evaluator_compares_driver_names_by_compact_key(self) -> None:
        evaluation = evaluate_prediction_arena(
            run_id="run-1",
            weekend_id="2026-test-gp",
            arena_payload={
                "predictions": {
                    "intel1_consensus": {
                        "predicted_winner": "Driver One",
                        "win_probabilities": [{"driver": "Driver One", "probability": 52.0}],
                        "podium_probabilities": [{"driver": "Driver One", "probability": 80.0}],
                        "top10_probabilities": [{"driver": "Driver One", "probability": 95.0}],
                        "dnf_risk": [],
                    }
                }
            },
            session_results={"race": [session_row(1, "Driver ONE", "Team A", "1:30:00")]},
        )

        self.assertEqual(evaluation["evaluations"]["intel1_consensus"]["prediction_accuracy"], 1.0)

    def test_learning_state_updates_after_classified_race(self) -> None:
        weekend = sample_weekend(stage="after_qualifying")
        baseline = build_prediction(
            run_id="run-1",
            weekend=weekend,
            drivers=sample_drivers(),
            signals=[],
            source_count=3,
            session_results=sample_session_results(),
        )
        arena = build_prediction_arena(
            run_id="run-1",
            weekend=weekend,
            baseline_prediction=baseline,
            stored_signals=[],
            session_results=sample_session_results(),
            chatgpt_provider=FakeProvider(prediction_payload("Driver Two", "Driver One")),
            deepseek_provider=FakeProvider(prediction_payload("Driver Two", "Driver One"), model_name="deepseek-v4-pro"),
        )
        results = sample_session_results()
        results["race"] = [
            session_row(1, "Driver One", "Team A", "1:30:00"),
            session_row(2, "Driver Two", "Team B", "+2.0"),
            session_row(3, "Driver Three", "Team C", "+5.0"),
        ]
        evaluation = evaluate_prediction_arena(
            run_id="run-1",
            weekend_id=weekend.weekend_id,
            arena_payload=arena,
            session_results=results,
        )

        learning_state = update_learning_state(
            {},
            run_id="run-1",
            weekend_id=weekend.weekend_id,
            arena_payload=arena,
            evaluation=evaluation,
            session_results=results,
        )

        self.assertEqual(learning_state["events"][0]["session"], "race")
        self.assertIn("Driver One", learning_state["driver_adjustments"])
        self.assertGreater(learning_state["driver_adjustments"]["Driver One"]["score_delta"], 0)

    def test_learning_adjustment_can_move_future_baseline(self) -> None:
        drivers = [
            {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 0},
            {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 99.0, "wins": 0},
        ]
        base = build_prediction(
            run_id="run-1",
            weekend=sample_weekend(stage="pre_weekend"),
            drivers=drivers,
            signals=[],
            source_count=0,
            session_results=empty_session_results(),
        )
        learned = build_prediction(
            run_id="run-2",
            weekend=sample_weekend(stage="pre_weekend"),
            drivers=drivers,
            signals=[],
            source_count=0,
            session_results=empty_session_results(),
            learning_state={"driver_adjustments": {"Driver Two": {"score_delta": 0.15}}},
        )

        base_two = next(item for item in base["race"]["driver_win_probabilities"] if item["driver"] == "Driver Two")
        learned_two = next(item for item in learned["race"]["driver_win_probabilities"] if item["driver"] == "Driver Two")
        self.assertGreater(learned_two["probability"], base_two["probability"])

def sample_weekend(is_sprint_weekend: bool = False, stage: str = "pre_weekend") -> WeekendContext:
    return WeekendContext(
        weekend_id="2026-test-gp",
        grand_prix_name="Test Grand Prix",
        circuit_name="Test Circuit",
        country="Testland",
        year=2026,
        round_number=1,
        race_date="2026-05-24",
        is_sprint_weekend=is_sprint_weekend,
        stage=stage,
        next_relevant_session="fp1",
        session_schedule=[
            {"session": "fp1", "label": "Practice 1"},
            {"session": "fp2", "label": "Practice 2"},
            {"session": "fp3", "label": "Practice 3"},
            {"session": "sprint_qualifying", "label": "Sprint Qualifying"},
            {"session": "sprint", "label": "Sprint"},
            {"session": "qualifying", "label": "Qualifying"},
            {"session": "race", "label": "Grand Prix"},
        ],
    )


def sample_drivers() -> list[dict[str, object]]:
    return [
        {"driver": "Driver One", "team": "Team A", "position": 1, "points": 100.0, "wins": 3},
        {"driver": "Driver Two", "team": "Team B", "position": 2, "points": 92.0, "wins": 2},
        {"driver": "Driver Three", "team": "Team C", "position": 3, "points": 89.0, "wins": 1},
        {"driver": "Driver Four", "team": "Team D", "position": 4, "points": 72.0, "wins": 0},
    ]


def empty_session_results() -> dict[str, list[dict[str, object]]]:
    return {key: [] for key in ["fp1", "fp2", "fp3", "sprint_qualifying", "sprint", "qualifying", "race"]}


def sample_session_results(driver_three_start: int = 3) -> dict[str, list[dict[str, object]]]:
    results = empty_session_results()
    qualifying = [
        session_row(1, "Driver One", "Team A", "1:10.000"),
        session_row(2, "Driver Two", "Team B", "+0.100"),
        session_row(driver_three_start, "Driver Three", "Team C", "+0.220"),
        session_row(4, "Driver Four", "Team D", "+0.330"),
    ]
    results["qualifying"] = qualifying
    results["sprint_qualifying"] = qualifying
    return results


def session_row(position: int, driver: str, constructor: str, time_or_gap: str) -> dict[str, object]:
    return {
        "position": position,
        "driver": driver,
        "constructor": constructor,
        "time_or_gap": time_or_gap,
        "laps": None,
        "status": "classified",
        "source": "Jolpica",
        "is_official": True,
    }


def standings_payload_from_rows(drivers: list[dict[str, object]], constructors: list[dict[str, object]]) -> dict[str, object]:
    return {
        "driver_standings": [
            {
                "id": item.get("driver_id") or item.get("driver"),
                "position": item.get("position"),
                "points": item.get("points"),
                "wins": item.get("wins"),
                "primary_label": item.get("driver"),
                "secondary_label": item.get("team"),
                "team_label": item.get("team"),
                "team_id": item.get("team_id"),
                "driver_id": item.get("driver_id"),
            }
            for item in drivers
        ],
        "constructor_standings": [
            {
                "id": item.get("team_id") or item.get("team"),
                "position": item.get("position"),
                "points": item.get("points"),
                "wins": item.get("wins"),
                "primary_label": item.get("team"),
                "team_label": item.get("team"),
                "team_id": item.get("team_id"),
            }
            for item in constructors
        ],
    }


def sample_signal(signal_type: str, drivers: list[str], teams: list[str], evidence_summary: str) -> ExtractedSignal:
    return ExtractedSignal(
        signal_id="sig-1",
        weekend_id="2026-test-gp",
        run_id="run-1",
        session_context="after_qualifying",
        source_item_id="item-1",
        source_content_hash="hash-1",
        source_id="test",
        source_name="Test Source",
        source_tier="A",
        source_reliability_weight=1.0,
        source_url="https://example.com",
        source_published_at=None,
        teams=teams,
        drivers=drivers,
        signal_type=signal_type,
        direction="positive",
        impact_level="medium",
        confidence=0.9,
        evidence_summary=evidence_summary,
        model_relevance=["race"],
        is_confirmed=True,
        requires_corroboration=False,
        evidence_type="session_data",
        corroboration_status="officially_confirmed",
        contradicting_signal_ids=[],
        prediction_impact_targets=drivers + teams,
        severity_score=0.6,
        expiry_stage=None,
        can_shift_probability=True,
        should_surface_in_app=True,
        material_change=True,
        raw_evidence_excerpt=None,
        event_category="performance",
        linked_document_type=None,
    )


class FakeProvider:
    provider_name = "fake"

    def __init__(self, payload: dict[str, object], model_name: str = "fake-model", request_id: str = "fake-request") -> None:
        self.payload = payload
        self.model_name = model_name
        self.model_temperature = 0.0
        self.request_id = request_id

    def complete_json(self, *, system_prompt: str, user_payload: dict[str, object]) -> ProviderJSONResponse:
        return ProviderJSONResponse(
            payload=self.payload,
            provider_request_id=self.request_id,
            model_used=self.model_name,
            model_temperature=self.model_temperature,
        )


class FailingProvider(FakeProvider):
    def __init__(self, model_name: str = "failing-model") -> None:
        super().__init__({}, model_name=model_name)

    def complete_json(self, *, system_prompt: str, user_payload: dict[str, object]) -> ProviderJSONResponse:
        raise RuntimeError("provider unavailable")


def prediction_payload(winner: str, second: str) -> dict[str, object]:
    return {
        "confidence": 0.72,
        "predicted_winner": winner,
        "win_probabilities": [
            {"driver": winner, "probability": 46},
            {"driver": second, "probability": 28},
            {"driver": "Driver Three", "probability": 16},
            {"driver": "Driver Four", "probability": 10},
        ],
        "constructor_win_probabilities": [
            {"team": "Team A", "probability": 45},
            {"team": "Team B", "probability": 30},
            {"team": "Team C", "probability": 15},
            {"team": "Team D", "probability": 10},
        ],
        "podium_probabilities": [
            {"driver": winner, "probability": 80},
            {"driver": second, "probability": 65},
            {"driver": "Driver Three", "probability": 40},
        ],
        "top10_probabilities": [
            {"driver": winner, "probability": 96},
            {"driver": second, "probability": 92},
            {"driver": "Driver Three", "probability": 88},
            {"driver": "Driver Four", "probability": 82},
        ],
        "dnf_risk": [
            {"driver": winner, "probability": 8},
            {"driver": second, "probability": 8},
        ],
        "safety_car_probability": 48,
        "key_reasons": ["Structured signals favour the selected winner."],
        "weak_assumptions": ["Provider does not know final race result."],
    }


if __name__ == "__main__":
    unittest.main()
