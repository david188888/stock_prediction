import pandas

from stock_prediction.evaluation.metrics import calculate_metrics
from stock_prediction.evaluation.reporting import (
    build_model_comparison_frame,
    save_conclusion_markdown,
    save_summary_markdown,
)


def test_summary_markdown_includes_aggregated_metrics(tmp_path):
    frame = pandas.DataFrame(
        [
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "holdout", "mae": 1.0, "mse": 1.0, "rmse": 1.0, "da": 0.5},
            {"symbol": "BBB", "model": "linear_regression", "evaluation": "holdout", "mae": 2.0, "mse": 4.0, "rmse": 2.0, "da": 0.6},
            {"symbol": "AAA", "model": "arima", "evaluation": "walk_forward", "mae": 1.5, "mse": 2.25, "rmse": 1.5, "da": 0.7},
        ]
    )
    path = tmp_path / "summary.md"
    save_summary_markdown(path, frame)
    content = path.read_text(encoding="utf-8")
    assert "## Aggregated Metrics" in content
    assert "linear_regression" in content


def test_conclusion_markdown_mentions_best_model(tmp_path):
    frame = pandas.DataFrame(
        [
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "holdout", "mae": 1.0, "mse": 1.0, "rmse": 1.0, "da": 0.8},
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "walk_forward", "mae": 1.1, "mse": 1.21, "rmse": 1.1, "da": 0.7},
            {"symbol": "AAA", "model": "arima", "evaluation": "holdout", "mae": 2.0, "mse": 4.0, "rmse": 2.0, "da": 0.4},
            {"symbol": "AAA", "model": "arima", "evaluation": "walk_forward", "mae": 2.2, "mse": 4.84, "rmse": 2.2, "da": 0.3},
        ]
    )
    path = tmp_path / "conclusion.md"
    save_conclusion_markdown(path, frame)
    content = path.read_text(encoding="utf-8")
    assert "holdout" in content
    assert "linear_regression" in content


def test_calculate_metrics_includes_directional_accuracy():
    metrics = calculate_metrics([11.0, 9.0, 10.0], [12.0, 8.0, 11.0], [10.0, 10.0, 10.0])
    assert metrics["rmse"] > 0
    assert metrics["da"] == 1.0
    assert metrics["direction_count"] == 2


def test_build_model_comparison_frame_marks_extreme_events():
    frame = pandas.DataFrame(
        [
            {
                "evaluation": "holdout",
                "symbol": "AAA",
                "date": "2024-01-03",
                "feature_date": "2024-01-02",
                "target_date": "2024-01-03",
                "current_close": 10.0,
                "actual_close": 11.0,
                "split_start_date": "2024-01-02",
                "model": "arima",
                "predicted": 10.8,
                "abs_error": 0.2,
                "direction_correct": True,
            },
            {
                "evaluation": "holdout",
                "symbol": "AAA",
                "date": "2024-01-03",
                "feature_date": "2024-01-02",
                "target_date": "2024-01-03",
                "current_close": 10.0,
                "actual_close": 11.0,
                "split_start_date": "2024-01-02",
                "model": "lstm",
                "predicted": 10.7,
                "abs_error": 0.3,
                "direction_correct": True,
            },
            {
                "evaluation": "holdout",
                "symbol": "AAA",
                "date": "2024-01-04",
                "feature_date": "2024-01-03",
                "target_date": "2024-01-04",
                "current_close": 11.0,
                "actual_close": 7.0,
                "split_start_date": "2024-01-02",
                "model": "arima",
                "predicted": 9.5,
                "abs_error": 2.5,
                "direction_correct": False,
            },
            {
                "evaluation": "holdout",
                "symbol": "AAA",
                "date": "2024-01-04",
                "feature_date": "2024-01-03",
                "target_date": "2024-01-04",
                "current_close": 11.0,
                "actual_close": 7.0,
                "split_start_date": "2024-01-02",
                "model": "lstm",
                "predicted": 8.5,
                "abs_error": 1.5,
                "direction_correct": True,
            },
        ]
    )
    comparison = build_model_comparison_frame(
        frame,
        model_order=["arima", "lstm"],
        extreme_volatility_quantile=0.5,
    )
    assert "pred_arima" in comparison.columns
    assert "pred_lstm" in comparison.columns
    assert comparison["is_extreme_volatility"].any()
    assert comparison["event_id"].notna().any()
