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
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "holdout", "mae": 1.0, "mse": 1.0, "rmse": 1.0, "smape": 0.10, "da": 0.5, "up_precision": 0.6, "up_recall": 0.7, "up_f1": 0.646154, "down_precision": 0.4, "down_recall": 0.5, "down_f1": 0.444444},
            {"symbol": "BBB", "model": "linear_regression", "evaluation": "holdout", "mae": 2.0, "mse": 4.0, "rmse": 2.0, "smape": 0.20, "da": 0.6, "up_precision": 0.5, "up_recall": 0.6, "up_f1": 0.545455, "down_precision": 0.7, "down_recall": 0.8, "down_f1": 0.746667},
            {"symbol": "AAA", "model": "arima", "evaluation": "walk_forward", "mae": 1.5, "mse": 2.25, "rmse": 1.5, "smape": 0.15, "da": 0.7, "up_precision": 0.8, "up_recall": 0.75, "up_f1": 0.774194, "down_precision": 0.65, "down_recall": 0.7, "down_f1": 0.674074},
        ]
    )
    path = tmp_path / "summary.md"
    save_summary_markdown(path, frame)
    content = path.read_text(encoding="utf-8")
    assert "## Aggregated Metrics" in content
    assert "linear_regression" in content
    assert "mean_smape" in content
    assert "mean_up_precision" in content
    assert "mean_down_f1" in content


def test_conclusion_markdown_mentions_best_model(tmp_path):
    frame = pandas.DataFrame(
        [
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "holdout", "mae": 1.0, "mse": 1.0, "rmse": 1.0, "smape": 0.10, "da": 0.8, "up_precision": 0.9, "up_recall": 0.8, "up_f1": 0.847059, "down_precision": 0.75, "down_recall": 0.6, "down_f1": 0.666667},
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "walk_forward", "mae": 1.1, "mse": 1.21, "rmse": 1.1, "smape": 0.11, "da": 0.7, "up_precision": 0.8, "up_recall": 0.75, "up_f1": 0.774194, "down_precision": 0.7, "down_recall": 0.55, "down_f1": 0.616000},
            {"symbol": "AAA", "model": "arima", "evaluation": "holdout", "mae": 2.0, "mse": 4.0, "rmse": 2.0, "smape": 0.20, "da": 0.4, "up_precision": 0.5, "up_recall": 0.4, "up_f1": 0.444444, "down_precision": 0.45, "down_recall": 0.4, "down_f1": 0.423529},
            {"symbol": "AAA", "model": "arima", "evaluation": "walk_forward", "mae": 2.2, "mse": 4.84, "rmse": 2.2, "smape": 0.22, "da": 0.3, "up_precision": 0.35, "up_recall": 0.3, "up_f1": 0.323077, "down_precision": 0.4, "down_recall": 0.35, "down_f1": 0.373333},
        ]
    )
    path = tmp_path / "conclusion.md"
    save_conclusion_markdown(path, frame)
    content = path.read_text(encoding="utf-8")
    assert "holdout" in content
    assert "linear_regression" in content
    assert "平均 MAE" in content
    assert "平均 SMAPE" in content
    assert "上涨 F1" in content
    assert "下跌 F1" in content


def test_calculate_metrics_includes_new_error_and_direction_metrics():
    metrics = calculate_metrics([11.0, 9.0, 10.0], [12.0, 8.0, 11.0], [10.0, 10.0, 10.0])
    assert metrics["rmse"] > 0
    assert metrics["smape"] > 0
    assert metrics["da"] == 1.0
    assert metrics["direction_count"] == 2
    assert metrics["up_precision"] == 1.0
    assert metrics["up_recall"] == 1.0
    assert metrics["up_f1"] == 1.0
    assert metrics["down_precision"] == 1.0
    assert metrics["down_recall"] == 1.0
    assert metrics["down_f1"] == 1.0


def test_calculate_metrics_excludes_flat_actuals_from_directional_stats():
    metrics = calculate_metrics([10.0, 10.0, 12.0], [11.0, 9.0, 8.0], [10.0, 10.0, 10.0])
    assert metrics["direction_count"] == 1
    assert metrics["da"] == 0.0
    assert pandas.isna(metrics["up_precision"])
    assert metrics["up_recall"] == 0.0
    assert pandas.isna(metrics["up_f1"])
    assert metrics["down_precision"] == 0.0
    assert pandas.isna(metrics["down_recall"])
    assert pandas.isna(metrics["down_f1"])


def test_calculate_metrics_handles_missing_predicted_class():
    metrics = calculate_metrics([11.0, 9.0], [9.0, 8.0], [10.0, 10.0])
    assert metrics["da"] == 0.5
    assert pandas.isna(metrics["up_precision"])
    assert metrics["up_recall"] == 0.0
    assert pandas.isna(metrics["up_f1"])
    assert metrics["down_precision"] == 0.5
    assert metrics["down_recall"] == 1.0
    assert round(metrics["down_f1"], 6) == round(2 * 0.5 * 1.0 / 1.5, 6)


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
