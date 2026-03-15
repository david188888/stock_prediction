import pandas

from stock_prediction.evaluation.reporting import save_conclusion_markdown, save_summary_markdown


def test_summary_markdown_includes_aggregated_metrics(tmp_path):
    frame = pandas.DataFrame(
        [
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "holdout", "mae": 1.0, "mse": 1.0, "rmse": 1.0},
            {"symbol": "BBB", "model": "linear_regression", "evaluation": "holdout", "mae": 2.0, "mse": 4.0, "rmse": 2.0},
            {"symbol": "AAA", "model": "arima", "evaluation": "walk_forward", "mae": 1.5, "mse": 2.25, "rmse": 1.5},
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
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "holdout", "mae": 1.0, "mse": 1.0, "rmse": 1.0},
            {"symbol": "AAA", "model": "linear_regression", "evaluation": "walk_forward", "mae": 1.1, "mse": 1.21, "rmse": 1.1},
            {"symbol": "AAA", "model": "arima", "evaluation": "holdout", "mae": 2.0, "mse": 4.0, "rmse": 2.0},
            {"symbol": "AAA", "model": "arima", "evaluation": "walk_forward", "mae": 2.2, "mse": 4.84, "rmse": 2.2},
        ]
    )
    path = tmp_path / "conclusion.md"
    save_conclusion_markdown(path, frame)
    content = path.read_text(encoding="utf-8")
    assert "holdout" in content
    assert "linear_regression" in content
