from __future__ import annotations

import os
from pathlib import Path

from stock_prediction.utils.dependencies import require_dependency
from stock_prediction.utils.io import ensure_dir


def save_predictions(path: Path, frame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def save_metrics(path: Path, frame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def save_training_log(path: Path, frame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def _render_markdown_table(frame) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(frame.columns.astype(str)) + " |")
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return lines


def _aggregate_metrics(frame):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    if frame.empty or "evaluation" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["rmse", "mae"])
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for (evaluation, model), group in frame.groupby(["evaluation", "model"], dropna=False):
        ranked = group.sort_values("rmse")
        rows.append(
            {
                "evaluation": evaluation,
                "model": model,
                "count": len(group),
                "mean_rmse": round(float(group["rmse"].mean()), 6),
                "median_rmse": round(float(group["rmse"].median()), 6),
                "mean_mae": round(float(group["mae"].mean()), 6),
                "median_mae": round(float(group["mae"].median()), 6),
                "mean_da": round(float(group["da"].dropna().mean()), 6) if "da" in group.columns and group["da"].notna().any() else pd.NA,
                "best_symbol": ranked.iloc[0]["symbol"],
                "worst_symbol": ranked.iloc[-1]["symbol"],
            }
        )
    summary = pd.DataFrame(rows).sort_values(["evaluation", "mean_rmse", "median_rmse"]).reset_index(drop=True)
    summary["rank"] = summary.groupby("evaluation")["mean_rmse"].rank(method="dense").astype("Int64")
    return summary[
        ["evaluation", "rank", "model", "count", "mean_rmse", "median_rmse", "mean_mae", "median_mae", "mean_da", "best_symbol", "worst_symbol"]
    ]


def save_summary_markdown(path: Path, frame) -> None:
    ensure_dir(path.parent)
    lines = ["# Experiment Summary", ""]
    aggregated = _aggregate_metrics(frame)
    if not aggregated.empty:
        lines.extend(["## Aggregated Metrics", ""])
        lines.extend(_render_markdown_table(aggregated))
        lines.append("")
    lines.extend(["## Detailed Results", ""])
    lines.extend(_render_markdown_table(frame))
    path.write_text("\n".join(lines), encoding="utf-8")


def save_conclusion_markdown(path: Path, frame) -> None:
    ensure_dir(path.parent)
    aggregated = _aggregate_metrics(frame)
    lines = ["# Conclusions", ""]
    if aggregated.empty:
        lines.append("No aggregated metrics available.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    best_by_eval = aggregated.sort_values(["evaluation", "mean_rmse", "median_rmse"]).groupby("evaluation").head(1)
    for _, row in best_by_eval.iterrows():
        lines.append(
            f"- `{row['evaluation']}` 最优模型是 `{row['model']}`，平均 RMSE 为 `{row['mean_rmse']}`，方向准确率为 `{row['mean_da']}`，最佳股票为 `{row['best_symbol']}`。"
        )

    holdout_best = best_by_eval[best_by_eval["evaluation"] == "holdout"]["model"].tolist()
    walk_best = best_by_eval[best_by_eval["evaluation"] == "walk_forward"]["model"].tolist()
    if holdout_best and walk_best:
        consistency = "一致" if holdout_best[0] == walk_best[0] else "不一致"
        lines.append(f"- holdout 与 walk-forward 的最优模型{consistency}。")

    if {"arima", "arima_residual_lstm"}.issubset(set(frame["model"].unique())):
        hybrid = frame[frame["model"] == "arima_residual_lstm"].set_index(["symbol", "evaluation"])
        arima = frame[frame["model"] == "arima"].set_index(["symbol", "evaluation"])
        overlap = hybrid.join(arima, lsuffix="_hybrid", rsuffix="_arima", how="inner")
        if not overlap.empty:
            delta = float((overlap["rmse_hybrid"] - overlap["rmse_arima"]).mean())
            lines.append(f"- hybrid 相对 ARIMA 的平均 RMSE 改变量为 `{delta:.6f}`。负值表示 hybrid 更优。")

    deep_models = frame[frame["model"].isin(["lstm", "gru"])]
    if not deep_models.empty:
        worst_cases = deep_models.sort_values("rmse", ascending=False).head(3)
        symbols = ", ".join(worst_cases["symbol"].astype(str).tolist())
        lines.append(f"- 深度模型当前的高误差样本主要集中在 `{symbols}`，优先检查特征稳定性与样本尺度。")

    path.write_text("\n".join(lines), encoding="utf-8")


def _plt():
    cache_root = Path("outputs") / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    return require_dependency("matplotlib.pyplot", "Run `uv sync` to install runtime dependencies.")


def save_model_metric_bar_chart(path: Path, frame, title: str) -> None:
    if frame.empty:
        return
    aggregated = _aggregate_metrics(frame)
    if aggregated.empty:
        return
    plt = _plt()
    ensure_dir(path.parent)
    pivot = aggregated.pivot(index="model", columns="evaluation", values="mean_rmse").fillna(0)
    pivot.plot(kind="bar", figsize=(10, 5), title=title)
    plt.ylabel("mean RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_top_bottom_plot(path: Path, frame, title: str) -> None:
    if frame.empty:
        return
    plt = _plt()
    ensure_dir(path.parent)
    ranked = frame[frame["evaluation"] == "holdout"].sort_values("rmse")
    if ranked.empty:
        ranked = frame.sort_values("rmse")
    focus = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.").concat(
        [ranked.head(5), ranked.tail(5)],
        ignore_index=True,
    )
    labels = focus["model"].astype(str) + ":" + focus["symbol"].astype(str)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, focus["rmse"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_walk_forward_error_plot(path: Path, frame, title: str) -> None:
    if frame.empty:
        return
    plt = _plt()
    ensure_dir(path.parent)
    frame = frame.copy()
    frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
    grouped = frame.groupby("step", dropna=False)["abs_error"].mean().reset_index()
    plt.figure(figsize=(10, 4))
    plt.plot(grouped["step"], grouped["abs_error"], marker="o")
    plt.xlabel("walk-forward step")
    plt.ylabel("mean absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _safe_model_column_name(model_name: str) -> str:
    return model_name.replace("-", "_")


def _annotate_extreme_volatility(frame, quantile: float):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    if frame.empty:
        return frame

    annotated = frame.copy()
    annotated["abs_return_1d"] = ((annotated["actual_close"] - annotated["current_close"]) / annotated["current_close"]).abs()
    thresholds = annotated.groupby("symbol", dropna=False)["abs_return_1d"].transform(lambda series: series.quantile(quantile))
    annotated["is_extreme_volatility"] = annotated["abs_return_1d"] >= thresholds
    annotated["event_id"] = pd.NA

    for (evaluation, symbol), group in annotated.groupby(["evaluation", "symbol"], sort=False, dropna=False):
        event_index = 0
        previous_flag = False
        event_ids: list[object] = []
        for flag in group["is_extreme_volatility"].tolist():
            flag = bool(flag)
            if flag:
                if not previous_flag:
                    event_index += 1
                event_ids.append(f"{symbol}-{evaluation}-EV{event_index:03d}")
            else:
                event_ids.append(pd.NA)
            previous_flag = flag
        annotated.loc[group.index, "event_id"] = event_ids

    return annotated


def build_model_comparison_frame(frame, *, model_order: list[str], extreme_volatility_quantile: float):
    pd = require_dependency("pandas", "Run `uv sync` to install runtime dependencies.")
    if frame.empty:
        return pd.DataFrame()

    id_columns = [
        "evaluation",
        "symbol",
        "date",
        "feature_date",
        "target_date",
        "current_close",
        "actual_close",
        "split_start_date",
    ]
    base = frame[id_columns].drop_duplicates().copy()

    for model_name in model_order:
        model_frame = frame[frame["model"] == model_name].copy()
        if model_frame.empty:
            continue
        safe_name = _safe_model_column_name(model_name)
        renamed = model_frame[id_columns + ["predicted", "abs_error", "direction_correct"]].rename(
            columns={
                "predicted": f"pred_{safe_name}",
                "abs_error": f"abs_error_{safe_name}",
                "direction_correct": f"dir_correct_{safe_name}",
            }
        )
        base = base.merge(renamed, on=id_columns, how="left")

    base = base.sort_values(["evaluation", "symbol", "target_date", "feature_date"]).reset_index(drop=True)
    return _annotate_extreme_volatility(base, quantile=extreme_volatility_quantile)
