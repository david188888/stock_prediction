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


def save_summary_markdown(path: Path, frame) -> None:
    ensure_dir(path.parent)
    lines = ["# Experiment Summary", ""]
    lines.append("| " + " | ".join(frame.columns.astype(str)) + " |")
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_prediction_plot(path: Path, frame, title: str) -> None:
    cache_root = path.parent.parent / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    try:
        plt = require_dependency("matplotlib.pyplot", "Run `uv sync` to install runtime dependencies.")
    except RuntimeError:
        return
    ensure_dir(path.parent)
    plt.figure(figsize=(10, 4))
    plt.plot(frame["date"], frame["actual"], label="actual")
    plt.plot(frame["date"], frame["predicted"], label="predicted")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("close")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
