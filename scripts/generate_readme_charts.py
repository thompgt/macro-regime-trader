"""Regenerate the charts embedded in README.md from a real backtest run.

Run from the repo root:

    python scripts/generate_readme_charts.py [--ticker SPY] [--start 2015-01-01]

Writes PNGs into ``images/`` and prints the metrics table to stdout so the
README's numbers can be updated from the same run that produced the charts.

Palette note: series and regime colors are taken from a categorical palette
validated for colorblind separation (worst all-pairs CVD deltaE 13.0). Notably
``sustained_bull`` is blue rather than green -- a green/red bull/bear pair is
the single most important distinction in these charts and it fails protanopia
separation, so hue is not carrying that contrast alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from macro_regime_trader.backtest.analytics import compute_metrics
from macro_regime_trader.backtest.benchmarks import buy_and_hold_equity, dma_crossover_equity
from macro_regime_trader.backtest.engine import (
    chain_walk_forward_equity,
    rebase,
    run_backtest,
    run_backtest_from,
    run_walk_forward_backtest,
)
from macro_regime_trader.config import get_settings
from macro_regime_trader.data.yfinance_provider import YFinanceProvider

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

SERIES = {
    "strategy": "#2a78d6",
    "buy_and_hold": "#eb6834",
    "dma_crossover": "#1baf7a",
}
SERIES_LABEL = {
    "strategy": "Macro regime strategy",
    "buy_and_hold": "Buy & hold",
    "dma_crossover": "200-DMA crossover",
}

REGIME_COLOR = {
    "sustained_bull": "#2a78d6",
    "volatile_distribution": "#eda100",
    "compressed_liquidity": "#4a3aa7",
    "structural_bear": "#e34948",
}
REGIME_LABEL = {
    "sustained_bull": "Sustained bull",
    "volatile_distribution": "Volatile distribution",
    "compressed_liquidity": "Compressed liquidity",
    "structural_bear": "Structural bear",
}

IMAGES_DIR = Path("images")


def _style_axes(ax: plt.Axes, *, ylabel: str = "") -> None:
    """Recessive grid and axes; the data should be the only assertive ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def _money(value: float, _pos: float = 0) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value / 1_000:.0f}k"


def _pct(value: float, _pos: float = 0) -> str:
    return f"{value * 100:.0f}%"


def _label_line_ends(
    ax: plt.Axes, entries: list[tuple[pd.Series, str, str]], min_gap: float = 0.055
) -> None:
    """Direct-label each series at its right end, pushing labels apart if needed.

    Required relief, not decoration: two of these palette slots sit below 3:1
    contrast on this surface, so identity must not rest on the legend swatch
    alone. Curves that finish at similar values would collide, so labels are
    placed in axes-fraction space and separated to a minimum gap.
    """
    # Go through the real data->display->axes chain rather than transLimits,
    # which is only linear-scale correct and silently misplaces labels on a
    # log axis.
    to_axes = ax.transAxes.inverted()
    placed: list[tuple[float, str, str, pd.Series]] = []
    for series, color, text in entries:
        clean = series.dropna()
        display = ax.transData.transform((ax.get_xlim()[1], clean.iloc[-1]))
        y_axes = float(to_axes.transform(display)[1])
        placed.append((y_axes, color, text, clean))

    placed.sort(key=lambda item: item[0])
    positions = [item[0] for item in placed]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap

    # Spreading can push the top label past the axes; slide the whole group back
    # inside rather than letting labels float above the plot.
    overshoot = positions[-1] - 1.0
    if overshoot > 0:
        positions = [p - overshoot for p in positions]

    for position, (_y, color, text, _clean) in zip(positions, placed, strict=True):
        ax.annotate(
            text,
            xy=(1.0, position),
            xycoords="axes fraction",
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="600",
            va="center",
            ha="left",
            annotation_clip=False,
        )


def _regime_runs(regimes: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Collapse a per-bar regime series into contiguous (start, end, regime) spans."""
    valid = regimes.dropna()
    if valid.empty:
        return []
    runs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current = valid.iloc[0]
    start = valid.index[0]
    prev = valid.index[0]
    for timestamp, value in valid.items():
        if value != current:
            runs.append((start, prev, str(current)))
            current, start = value, timestamp
        prev = timestamp
    runs.append((start, prev, str(current)))
    return runs


def chart_equity_and_drawdown(curves: pd.DataFrame, metrics: dict, out: Path) -> None:
    """Headline chart: equity curves plus the drawdown they were earned through."""
    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(11, 7.6),
        height_ratios=[2.4, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    for name in ("dma_crossover", "buy_and_hold", "strategy"):
        series = curves[name].dropna()
        ax_eq.plot(
            series.index,
            series.values,
            color=SERIES[name],
            linewidth=2.2 if name == "strategy" else 1.5,
            zorder=3 if name == "strategy" else 2,
            solid_capstyle="round",
        )

    _style_axes(ax_eq, ylabel="Portfolio value")
    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(FuncFormatter(_money))
    ax_eq.set_yticks([100_000, 200_000, 300_000, 400_000, 500_000])
    _label_line_ends(
        ax_eq,
        [(curves[name], SERIES[name], SERIES_LABEL[name]) for name in SERIES],
    )

    fig.text(
        0.075,
        0.965,
        "Regime strategy vs. buy & hold",
        color=INK,
        fontsize=15,
        fontweight="600",
        va="top",
    )
    fig.text(
        0.075,
        0.925,
        f"{metrics['ticker']} daily bars, {metrics['start']} to {metrics['end']} "
        f"- net of slippage and financing. Log scale, so equal vertical "
        f"distance is equal percentage gain.",
        color=INK_SECONDARY,
        fontsize=9.5,
        va="top",
    )

    # Only the strategy and the benchmark it must beat: a third drawdown trace
    # turns this panel into noise without answering the question it exists for.
    for name in ("buy_and_hold", "strategy"):
        series = curves[name].dropna()
        drawdown = series / series.cummax() - 1.0
        ax_dd.plot(
            drawdown.index,
            drawdown.values,
            color=SERIES[name],
            linewidth=1.8 if name == "strategy" else 1.4,
            zorder=3 if name == "strategy" else 2,
        )
    strategy_dd = curves["strategy"] / curves["strategy"].cummax() - 1.0
    ax_dd.fill_between(
        strategy_dd.index, strategy_dd.values, 0, color=SERIES["strategy"], alpha=0.10, zorder=1
    )

    _style_axes(ax_dd, ylabel="Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax_dd.axhline(0, color=GRID, linewidth=0.8)
    ax_dd.annotate(
        "Drawdown from running peak (strategy vs. buy & hold)",
        xy=(0, 1.06),
        xycoords="axes fraction",
        color=INK_SECONDARY,
        fontsize=9.5,
        va="bottom",
    )

    handles = [
        Line2D([], [], color=SERIES[name], linewidth=2.2, label=SERIES_LABEL[name])
        for name in ("strategy", "buy_and_hold", "dma_crossover")
    ]
    legend = ax_dd.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=9,
        ncol=3,
        bbox_to_anchor=(0, -0.46),
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.subplots_adjust(left=0.075, right=0.80, top=0.875, bottom=0.135)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_regime_timeline(
    close: pd.Series, regimes: pd.Series, exposure: pd.Series, out: Path
) -> None:
    """Price, the regime ribbon beneath it, and the exposure it produced.

    The regime is drawn as a compact ribbon rather than as translucent shading
    across the price panel: regimes flip from bar to bar, and full-height bands
    at that frequency wash out into stripes that obscure the price instead of
    annotating it.
    """
    fig, (ax_px, ax_rib, ax_ex) = plt.subplots(
        3,
        1,
        figsize=(11, 7.6),
        height_ratios=[2.4, 0.22, 1.15],
        sharex=True,
        gridspec_kw={"hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)

    ax_px.plot(close.index, close.values, color=INK, linewidth=1.4, zorder=4)

    # Mark the risk-off stretches on the price panel only -- one regime, so the
    # bands stay legible and carry a single clear meaning.
    for start, end, regime in _regime_runs(regimes):
        if regime == "structural_bear":
            ax_px.axvspan(start, end, color=REGIME_COLOR[regime], alpha=0.16, lw=0, zorder=1)
    _style_axes(ax_px, ylabel="Close")
    fig.text(
        0.075,
        0.965,
        "What the regime engine sees, and what it does about it",
        color=INK,
        fontsize=15,
        fontweight="600",
        va="top",
    )
    fig.text(
        0.075,
        0.925,
        "Red bands are the risk-off regime. The ribbon below the price is the "
        "full classification; exposure is what the sizer did with it.",
        color=INK_SECONDARY,
        fontsize=9.5,
        va="top",
    )

    # Regime ribbon: one row of solid color per bar.
    codes = pd.Categorical(
        regimes.astype(object).where(regimes.notna(), None),
        categories=list(REGIME_COLOR),
    ).codes.astype(float)
    codes[codes < 0] = float("nan")
    x_numeric = mdates.date2num(regimes.index.to_pydatetime())
    ax_rib.imshow(
        codes.reshape(1, -1),
        aspect="auto",
        interpolation="nearest",
        extent=(x_numeric[0], x_numeric[-1], 0.0, 1.0),
        cmap=ListedColormap(list(REGIME_COLOR.values())),
        vmin=-0.5,
        vmax=len(REGIME_COLOR) - 0.5,
    )
    ax_rib.set_yticks([])
    ax_rib.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax_rib.spines[side].set_visible(False)
    ax_rib.tick_params(labelsize=9, length=0, colors=INK_SECONDARY)
    ax_rib.set_ylabel(
        "Regime", color=INK_SECONDARY, fontsize=9, rotation=0, ha="right", va="center"
    )

    ax_ex.plot(exposure.index, exposure.values, color=SERIES["strategy"], linewidth=1.6, zorder=4)
    ax_ex.fill_between(
        exposure.index, exposure.values, 0, color=SERIES["strategy"], alpha=0.14, zorder=3
    )
    ax_ex.axhline(1.0, color=INK_MUTED, linewidth=1.0, dashes=(4, 3), zorder=5)
    _style_axes(ax_ex, ylabel="Exposure")
    ax_ex.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}x"))
    ax_ex.set_ylim(0, max(1.75, float(exposure.max()) * 1.08))

    # Solid swatches so the legend matches the ribbon's saturation exactly.
    handles = [
        Patch(facecolor=color, label=REGIME_LABEL[key]) for key, color in REGIME_COLOR.items()
    ]
    legend = ax_ex.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=9,
        ncol=4,
        bbox_to_anchor=(0, -0.46),
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.subplots_adjust(left=0.075, right=0.97, top=0.875, bottom=0.145)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_walkforward(
    full_sample: pd.Series, walk_forward: pd.Series, benchmark: pd.Series, out: Path
) -> None:
    """Full-sample vs chained out-of-sample equity, against the benchmark."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.patch.set_facecolor(SURFACE)

    entries = [
        (benchmark, SERIES["buy_and_hold"], "Buy & hold", 1.8, (0, ())),
        (full_sample, SERIES["strategy"], "Strategy (full sample)", 2.4, (0, ())),
        (walk_forward, SERIES["dma_crossover"], "Strategy (walk-forward OOS)", 2.2, (0, (5, 2))),
    ]
    for series, color, _label, width, dash in entries:
        clean = series.dropna()
        ax.plot(
            clean.index,
            clean.values,
            color=color,
            linewidth=width,
            linestyle=dash,
            solid_capstyle="round",
        )

    _style_axes(ax, ylabel="Portfolio value")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_yticks([100_000, 200_000, 300_000, 400_000, 500_000])
    _label_line_ends(ax, [(s, c, label) for s, c, label, _w, _d in entries], min_gap=0.075)

    fig.text(
        0.075,
        0.965,
        "Out-of-sample holds up",
        color=INK,
        fontsize=15,
        fontweight="600",
        va="top",
    )
    fig.text(
        0.075,
        0.905,
        "Walk-forward reports only bars outside each window's own warmup, with "
        "every window's returns compounded onto the last.",
        color=INK_SECONDARY,
        fontsize=9.5,
        va="top",
    )

    handles = [
        Line2D([], [], color=color, linewidth=w, linestyle=d, label=label)
        for _s, color, label, w, d in entries
    ]
    legend = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.subplots_adjust(left=0.075, right=0.755, top=0.83, bottom=0.09)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _markdown_table(rows: dict[str, dict[str, float]]) -> str:
    columns = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    lines = [
        "| model | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for name, metrics in rows.items():
        cells = " | ".join(f"{metrics[c]:.4f}" for c in columns)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    settings = get_settings()
    provider = YFinanceProvider(cache_dir=settings.data_cache_dir)

    evaluation_start = pd.Timestamp(args.start)
    fetch_start = evaluation_start - pd.Timedelta(days=settings.warmup_calendar_days)
    print(f"Fetching {args.ticker} from {fetch_start.date()} (warmup) ...")
    ohlcv = provider.get_ohlcv(
        args.ticker, start=fetch_start.strftime("%Y-%m-%d"), end=args.end, interval="1d"
    )

    result = run_backtest_from(ohlcv, evaluation_start, settings)
    index = result.equity_curve.index
    evaluated = ohlcv.loc[index]

    curves = pd.DataFrame(
        {
            "strategy": result.equity_curve,
            "buy_and_hold": buy_and_hold_equity(evaluated, settings),
            "dma_crossover": dma_crossover_equity(evaluated, settings),
        }
    )

    rows = {name: compute_metrics(curves[name].dropna()) for name in curves.columns}

    IMAGES_DIR.mkdir(exist_ok=True)
    meta = {
        "ticker": args.ticker,
        "start": str(index[0].date()),
        "end": str(index[-1].date()),
    }

    chart_equity_and_drawdown(curves, meta, IMAGES_DIR / "equity_curves.png")
    chart_regime_timeline(
        evaluated["close"],
        result.regimes,
        result.exposure,
        IMAGES_DIR / "regime_timeline.png",
    )

    print("Running walk-forward ...")
    windows = run_walk_forward_backtest(ohlcv, settings)
    walk_forward = chain_walk_forward_equity(windows, settings.starting_balance)
    wf_index = walk_forward.index

    # All three curves must span the same bars and be rebased at the same point,
    # or the chart compares differently-dated runs. The walk-forward span starts
    # earlier than the reported evaluation window, so run the full-sample
    # simulation over everything and slice it to the walk-forward index.
    full_sample_all = run_backtest(ohlcv, settings).equity_curve
    chart_walkforward(
        rebase(full_sample_all.loc[wf_index], settings.starting_balance),
        walk_forward,
        rebase(buy_and_hold_equity(ohlcv.loc[wf_index], settings), settings.starting_balance),
        IMAGES_DIR / "walkforward_vs_fullsample.png",
    )

    wf_rows = {
        "strategy_walk_forward": compute_metrics(walk_forward),
        "buy_and_hold": compute_metrics(buy_and_hold_equity(ohlcv.loc[wf_index], settings)),
        "dma_crossover": compute_metrics(dma_crossover_equity(ohlcv.loc[wf_index], settings)),
    }

    print(f"\nEvaluated {len(index)} bars: {meta['start']} -> {meta['end']}")
    print(f"Mean exposure: {result.exposure.mean():.2f}x")
    print(f"Trades: {int((result.ledger['side'] != 'hold').sum())}")
    print("\n== Full sample ==")
    print(_markdown_table(rows))
    print(f"\n== Walk-forward ({len(windows)} OOS windows) ==")
    print(_markdown_table(wf_rows))
    print(f"\nWrote charts to {IMAGES_DIR.resolve()}")


if __name__ == "__main__":
    main()
