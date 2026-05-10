"""
plot_style.py — Shared figure style for SAT IRD Paper
======================================================
Import this at the top of every figure script:

    from plot_style import apply_style, COLORS, SEASON_BANDS, add_season_bands

All figures will share fonts, sizes, colors, and layout conventions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    # Primary palette — Ocean Gradient
    "deep_blue"   : "#065A82",
    "teal"        : "#1C7293",
    "light_blue"  : "#9DCBDA",
    "pale_blue"   : "#D6EAF3",

    # Accent
    "orange"      : "#E07B39",
    "red"         : "#C0392B",
    "green"       : "#27AE60",
    "purple"      : "#7D3C98",
    "gold"        : "#D4AC0D",

    # Neutrals
    "dark_gray"   : "#2C3E50",
    "mid_gray"    : "#7F8C8D",
    "light_gray"  : "#BDC3C7",
    "near_white"  : "#F4F6F7",

    # Flagging
    "outlier_red" : "#E74C3C",
    "held_out"    : "#8E44AD",
    "naive"       : "#95A5A6",
}

# Field colors — consistent across all figures
FIELD_COLORS = {
    "Soreq 2" : "#065A82",   # deep blue
    "Yavne 1" : "#1C7293",   # teal
    "Yavne 2" : "#E07B39",   # orange
    "Yavne 3" : "#27AE60",   # green
    "Yavne 4" : "#7D3C98",   # purple
}

# Held-out basin markers — one per field, consistent across M1 and M2 figures
HELD_OUT_MARKERS = {
    3203: ("o", FIELD_COLORS["Soreq 2"], "Basin 3203 — Soreq 2"),
    4104: ("s", FIELD_COLORS["Yavne 1"], "Basin 4104 — Yavne 1"),
    5102: ("^", FIELD_COLORS["Yavne 2"], "Basin 5102 — Yavne 2"),
    6303: ("D", FIELD_COLORS["Yavne 3"], "Basin 6303 — Yavne 3"),
    7201: ("P", FIELD_COLORS["Yavne 4"], "Basin 7201 — Yavne 4"),
}

# Split colors (train/val/test) — used in per-basin diagnostic plots
SPLIT_COLORS = {
    "train" : COLORS["light_gray"],
    "val"   : COLORS["teal"],
    "test"  : COLORS["deep_blue"],
}

# Outlier type colors
OUTLIER_TYPE_COLORS = {
    "Type1" : COLORS["orange"],
    "Type3" : COLORS["outlier_red"],
    "clean" : COLORS["teal"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL BACKGROUND BANDS
# Used on any figure with calendar time on x-axis (10-year time series)
# ─────────────────────────────────────────────────────────────────────────────

# Season definitions (Israel / Mediterranean)
# Summer: Jun–Aug (peak clogging, high radiation)
# Winter: Dec–Feb (low radiation, rain)
# Spring/Autumn: transitions
SEASON_COLORS = {
    "Summer" : ("#FFF3CD", 0.45),   # warm yellow, semi-transparent
    "Autumn" : ("#D5E8D4", 0.35),   # soft green
    "Winter" : ("#DAE8FC", 0.40),   # cool blue
    "Spring" : ("#FCE4D6", 0.35),   # soft peach
}

# Month → season mapping (1-indexed)
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring",  4: "Spring", 5: "Spring",
    6: "Summer",  7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def add_season_bands(ax, date_min, date_max):
    """
    Add seasonal background color bands to a time-series axis.

    Parameters
    ----------
    ax        : matplotlib Axes with datetime x-axis
    date_min  : start date (datetime or Timestamp)
    date_max  : end date   (datetime or Timestamp)

    Usage
    -----
        add_season_bands(ax, df["date"].min(), df["date"].max())

    Returns a legend patch list — add to legend if desired:
        patches = add_season_bands(ax, ...)
        ax.legend(handles=existing_handles + patches, ...)
    """
    import pandas as pd
    from matplotlib.dates import date2num

    # Build year range
    y_start = pd.Timestamp(date_min).year
    y_end   = pd.Timestamp(date_max).year + 1

    season_spans = {
        "Winter" : [(12, 1), (2, 28)],   # Dec–Feb (crosses year boundary)
        "Spring" : [(3, 1),  (5, 31)],
        "Summer" : [(6, 1),  (8, 31)],
        "Autumn" : [(9, 1),  (11, 30)],
    }

    patches_added = set()

    for year in range(y_start, y_end):
        for season, ((sm, sd), (em, ed)) in season_spans.items():
            color, alpha = SEASON_COLORS[season]

            if season == "Winter":
                # Winter spans Dec of previous year → Feb of current year
                s = pd.Timestamp(year - 1, 12, 1)
                e = pd.Timestamp(year,      2, 28)
            else:
                s = pd.Timestamp(year, sm, sd)
                e = pd.Timestamp(year, em, ed)

            # Clip to data range
            s = max(s, pd.Timestamp(date_min))
            e = min(e, pd.Timestamp(date_max))
            if s >= e:
                continue

            ax.axvspan(s, e, facecolor=color, alpha=alpha, zorder=0,
                       label=season if season not in patches_added else None)
            patches_added.add(season)

    # Return legend patches for seasons actually plotted
    legend_patches = [
        mpatches.Patch(
            facecolor=SEASON_COLORS[s][0],
            alpha=SEASON_COLORS[s][1] + 0.2,
            label=s
        )
        for s in ["Winter", "Spring", "Summer", "Autumn"]
        if s in patches_added
    ]
    return legend_patches


# ─────────────────────────────────────────────────────────────────────────────
# FONT & SIZE SYSTEM
# Minimum body text: 16pt (on screen); journal figures: scale down uniformly
# ─────────────────────────────────────────────────────────────────────────────

FONT = {
    "family"       : "serif",          # matches journal convention (Times-like)
    "title"        : 9,               # panel title / suptitle
    "axis_label"   : 12,               # x/y axis labels  ← minimum for reviewers
    "tick"         : 12,               # tick labels
    "legend"       : 9,               # legend text
    "annotation"   : 9,               # in-plot text (R², RMSE, etc.)
    "caption"      : 9,               # figure caption (not rendered in matplotlib)
    "small"        : 10,               # secondary annotations
}


def apply_style():
    """
    Call once at the top of each figure script.
    Sets global rcParams — affects all subsequent plt calls.
    """
    plt.rcParams.update({
        # Font family
        "font.family"           : "serif",
        "font.serif"            : ["Times New Roman", "DejaVu Serif", "serif"],

        # Font sizes
        "font.size"             : FONT["axis_label"],
        "axes.titlesize"        : FONT["title"],
        "axes.labelsize"        : FONT["axis_label"],
        "xtick.labelsize"       : FONT["tick"],
        "ytick.labelsize"       : FONT["tick"],
        "legend.fontsize"       : FONT["legend"],
        "figure.titlesize"      : FONT["title"],

        # Lines & markers
        "lines.linewidth"       : 2.0,
        "lines.markersize"      : 7,
        "patch.linewidth"       : 1.2,

        # Axes appearance
        "axes.spines.top"       : False,
        "axes.spines.right"     : False,
        "axes.grid"             : True,
        "grid.alpha"            : 0.25,
        "grid.linestyle"        : "--",
        "grid.linewidth"        : 0.6,
        "axes.axisbelow"        : True,

        # Figure
        "figure.dpi"            : 150,
        "savefig.dpi"           : 300,
        "savefig.bbox"          : "tight",
        "figure.facecolor"      : "white",
        "axes.facecolor"        : "white",

        # Legend
        "legend.framealpha"     : 0.85,
        "legend.edgecolor"      : COLORS["light_gray"],
        "legend.frameon"        : True,
    })


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE SIZE PRESETS  (width, height) in inches
# Water Research: single-col = 8.5cm ≈ 3.35", double-col = 17.5cm ≈ 6.89"
# ─────────────────────────────────────────────────────────────────────────────

FIG_SIZES = {
    "single_col"      : (3.35, 2.8),    # journal single-column
    "double_col"      : (6.89, 4.5),    # journal double-column
    "double_col_tall" : (6.89, 6.0),    # for 2-row panel figures
    "wide_screen"     : (12.0, 5.0),    # presentation / PPT export
    "square"          : (5.5,  5.5),    # scatter plots
    "five_panel_row"  : (14.0, 3.5),    # 5 held-out basins side by side (M1/M2)
    "five_panel_2row" : (14.0, 7.0),    # 5 basins × 2 row (time series + scatter)
    "three_panel_row" : (12.0, 4.0),    # Fig 2: 3 representative basins
}


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def annotate_metrics(ax, r2, rmse, mape, n=None, loc="upper left", fontsize=None):
    """
    Add a standard metrics box (R², RMSE, MAPE) to an axes.

    Parameters
    ----------
    ax       : matplotlib Axes
    r2       : float
    rmse     : float  (cm/h)
    mape     : float  (%)
    n        : int or None
    loc      : "upper left" | "upper right" | "lower right" | "lower left"
    fontsize : override FONT["annotation"] if provided
    """
    fs = fontsize or FONT["annotation"]
    lines = [
        f"$R^2$ = {r2:+.3f}",
        f"RMSE = {rmse:.3f} cm/h",
        f"MAPE = {mape:.1f}%",
    ]
    if n is not None:
        lines.append(f"$n$ = {n:,}")
    text = "\n".join(lines)

    xy_map = {
        "upper left"  : (0.04, 0.96),
        "upper right" : (0.96, 0.96),
        "lower right" : (0.96, 0.04),
        "lower left"  : (0.04, 0.04),
    }
    va_map = {
        "upper left"  : "top",
        "upper right" : "top",
        "lower right" : "bottom",
        "lower left"  : "bottom",
    }
    ha_map = {
        "upper left"  : "left",
        "upper right" : "right",
        "lower right" : "right",
        "lower left"  : "left",
    }

    ax.annotate(
        text,
        xy=xy_map[loc], xycoords="axes fraction",
        fontsize=fs, va=va_map[loc], ha=ha_map[loc],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=COLORS["light_gray"],
            alpha=0.90,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1:1 LINE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def add_unity_line(ax, color=None, lw=1.5):
    """Add a 1:1 dashed line that spans the current axis limits."""
    color = color or COLORS["dark_gray"]
    lims  = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color=color, linewidth=lw, alpha=0.6, zorder=1)
    ax.set_xlim(lims); ax.set_ylim(lims)


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT COLOR CYCLE  (for coloring segments within a basin)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.cm as cm

def segment_colors(n_segments):
    """
    Return a list of n_segments colors using the 'viridis' colormap.
    Used in Figure 2 to color each flooding segment distinctly.
    """
    cmap = plt.colormaps["viridis"].resampled(max(n_segments, 2))
    return [cmap(i / max(n_segments - 1, 1)) for i in range(n_segments)]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES["double_col"])

    # Panel A — scatter
    ax = axes[0]
    np.random.seed(42)
    x = np.random.uniform(0.5, 5, 80)
    y = x + np.random.normal(0, 0.5, 80)
    ax.scatter(x, y, color=COLORS["deep_blue"], alpha=0.7, s=40)
    add_unity_line(ax)
    annotate_metrics(ax, r2=0.842, rmse=0.512, mape=14.3, n=80)
    ax.set_xlabel("IRD actual (cm/h)")
    ax.set_ylabel("IRD predicted (cm/h)")
    ax.set_title("Panel A — scatter example")

    # Panel B — time series with season bands
    import pandas as pd
    ax = axes[1]
    dates = pd.date_range("2018-01-01", "2022-12-31", freq="W")
    y2    = np.sin(np.linspace(0, 8 * np.pi, len(dates))) * 1.5 + 3
    season_patches = add_season_bands(ax, dates[0], dates[-1])
    ax.plot(dates, y2, color=COLORS["deep_blue"], linewidth=1.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("IRD (cm/h)")
    ax.set_title("Panel B — season bands example")
    # ax.legend(handles=season_patches, fontsize=FONT["small"],
    #           loc="upper right", title="Season")

    fig.suptitle("plot_style.py — Self-test", fontsize=FONT["title"])
    plt.tight_layout()
    plt.show()
    print("plot_style.py self-test passed.")