

import calendar
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


import matplotlib.ticker as ticker
from collections import defaultdict



# input:  
#   data      : dict returned by load_workout_directory()
#   keys      : "W-NNN-X", values: parsed workout dicts
#   save_path : str | None — if given, saves PNG to that path
#
# output: 
#   matplotlib.figure.Figure
#   A calendar heatmap where each day cell is coloured by
#   session type(s) logged that day (A= blue, B= orange, C= green)
#   Cells with no session are empty (light grey)

def plot_workout_heatmap(data: dict, save_path: str | None = None):
    """
        render a GitHub-style calendar heatmap of all workout sessions
    """

    # 1. Collect (date, session_type) pairs
    sessions: list[tuple[date, str]] = []
    for w in data.values():
        if w["date_str"] and w["session_type"]:
            d = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
            sessions.append((d, w["session_type"]))

    if not sessions:
        raise ValueError("No sessions with valid dates found in data.")

    # 2. Determine date range (full years)
    min_date = min(d for d, _ in sessions)
    max_date = max(d for d, _ in sessions)
    start = date(min_date.year, 1, 1)
    end   = date(max_date.year, 12, 31)

    # 3. Build lookup: date => list of session types
    from collections import defaultdict
    day_sessions: dict[date, list[str]] = defaultdict(list)
    for d, stype in sessions:
        day_sessions[d].append(stype)

    # 4. Color map
    TYPE_COLOR = {"A": "#4C9BE8", "B": "#F5A623", "C": "#57C26A"}
    EMPTY_COLOR = "#EEEEEE"

    def cell_color(day: date) -> str:
        types = day_sessions.get(day, [])
        if not types:
            return EMPTY_COLOR
        # If multiple types logged same day, pick A > B > C priority
        for t in ("A", "B", "C"):
            if t in types:
                return TYPE_COLOR[t]
        return EMPTY_COLOR

    # 5. Layout: one row per year, columns = week of year
    years = list(range(start.year, end.year + 1))
    n_years = len(years)

    fig, axes = plt.subplots(
        n_years, 1,
        figsize=(20, 2.2 * n_years + 0.8),
        facecolor="white"
    )
    if n_years == 1:
        axes = [axes]

    DAYS   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for ax, year in zip(axes, years):
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, 53.5)
        ax.set_ylim(-0.5, 7.5)
        ax.invert_yaxis()
        ax.set_yticks(range(7))
        ax.set_yticklabels(DAYS, fontsize=8)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(str(year), loc="left", fontsize=11, fontweight="bold", pad=6)

        # Month label positions
        month_week_starts = {}
        for month in range(1, 13):
            first = date(year, month, 1)
            wk = first.isocalendar()[1]
            # week 1 of following year wraps to 0
            if month == 1:
                wk = 0
            month_week_starts[month] = wk

        for month, wk in month_week_starts.items():
            ax.text(wk, 0, MONTHS[month - 1],
                    fontsize=7.5, color="#666666", va="bottom")

        # Draw squares
        for month in range(1, 13):
            _, n_days = calendar.monthrange(year, month)
            for day_num in range(1, n_days + 1):
                d = date(year, month, day_num)
                iso = d.isocalendar()          # (year, week, weekday)
                week_col = iso[1] - 1          # 0-indexed column
                if month == 1 and iso[1] > 50: # last week of prev year
                    week_col = 0
                dow = d.weekday()              # 0=Mon … 6=Sun
                color = cell_color(d)
                rect = mpatches.FancyBboxPatch(
                    (week_col + 0.07, dow + 0.07),
                    0.86, 0.86,
                    boxstyle="round,pad=0.05",
                    linewidth=0,
                    facecolor=color
                )
                ax.add_patch(rect)

    # 6. Legend
    legend_handles = [
        mpatches.Patch(color=c, label=f"Session {t}")
        for t, c in TYPE_COLOR.items()
    ] + [mpatches.Patch(color=EMPTY_COLOR, label="No session")]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0)
    )

    fig.suptitle("Workout Calendar Heatmap", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    # 7. Save or show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved, {save_path}")

    return fig






# DISPLAY WEEKLY VOLUME
def _compute_set_volume(s: dict) -> float:
    """
        input:  one set dict   {"weight_kg": float|None, "bodyweight": bool, "reps": list[int]}
        output: float          total kg lifted for this set  (weight x total reps)
                                bodyweight sets contribute 0 kg (no bar loaded).
    """
    if s["bodyweight"] or s["weight_kg"] is None:
        return 0.0
    return s["weight_kg"] * sum(s["reps"])


def _workout_volume_kg(workout: dict) -> float:
    """
        input:  one parsed workout dict (as returned by parse_workout_file)
        output: float   total kg lifted across all exercises and sets
    """
    total = 0.0
    for ex in workout["exercises"]:
        for s in ex["sets"]:
            total += _compute_set_volume(s)
    return total


def _iso_week_label(d: date) -> str:
    """
        input:  date object
        output: str   "YYYY-Www"  e.g. "2023-W03"
                    used as the grouping key and X-axis tick label.
    """
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _week_start(iso_label: str) -> date:
    """
        input:  "YYYY-Www" string
        output: date of Monday that starts that ISO week  (for sorting)
    """
    year, week = iso_label.split("-W")
    # ISO week 1 Monday
    jan4 = date(int(year), 1, 4)
    monday = jan4 - timedelta(days=jan4.weekday())
    return monday + timedelta(weeks=int(week) - 1)



# input
#   data      : dict[str, workout_dict]
#               Output of load_workout_directory().
#               Each value must contain:
#                 "date_str"      "YYYY-MM-DD" or None
#                 "session_type"  "A" | "B" | "C" or None
#                 "exercises"     list of exercise dicts with "sets"
#
#   save_path : str | None
#               If given, saves the figure as a PNG at that path.
#               If None, the figure is returned for interactive use.
#
# output
#   matplotlib.figure.Figure
#     Left y-axis  : stacked / grouped bars  total kg lifted per week
#                    colour-coded by session type (A=blue, B=orange, C=green)
#     Right y-axis : line + markers           number of sessions that week
#     x-axis       : ISO week labels ("YYYY-Www"), spaced to avoid clutter

def plot_weekly_volume(data: dict, save_path: str | None = None):
    """
        weekly training volume (kg lifted) with session-count overlay
    """

    TYPE_COLOR = {"A": "#4C9BE8", "B": "#F5A623", "C": "#57C26A"}
    TYPES = ["A", "B", "C"]

    # 1. Aggregate per week × session type
    # week_data[iso_label][session_type] = total kg
    week_data: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    week_counts: dict[str, int] = defaultdict(int)

    for w in data.values():
        if not w["date_str"] or not w["session_type"]:
            continue
        d = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
        label = _iso_week_label(d)
        stype = w["session_type"].upper()
        week_data[label][stype] += _workout_volume_kg(w)
        week_counts[label] += 1

    if not week_data:
        raise ValueError("No sessions with valid dates and volume found.")

    # 2. Sort weeks chronologically
    weeks = sorted(week_data.keys(), key=_week_start)
    x = np.arange(len(weeks))

    # 3. Build per-type volume arrays
    volumes = {t: np.array([week_data[w].get(t, 0.0) for w in weeks]) for t in TYPES}
    session_counts = np.array([week_counts[w] for w in weeks])

    # 4. Plot
    fig, ax1 = plt.subplots(figsize=(max(14, len(weeks) * 0.45), 6), facecolor="white")
    ax1.set_facecolor("white")

    bar_width = 0.7
    bottoms = np.zeros(len(weeks))

    bar_containers = []
    for stype in TYPES:
        bars = ax1.bar(
            x, volumes[stype],
            width=bar_width,
            bottom=bottoms,
            color=TYPE_COLOR[stype],
            label=f"Session {stype}",
            zorder=2,
            linewidth=0,
        )
        bar_containers.append(bars)
        bottoms += volumes[stype]

    # 5. Session count line on right axis
    ax2 = ax1.twinx()
    ax2.plot(
        x, session_counts,
        color="#E85D75",
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor="#E85D75",
        label="Sessions",
        zorder=3,
    )
    ax2.set_ylabel("Sessions per week", fontsize=10, color="#E85D75", labelpad=10)
    ax2.tick_params(axis="y", colors="#E85D75", labelsize=9)
    ax2.set_ylim(0, max(session_counts) * 2.2)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # 6. X-axis ticks — show every N weeks to avoid crowding
    n = max(1, len(weeks) // 30)   # target ~30 labels maximum
    tick_positions = x[::n]
    tick_labels    = [weeks[i] for i in range(0, len(weeks), n)]

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_xlim(-0.8, len(weeks) - 0.2)

    # 7. Left axis labels
    ax1.set_ylabel("Volume (kg)", fontsize=10, labelpad=10)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.tick_params(axis="y", labelsize=9)
    ax1.tick_params(axis="x", labelsize=8)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#cccccc")
    ax1.spines["bottom"].set_color("#cccccc")
    ax1.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    # 8. Legend (combined from both axes)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1  + labels2,
        loc="upper left",
        fontsize=9,
        frameon=False,
    )

    ax1.set_title(
        "Weekly Training Volume & Frequency",
        fontsize=13,
        fontweight="bold",
        pad=14,
        loc="left",
    )

    fig.tight_layout()

    # 9. Save or show
    if save_path:
        fig.savefig(save_path, dpi= 150, bbox_inches= "tight")
        print(f"Weekly volume chart saved, {save_path}")

    return fig





# STRENGTH PROGRESSION GRAPH

TRACKED_EXERCISES = [
    "bench_press",
    "db_shoulder_press",
    "leg_extension",
    "lat_pulldown",
]

EXERCISE_COLOR = {
    "bench_press":     "#4C9BE8",
    "db_shoulder_press":  "#F5A623",
    "leg_extension":   "#57C26A",
    "lat_pulldown":    "#E85D75",
}


def _max_weight_for_exercise(workout: dict, exercise_name: str):
    """
        Find the heaviest non-bodyweight weight used for a given exercise in one workout.
        Returns float or None if the exercise was not done that day.
    """
    for ex in workout["exercises"]:
        if ex["name"] != exercise_name:
            continue
        weights = [
            s["weight_kg"]
            for s in ex["sets"]
            if not s["bodyweight"] and s["weight_kg"] is not None and s["weight_kg"] > 0
        ]
        if weights:
            return max(weights)
    return None


def plot_strength_progression(data: dict, save_path: str | None = None):
    """
        Plot strength progression over time for four exercises:
        bench_press, shoulder_press, leg_extension, lat_pulldown.

        Each exercise gets its own subplot. The x-axis is the workout date
        and the y-axis is the heaviest weight used in that session (kg).
        Bodyweight sets are excluded.

        Parameters
        ----------
        data : dict
            Output of load_workout_directory().
        save_path : str or None
            File path to save the figure as a PNG.
            Pass None to skip saving.

        Returns
        -------
        matplotlib.figure.Figure
    """

    # Build a timeline per exercise: list of (date, max_weight_kg)
    timelines: dict[str, list[tuple[date, float]]] = defaultdict(list)

    for w in data.values():
        if not w["date_str"]:
            continue
        d = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
        for exercise_name in TRACKED_EXERCISES:
            max_w = _max_weight_for_exercise(w, exercise_name)
            if max_w is not None:
                timelines[exercise_name].append((d, max_w))

    # Sort each timeline by date
    for exercise_name in TRACKED_EXERCISES:
        timelines[exercise_name].sort(key=lambda x: x[0])

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 10),
        facecolor="white",
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    axes_flat = axes.flatten()

    for ax, exercise_name in zip(axes_flat, TRACKED_EXERCISES):
        ax.set_facecolor("white")
        color = EXERCISE_COLOR[exercise_name]
        points = timelines[exercise_name]

        if not points:
            ax.text(
                0.5, 0.5,
                "No data found",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#aaaaaa",
            )
            ax.set_title(exercise_name.replace("_", " ").title(), fontsize=12, fontweight="bold", loc="left", pad=8)
            continue

        dates = [p[0] for p in points]
        weights = [p[1] for p in points]

        # Main line
        ax.plot(
            dates, weights,
            color=color,
            linewidth=1.8,
            zorder=2,
        )

        # Scatter dots on top
        ax.scatter(
            dates, weights,
            color=color,
            s=22,
            zorder=3,
            linewidths=0,
        )

        # Light fill below the curve
        ax.fill_between(
            dates, weights,
            alpha=0.12,
            color=color,
            zorder=1,
        )

        # Rolling 5-session average as a guide line (only if enough data)
        if len(weights) >= 5:
            window = 5
            smoothed = np.convolve(weights, np.ones(window) / window, mode="valid")
            smoothed_dates = dates[window - 1:]
            ax.plot(
                smoothed_dates, smoothed,
                color=color,
                linewidth=2.5,
                alpha=0.45,
                linestyle="--",
                zorder=2,
                label="5-session avg",
            )
            ax.legend(fontsize=8, frameon=False, loc="upper left")

        # Axis styling
        ax.set_title(
            exercise_name.replace("_", " ").title(),
            fontsize=12, fontweight="bold", loc="left", pad=8,
        )
        ax.set_ylabel("Weight (kg)", fontsize=9, labelpad=8)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.0f} kg")
        )

        # Annotate the personal best
        peak_w = max(weights)
        peak_d = dates[weights.index(peak_w)]
        ax.annotate(
            f"PB {peak_w:.1f} kg",
            xy=(peak_d, peak_w),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    fig.suptitle(
        "Strength Progression Over Time",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Strength progression chart saved, {save_path}")

    return fig




# input:
#   data      : dict returned by load_workout_directory()
#   save_path : str | None
#
# output:
#   matplotlib.figure.Figure  (two-panel)
#     Top panel  : histogram of rest-gap lengths (days between consecutive
#                  sessions), coloured by whether the gap is 1 / 2 / 3+ days.
#                  A vertical dashed line marks the median gap.
#     Bottom panel: timeline of every gap, plotted as a stem chart so you
#                  can spot periods of high and low training density at a glance.
 
def plot_rest_gap_distribution(data: dict, save_path: str | None = None):
    """
        Analyse and visualise the number of rest days between consecutive workout sessions
    """
    # 1. collect sorted unique training dates
    training_dates: list[date] = []
    for w in data.values():
        if not w["date_str"]:
            continue
        training_dates.append(datetime.strptime(w["date_str"], "%Y-%m-%d").date())
 
    # Deduplicate (multiple sessions same day → one date) then sort
    training_dates = sorted(set(training_dates))
 
    if len(training_dates) < 2:
        raise ValueError("Need at least 2 sessions to compute rest gaps.")
 
    # 2. compute gaps between consecutive dates
    gaps: list[int] = []
    gap_dates: list[date] = []   # date of the session *after* the gap
 
    for i in range(1, len(training_dates)):
        g = (training_dates[i] - training_dates[i - 1]).days
        gaps.append(g)
        gap_dates.append(training_dates[i])
 
    gaps_arr = np.array(gaps)
    median_gap = float(np.median(gaps_arr))
    mean_gap   = float(np.mean(gaps_arr))
 
    # 3. colour buckets
    def gap_color(g: int) -> str:
        if g == 1:  return "#57C26A"   # green  — back-to-back
        if g == 2:  return "#4C9BE8"   # blue   — one rest day
        if g == 3:  return "#F5A623"   # orange — two rest days
        return "#E85D75"               # red    — long break (3+ rest days)
 
    colors = [gap_color(g) for g in gaps]
 
    # 4. figure: two rows
    fig, (ax_hist, ax_stem) = plt.subplots(
        2, 1,
        figsize=(16, 8),
        facecolor="white",
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    fig.subplots_adjust(hspace=0.45)
 
    # Top: histogram
    ax_hist.set_facecolor("white")
 
    max_gap = int(gaps_arr.max())
    bin_edges = np.arange(0.5, max_gap + 1.5, 1)
 
    # draw one bar per unique gap value, coloured by bucket
    for gap_val in range(1, max_gap + 1):
        count = int((gaps_arr == gap_val).sum())
        if count == 0:
            continue
        ax_hist.bar(
            gap_val, count,
            width=0.7,
            color=gap_color(gap_val),
            linewidth=0,
            zorder=2,
        )
        ax_hist.text(gap_val, count + 0.15, str(count),
                     ha="center", va="bottom", fontsize=8, color="#555555")
 
    # median line
    ax_hist.axvline(median_gap, color="#333333", linewidth=1.5,
                    linestyle="--", zorder=3, label=f"Median {median_gap:.0f} d")
    ax_hist.axvline(mean_gap, color="#999999", linewidth=1.2,
                    linestyle=":", zorder=3, label=f"Mean {mean_gap:.1f} d")
 
    ax_hist.set_xlabel("Days between sessions", fontsize=10, labelpad=6)
    ax_hist.set_ylabel("Number of gaps", fontsize=10, labelpad=8)
    ax_hist.set_title("Rest Gap Distribution", fontsize=13,
                      fontweight="bold", loc="left", pad=10)
    ax_hist.set_xticks(range(1, max_gap + 1))
    ax_hist.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax_hist.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_hist.spines[spine].set_visible(False)
    ax_hist.spines["left"].set_color("#cccccc")
    ax_hist.spines["bottom"].set_color("#cccccc")
 
    # legend patches
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="#57C26A", label="1 day (back-to-back)"),
        mpatches.Patch(color="#4C9BE8", label="2 days (1 rest)"),
        mpatches.Patch(color="#F5A623", label="3 days (2 rest)"),
        mpatches.Patch(color="#E85D75", label="4+ days (long break)"),
    ]
    ax_hist.legend(
        handles=legend_patches + [
            plt.Line2D([0], [0], color="#333333", linewidth=1.5, linestyle="--",
                       label=f"Median {median_gap:.0f} d"),
            plt.Line2D([0], [0], color="#999999", linewidth=1.2, linestyle=":",
                       label=f"Mean {mean_gap:.1f} d"),
        ],
        fontsize=8.5, frameon=False, loc="upper right", ncol=2,
    )
 
    # bsottom: stem timeline
    ax_stem.set_facecolor("white")
 
    x_idx = np.arange(len(gaps))
    for xi, (g, c) in enumerate(zip(gaps, colors)):
        ax_stem.vlines(xi, 0, g, color=c, linewidth=1.4, zorder=2)
        ax_stem.scatter([xi], [g], color=c, s=20, zorder=3, linewidths=0)
 
    ax_stem.axhline(median_gap, color="#333333", linewidth=1.2,
                    linestyle="--", zorder=1, alpha=0.6)
 
    # X-axis: label every ~20th gap with its date
    label_step = max(1, len(gaps) // 20)
    tick_pos    = x_idx[::label_step]
    tick_labels = [str(gap_dates[i]) for i in range(0, len(gaps), label_step)]
    ax_stem.set_xticks(tick_pos)
    ax_stem.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=7.5)
 
    ax_stem.set_ylabel("Gap (days)", fontsize=9, labelpad=8)
    ax_stem.set_title("Gap Length Over Time", fontsize=11,
                      fontweight="bold", loc="left", pad=8)
    ax_stem.set_xlim(-1, len(gaps))
    ax_stem.set_ylim(0, max(gaps) + 1)
    ax_stem.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax_stem.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_stem.spines[spine].set_visible(False)
    ax_stem.spines["left"].set_color("#cccccc")
    ax_stem.spines["bottom"].set_color("#cccccc")
 
    # summary stats box
    pct_1 = 100 * (gaps_arr == 1).sum() / len(gaps_arr)
    pct_2 = 100 * (gaps_arr == 2).sum() / len(gaps_arr)
    pct_long = 100 * (gaps_arr >= 4).sum() / len(gaps_arr)
    stats_text = (
        f"Total gaps: {len(gaps)}\n"
        f"Back-to-back (1 d): {pct_1:.0f}%\n"
        f"1 rest day (2 d): {pct_2:.0f}%\n"
        f"Long breaks (≥4 d): {pct_long:.0f}%"
    )
    ax_stem.text(
        0.01, 0.97, stats_text,
        transform=ax_stem.transAxes,
        fontsize=8, va="top", color="#555555",
        bbox=dict(facecolor="white", edgecolor="#dddddd", boxstyle="round,pad=0.4"),
    )
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Rest gap distribution chart saved, {save_path}")
 
    return fig







# REP QUALITY INDEX (RQI) OVER TIME
#
# WHAT IS RQI?
#   For every set that has more than one rep count in its list, RQI is:
#       RQI = last_rep_count / first_rep_count   (clamped to [0, 1])
#   A value of 1.0 means you matched your opening rep count on the final set —
#   no fatigue-driven drop.  0.8 means your last set got 20% fewer reps than
#   your first.  Single-element rep lists (one uniform number) are treated as
#   1.0 because there is nothing to compare.
#
# WHY IS THIS MEANINGFUL?
#   Strength progression charts show weight going up, but RQI captures HOW
#   CLEANLY you lifted it.  A load increase paired with a stable RQI is a true
#   quality gain.  One where RQI tanks means you are grinding out ugly sets.
#   Comparing RQI across exercises reveals which movements break you down first.
#
# INPUT:
#   data      : dict returned by load_workout_directory()
#   exercises : list[str] | None  — defaults to TRACKED_EXERCISES
#   window    : int  — rolling-average window in sessions (default 6)
#   save_path : str | None
#
# OUTPUT:
#   matplotlib.figure.Figure  — one subplot per exercise
#     • Grey scatter : raw per-session mean RQI
#     • Coloured line : rolling average RQI
#     • Shaded band   : ±1 SD of the rolling window
#     • Dashed line   : overall mean RQI for that exercise
#     • Stats box     : n sessions, mean, std, % perfect (RQI == 1.0)
 
def _set_rqi(reps: list) -> float | None:
    """
        compute Rep Quality Index for one set's rep list.
        returns None when the list is empty (cannot compute)
        single-element lists return 1.0 (nothing to compare)
    """
    if not reps:
        return None
    if len(reps) == 1:
        return 1.0
    if reps[0] == 0:
        return None
    return min(1.0, reps[-1] / reps[0])
 
 
def plot_rep_quality_index(
    data: dict,
    exercises: list | None = None,
    window: int = 6,
    save_path: str | None = None,
):
    """
        plot the Rep Quality Index (RQI) over time for each tracked exercise.
    
        RQI = last_set_reps / first_set_reps, averaged across all sets of that
        exercise in a session.  1.0 = no rep drop-off; lower = fatigue within
        the session.
    """
    if exercises is None:
        exercises = TRACKED_EXERCISES
 
    # 1. Build per-exercise timeline: list of (date, mean_rqi_for_session)
    #    Only sessions where the exercise appears AND at least one set has
    #    a computable RQI are included.
    timelines: dict[str, list[tuple[date, float]]] = defaultdict(list)
 
    for w in data.values():
        if not w["date_str"]:
            continue
        d = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
 
        for ex in w["exercises"]:
            if ex["name"] not in exercises:
                continue
            rqis = [_set_rqi(s["reps"]) for s in ex["sets"]]
            rqis = [r for r in rqis if r is not None]
            if not rqis:
                continue
            timelines[ex["name"]].append((d, float(np.mean(rqis))))
 
    # Sort each timeline chronologically
    for name in exercises:
        timelines[name].sort(key=lambda p: p[0])
 
    # 2. Figure layout: 2-column grid, same as strength_progression
    n_ex   = len(exercises)
    n_cols = 2
    n_rows = (n_ex + 1) // n_cols
 
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 5 * n_rows),
        facecolor="white",
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
 
    axes_flat = np.array(axes).flatten() if n_ex > 1 else [axes]
 
    for ax, ex_name in zip(axes_flat, exercises):
        ax.set_facecolor("white")
        color  = EXERCISE_COLOR.get(ex_name, "#888888")
        points = timelines[ex_name]
 
        if not points:
            ax.text(0.5, 0.5, "No data found",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#aaaaaa")
            ax.set_title(ex_name.replace("_", " ").title(),
                         fontsize=12, fontweight="bold", loc="left", pad=8)
            continue
 
        dates  = [p[0] for p in points]
        rqis   = [p[1] for p in points]
        n_pts  = len(points)
 
        # Raw scatter
        ax.scatter(dates, rqis, color="#CCCCCC", s=22, zorder=2,
                   linewidths=0, label="Session RQI")
 
        # Rolling average + band
        win = min(window, n_pts)
        if win >= 2:
            roll_avg, roll_std, roll_dates = [], [], []
            for i in range(win - 1, n_pts):
                chunk = rqis[i - win + 1 : i + 1]
                roll_avg.append(float(np.mean(chunk)))
                roll_std.append(float(np.std(chunk)))
                roll_dates.append(dates[i])
 
            avg_arr = np.array(roll_avg)
            std_arr = np.array(roll_std)
 
            ax.fill_between(roll_dates,
                            np.clip(avg_arr - std_arr, 0, 1),
                            np.clip(avg_arr + std_arr, 0, 1),
                            color=color, alpha=0.15, zorder=1)
            ax.plot(roll_dates, avg_arr,
                    color=color, linewidth=2.2, zorder=4,
                    label=f"{win}-session avg")
 
        # Overall mean reference line
        overall_mean = float(np.mean(rqis))
        ax.axhline(overall_mean, color=color, linewidth=1.0,
                   linestyle="--", alpha=0.55, zorder=1,
                   label=f"Mean {overall_mean:.2f}")
 
        # y-axis ceiling
        ax.set_ylim(max(0.0, min(rqis) - 0.08), 1.05)
        ax.axhline(1.0, color="#dddddd", linewidth=0.8, zorder=0)
 
        # Stats box
        pct_perfect = 100.0 * sum(1 for r in rqis if r == 1.0) / n_pts
        std_all     = float(np.std(rqis))
        stats_text  = (
            f"n = {n_pts} sessions\n"
            f"mean RQI = {overall_mean:.2f}\n"
            f"std  = {std_all:.2f}\n"
            f"perfect sets = {pct_perfect:.0f}%"
        )
        ax.text(
            0.02, 0.05, stats_text,
            transform=ax.transAxes,
            fontsize=7.5, va="bottom", color="#555555",
            bbox=dict(facecolor="white", edgecolor="#dddddd",
                      boxstyle="round,pad=0.4"),
        )
 
        # Axis styling
        ax.set_title(ex_name.replace("_", " ").title(),
                     fontsize=12, fontweight="bold", loc="left", pad=8)
        ax.set_ylabel("Rep Quality Index", fontsize=9, labelpad=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.2f}")
        )
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
 
    for ax in axes_flat[n_ex:]:
        ax.set_visible(False)
 
    fig.suptitle(
        "Rep Quality Index Over Time\n"
        "( last set reps ÷ first set reps — 1.0 = no fatigue drop )",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Rep Quality Index chart saved, {save_path}")
 
    return fig


