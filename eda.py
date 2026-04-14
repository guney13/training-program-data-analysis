

import calendar
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# INPUT:  data dict returned by load_workout_directory()
#   keys: "W-NNN-X", values: parsed workout dicts
#   save_path  str | None — if given, saves PNG to that path
#
# OUTPUT: matplotlib.figure.Figure
#   A calendar heatmap where each day cell is coloured by
#   session type(s) logged that day (A= blue, B= orange, C= green).
#   Cells with no session are empty (light grey).



def plot_workout_heatmap(data: dict, save_path: str | None = None):
    """
        Render a GitHub-style calendar heatmap of all workout sessions.

        Parameters
        ----------
        data : dict
            Output of load_workout_directory(). Each value must contain
            'date_str' (ISO "YYYY-MM-DD") and 'session_type' ("A", "B", or "C").
        save_path : str or None
            File path to save the figure (e.g. "heatmap.png").
            If None the figure is shown interactively.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered heatmap figure.
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
        print(f"Heatmap saved → {save_path}")

    return fig





import matplotlib.ticker as ticker
from collections import defaultdict


# HELPERS

def _compute_set_volume(s: dict) -> float:
    """
    INPUT:  one set dict   {"weight_kg": float|None, "bodyweight": bool, "reps": list[int]}
    OUTPUT: float          total kg lifted for this set  (weight x total reps)
                            bodyweight sets contribute 0 kg (no bar loaded).
    """
    if s["bodyweight"] or s["weight_kg"] is None:
        return 0.0
    return s["weight_kg"] * sum(s["reps"])


def _workout_volume_kg(workout: dict) -> float:
    """
    INPUT:  one parsed workout dict (as returned by parse_workout_file)
    OUTPUT: float   total kg lifted across all exercises and sets
    """
    total = 0.0
    for ex in workout["exercises"]:
        for s in ex["sets"]:
            total += _compute_set_volume(s)
    return total


def _iso_week_label(d: date) -> str:
    """
    INPUT:  date object
    OUTPUT: str   "YYYY-Www"  e.g. "2023-W03"
                   used as the grouping key and X-axis tick label.
    """
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _week_start(iso_label: str) -> date:
    """
    INPUT:  "YYYY-Www" string
    OUTPUT: date of Monday that starts that ISO week  (for sorting)
    """
    year, week = iso_label.split("-W")
    # ISO week 1 Monday
    jan4 = date(int(year), 1, 4)
    monday = jan4 - timedelta(days=jan4.weekday())
    return monday + timedelta(weeks=int(week) - 1)





# ── INPUT ─────────────────────────────────────────────────────────────
#   data       : dict[str, workout_dict]
#                Output of load_workout_directory().
#                Each value must contain:
#                  "date_str"      "YYYY-MM-DD" or None
#                  "session_type"  "A" | "B" | "C" or None
#                  "exercises"     list of exercise dicts with "sets"
#
#   save_path  : str | None
#                If given, saves the figure as a PNG at that path.
#                If None, the figure is returned for interactive use.
#
# ── OUTPUT ────────────────────────────────────────────────────────────
#   matplotlib.figure.Figure
#     Left  Y-axis : stacked / grouped bars  total kg lifted per week
#                    colour-coded by session type (A=blue, B=orange, C=green)
#     Right Y-axis : line + markers           number of sessions that week
#     X-axis       : ISO week labels ("YYYY-Www"), spaced to avoid clutter
# ─────────────────────────────────────────────────────────────────────

def plot_weekly_volume(data: dict, save_path: str | None = None):
    """
    Weekly training volume (kg lifted) with session-count overlay.

    Parameters
    ----------
    data : dict
        Output of load_workout_directory().
    save_path : str or None
        File path to save PNG (e.g. "weekly_volume.png").
        Pass None to skip saving.

    Returns
    -------
    matplotlib.figure.Figure
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
        print(f"Weekly volume chart saved → {save_path}")

    return fig