

import calendar
from datetime import datetime, date
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
            ax.text(wk, -0.6, MONTHS[month - 1],
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