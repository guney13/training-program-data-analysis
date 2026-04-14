
"""
    hypothesis_test.py
    ------------------
    Tests whether higher training frequency correlates with faster strength gains.

    H0: no significant positive linear relationship between frequency and strength gain rate
    H1: higher frequency is positively associated with faster strength gain rate

    One-tailed Pearson correlation test, alpha = 0.05.
"""

import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from scipy import stats


TRACKED_EXERCISES = [
    "bench_press",
    "shoulder_press",
    "leg_extension",
    "lat_pulldown",
]

# Window size in days. 4 weeks gives enough sessions per bucket
# without being so wide that changes within it get lost
WINDOW_DAYS = 28

ALPHA = 0.05


def _max_weight_for_exercise(workout: dict, exercise_name: str):
    """
        Return the heaviest non-bodyweight load used for an exercise in one workout.
        Returns None if the exercise was not performed.
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


def _build_dated_records(data: dict):
    """
        Convert the raw data dict into a list of records, each being:
            { "date": date, "exercise_name": str, "max_weight_kg": float }

        Only includes workouts with a valid date and exercises from TRACKED_EXERCISES.
    """
    records = []
    for w in data.values():
        if not w["date_str"]:
            continue
        d = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
        for ex_name in TRACKED_EXERCISES:
            mw = _max_weight_for_exercise(w, ex_name)
            if mw is not None:
                records.append({"date": d, "exercise_name": ex_name, "max_weight_kg": mw})
    return sorted(records, key=lambda r: r["date"])


def _make_windows(records: list, window_days: int):
    """
        Slice the full date range into non-overlapping windows of width window_days.

        For each window, compute:
            - session_count    : number of workouts (distinct dates) that fall in it
            - avg_gain_kg_week : average weekly increase in max weight across
                                all tracked exercises that have data in both
                                this window and the previous one

        Returns a list of dicts, one per window that has enough data.
        Each window needs at least one exercise with data in the current
        AND previous window to produce a gain rate estimate.
    """
    if not records:
        return []

    min_date = records[0]["date"]
    max_date = records[-1]["date"]

    # Build windows as (window_start, window_end) pairs
    windows = []
    cursor = min_date
    while cursor <= max_date:
        windows.append((cursor, cursor + timedelta(days=window_days - 1)))
        cursor += timedelta(days=window_days)

    # For each window, group records by exercise
    # window_exercise_max[window_idx][exercise] = max weight seen in that window
    window_exercise_max = defaultdict(lambda: defaultdict(list))
    window_dates = defaultdict(set)

    for r in records:
        for i, (wstart, wend) in enumerate(windows):
            if wstart <= r["date"] <= wend:
                window_exercise_max[i][r["exercise_name"]].append(r["max_weight_kg"])
                window_dates[i].add(r["date"])
                break

    # Collapse lists to max weight per window per exercise
    window_peaks = {}
    for i in range(len(windows)):
        window_peaks[i] = {
            ex: max(weights)
            for ex, weights in window_exercise_max[i].items()
        }

    # Compute gain rate per window (needs comparison with previous window)
    result_windows = []

    for i in range(1, len(windows)):
        prev_peaks = window_peaks.get(i - 1, {})
        curr_peaks = window_peaks.get(i, {})

        # Exercises present in both windows
        common = set(prev_peaks.keys()) & set(curr_peaks.keys())
        if not common:
            continue

        # Gain in kg over one window, converted to kg/week
        gain_rates = []
        for ex in common:
            gain_kg = curr_peaks[ex] - prev_peaks[ex]
            gain_kg_per_week = gain_kg / (window_days / 7)
            gain_rates.append(gain_kg_per_week)

        avg_gain_rate = float(np.mean(gain_rates))

        session_count = len(window_dates.get(i, set()))

        wstart, wend = windows[i]
        result_windows.append({
            "window_label":         f"{wstart} to {wend}",
            "session_count":        session_count,
            "avg_gain_kg_per_week": avg_gain_rate,
            "exercises_tracked":    sorted(common),
        })

    return result_windows


def test_frequency_vs_strength(data: dict, window_days: int = WINDOW_DAYS, alpha: float = ALPHA):
    """
        Test whether training frequency correlates with strength gain rate.

        Parameters
        ----------
        data : dict
            Output of load_workout_directory().
        window_days : int
            Width of each time bucket in days (default 28).
        alpha : float
            Significance level for the one-tailed test (default 0.05).

        Returns
        -------
        dict with keys:
            windows              list of per-window dicts (frequency + gain rate)
            n                    number of windows used
            pearson_r            Pearson correlation coefficient
            t_statistic          computed t value
            t_critical           one-tailed critical value at alpha
            p_value              one-tailed p-value
            degrees_of_freedom   n - 2
            alpha                significance level used
            reject_null          bool, True if we reject H0
            conclusion           plain-English summary string
    """
    records = _build_dated_records(data)
    windows = _make_windows(records, window_days)

    if len(windows) < 4:
        raise ValueError(
            f"Only {len(windows)} usable windows found. "
            "Need at least 4. Try reducing window_days or check that your data "
            "has enough sessions across time."
        )

    freq   = np.array([w["session_count"]        for w in windows], dtype=float)
    gain   = np.array([w["avg_gain_kg_per_week"]  for w in windows], dtype=float)

    n  = len(windows)
    df = n - 2

    # Pearson r
    r, _ = stats.pearsonr(freq, gain)

    # t statistic
    # guard against perfect correlation blowing up the formula
    if abs(r) == 1.0:
        t_stat = np.inf * np.sign(r)
    else:
        t_stat = r * np.sqrt(df / (1 - r ** 2))

    # one-tailed critical value and p-value (right tail)
    t_crit  = stats.t.ppf(1 - alpha, df=df)
    p_value = stats.t.sf(t_stat, df=df)   # sf = 1 - cdf, i.e. right-tail area

    reject = bool(t_stat > t_crit)

    if reject:
        conclusion = (
            f"Reject H0 (p = {p_value:.4f} < alpha = {alpha}). "
            f"There is statistically significant evidence that higher training "
            f"frequency is associated with faster strength gains "
            f"(r = {r:.3f}, t({df}) = {t_stat:.3f})."
        )
    else:
        conclusion = (
            f"Fail to reject H0 (p = {p_value:.4f} >= alpha = {alpha}). "
            f"The data does not provide sufficient evidence that training "
            f"frequency drives faster strength gains "
            f"(r = {r:.3f}, t({df}) = {t_stat:.3f})."
        )

    return {
        "windows":            windows,
        "n":                  n,
        "pearson_r":          float(r),
        "t_statistic":        float(t_stat),
        "t_critical":         float(t_crit),
        "p_value":            float(p_value),
        "degrees_of_freedom": df,
        "alpha":              alpha,
        "reject_null":        reject,
        "conclusion":         conclusion,
    }


def print_hypothesis_results(results: dict):
    """
        Pretty-print the output of test_frequency_vs_strength()
    """
    sep = "-" * 52

    print(sep)
    print("  Frequency vs Strength Gain Rate — Hypothesis Test")
    print(sep)
    print(f"  H0 : no positive correlation between frequency and gain rate")
    print(f"  H1 : higher frequency -> faster strength gains  (one-tailed)")
    print(sep)
    print(f"  Windows analysed     : {results['n']}  (each ~{WINDOW_DAYS} days)")
    print(f"  Degrees of freedom   : {results['degrees_of_freedom']}")
    print(f"  Significance level   : alpha = {results['alpha']}")
    print()
    print(f"  Pearson r            : {results['pearson_r']:+.4f}")
    print(f"  t statistic          : {results['t_statistic']:+.4f}")
    print(f"  t critical (right)   : {results['t_critical']:+.4f}")
    print(f"  p-value (one-tailed) : {results['p_value']:.4f}")
    print()
    verdict = "REJECT H0" if results["reject_null"] else "FAIL TO REJECT H0"
    print(f"  Verdict  >>  {verdict}")
    print()
    print(f"  {results['conclusion']}")
    print(sep)

    print()
    print("  Per-window breakdown:")
    print(f"  {'Window':<26}  {'Sessions':>8}  {'Avg gain (kg/wk)':>16}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*16}")
    for w in results["windows"]:
        print(
            f"  {w['window_label']:<26}  "
            f"{w['session_count']:>8}  "
            f"{w['avg_gain_kg_per_week']:>+16.3f}"
        )
    print(sep)
