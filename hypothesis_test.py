"""
    Two Pearson-correlation hypothesis tests on workout log data

    Test 1 — Frequency vs Strength Gain Rate
        H0: no significant positive linear relationship between frequency and strength gain rate
        H1: higher frequency is positively associated with faster strength gain rate
        One-tailed Pearson correlation test, alpha = 0.05.

    Test 2 — Rest Gap vs Session Volume
        H0: the number of rest days before a session has no linear relationship with
            the total volume lifted in that session
        H1: rest gap and session volume are linearly associated (either direction)
        Two-tailed Pearson correlation test, alpha = 0.05.

        Rationale: Both under-recovery (gap too short → fatigue carries over) and
        detraining (gap too long → strength/readiness drops) are plausible, so we
        make no directional assumption and use a two-tailed test.
        Unit of analysis is the individual session, not a time window.
        Volume = Summation (weight_kg x reps) across all weighted sets in the session.
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





###
# Test 2
###

def _session_volume(workout: dict) -> float:
    """
        Total volume for one session: Summation (weight_kg x reps) across every
        weighted set in every exercise.  Bodyweight sets are excluded because
        the load is unknown and inconsistent across body-weight changes
        Returns 0.0 if no weighted sets exist
    """
    total = 0.0
    for ex in workout["exercises"]:
        for s in ex["sets"]:
            if s["bodyweight"] or s["weight_kg"] is None or s["weight_kg"] <= 0:
                continue
            total += s["weight_kg"] * sum(s["reps"])
    return total


def _build_gap_volume_pairs(data: dict) -> list[dict]:
    """
        For each session that has a valid date and non-zero volume, record:
            gap_days  - calendar days since the previous session (skipped for first)
            volume_kg - total weighted volume (kg x reps) of this session
            date_str  - for display
            key       - workout key e.g. "W-170-A"

        Sessions are sorted chronologically. The very first session has no predecessor so it is excluded from the output.
    """
    sessions = []
    for key, w in data.items():
        if not w["date_str"]:
            continue
        vol = _session_volume(w)
        if vol <= 0:
            continue
        d = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
        sessions.append({"key": key, "date": d, "date_str": w["date_str"], "volume_kg": vol})

    sessions.sort(key=lambda s: s["date"])

    pairs = []
    for i in range(1, len(sessions)):
        gap = (sessions[i]["date"] - sessions[i - 1]["date"]).days
        pairs.append({
            "key":        sessions[i]["key"],
            "date_str":   sessions[i]["date_str"],
            "gap_days":   gap,
            "volume_kg":  sessions[i]["volume_kg"],
        })

    return pairs


def test_rest_gap_vs_volume(data: dict, alpha: float = ALPHA) -> dict:
    """
    Test whether the rest gap before a session correlates with session volume.

    Parameters
        data : dict
            Output of load_workout_directory().
        alpha : float
            Significance level for the two-tailed test (default 0.05).

    Returns
        dict with keys:
            pairs                list of per-session dicts (gap_days + volume_kg)
            n                    number of sessions used
            pearson_r            Pearson correlation coefficient
            t_statistic          computed t value
            t_critical           two-tailed critical value at alpha
            p_value              two-tailed p-value
            degrees_of_freedom   n - 2
            alpha                significance level used
            reject_null          bool, True if we reject H0
            conclusion           plain-English summary string
    """
    pairs = _build_gap_volume_pairs(data)

    if len(pairs) < 4:
        raise ValueError(
            f"Only {len(pairs)} usable session pairs found. "
            "Need at least 4. Check that your data has valid dates and weighted sets."
        )

    gaps    = np.array([p["gap_days"]  for p in pairs], dtype=float)
    volumes = np.array([p["volume_kg"] for p in pairs], dtype=float)

    n  = len(pairs)
    df = n - 2

    r, _ = stats.pearsonr(gaps, volumes)

    if abs(r) == 1.0:
        t_stat = np.inf * np.sign(r)
    else:
        t_stat = r * np.sqrt(df / (1 - r ** 2))

    # Two-tailed: critical value splits alpha across both tails
    t_crit  = stats.t.ppf(1 - alpha / 2, df=df)
    p_value = 2 * stats.t.sf(abs(t_stat), df=df) # two-tailed p

    reject = bool(abs(t_stat) > t_crit)

    direction = "positively" if r > 0 else "negatively"
    if reject:
        conclusion = (
            f"Reject H0 (p = {p_value:.4f} < alpha = {alpha}). "
            f"Rest gap is {direction} associated with session volume "
            f"(r = {r:.3f}, t({df}) = {t_stat:.3f}). "
            + ("Longer rest → more volume." if r > 0
               else "Longer rest → less volume.")
        )
    else:
        conclusion = (
            f"Fail to reject H0 (p = {p_value:.4f} >= alpha = {alpha}). "
            f"No significant linear relationship between rest gap and session volume "
            f"(r = {r:.3f}, t({df}) = {t_stat:.3f})."
        )

    return {
        "pairs":              pairs,
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


###
# Test 3
###

def _build_session_type_volume_table(data: dict) -> list[dict]:
    """
        For every session with a valid date, known session type (A/B/C), and non-zero weighted volume, return:
            { "key", "date_str", "session_type", "volume_kg" }
    """
    rows = []
    for key, w in data.items():
        if not w["date_str"] or not w["session_type"]:
            continue
        vol = _session_volume(w)
        if vol <= 0:
            continue
        rows.append({
            "key":          key,
            "date_str":     w["date_str"],
            "session_type": w["session_type"].upper(),
            "volume_kg":    vol,
        })
    return rows


def test_session_type_vs_volume_tier(data: dict, alpha: float = ALPHA) -> dict:
    """
    Chi-squared test of independence: does session type (A / B / C) affect
    whether a session is "high" or "low" volume?

    Design
        For each session type separately, compute the median volume.
        Label every session "high" if its volume >= its type's median, else "low".
        Build a 3 x 2 contingency table (rows = A/B/C, cols = high/low) and apply
        scipy.stats.chi2_contingency (with Yates' correction disabled for a 3x2 table).

    Hypotheses
        H0 : session type and volume tier are independent - the proportion of high-volume sessions is the same across A, B, and C days.
        H1 : session type and volume tier are associated - at least one type systematically produces more (or fewer) high-volume sessions.

    Two-tailed by nature (chi-squared is always non-directional).

    Parameters
        data  : dict  — output of load_workout_directory()
        alpha : float — significance level (default 0.05)

    Returns
        dict with keys:
            rows                 raw per-session records used
            contingency_table    dict { session_type: {"high": int, "low": int} }
            session_type_medians dict { session_type: median_volume_kg }
            chi2_statistic       float
            p_value              float
            degrees_of_freedom   int  (= (r-1)*(c-1) = 2 for a 3×2 table)
            alpha                float
            reject_null          bool
            conclusion           str
    """
    rows = _build_session_type_volume_table(data)

    if len(rows) < 10:
        raise ValueError(
            f"Only {len(rows)} usable sessions found. Need at least 10."
        )

    # Compute per-type medians
    from collections import defaultdict
    type_volumes: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        type_volumes[r["session_type"]].append(r["volume_kg"])

    session_types = sorted(type_volumes.keys())
    if len(session_types) < 2:
        raise ValueError(
            "Need at least 2 distinct session types (A/B/C) to run this test."
        )

    medians = {t: float(np.median(vs)) for t, vs in type_volumes.items()}

    # Label each session high/low relative to its type's median
    for r in rows:
        r["tier"] = "high" if r["volume_kg"] >= medians[r["session_type"]] else "low"

    # Build contingency table as a 2-D numpy array: rows=type, cols=high/low
    tiers = ["high", "low"]
    table_dict: dict[str, dict[str, int]] = {
        t: {"high": 0, "low": 0} for t in session_types
    }
    for r in rows:
        table_dict[r["session_type"]][r["tier"]] += 1

    observed = np.array(
        [[table_dict[t][tier] for tier in tiers] for t in session_types],
        dtype=float,
    )

    chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)

    reject = bool(p_value < alpha)

    if reject:
        # Find which type deviates most from independence
        residuals = (observed - expected) / np.sqrt(expected)
        max_idx   = np.unravel_index(np.abs(residuals).argmax(), residuals.shape)
        deviant_type = session_types[max_idx[0]]
        deviant_tier = tiers[max_idx[1]]
        conclusion = (
            f"Reject H0 (p = {p_value:.4f} < alpha = {alpha}). "
            f"Session type and volume tier are not independent "
            f"(χ²({dof}) = {chi2:.3f}). "
            f"The largest deviation from independence is in type-{deviant_type} "
            f"sessions, which have more {deviant_tier}-volume sessions than expected."
        )
    else:
        conclusion = (
            f"Fail to reject H0 (p = {p_value:.4f} >= alpha = {alpha}). "
            f"No significant association between session type and volume tier "
            f"(χ²({dof}) = {chi2:.3f}). "
            f"The high/low volume split is consistent across A, B, and C sessions."
        )

    return {
        "rows":                  rows,
        "contingency_table":     table_dict,
        "session_type_medians":  medians,
        "chi2_statistic":        float(chi2),
        "p_value":               float(p_value),
        "degrees_of_freedom":    dof,
        "alpha":                 alpha,
        "reject_null":           reject,
        "conclusion":            conclusion,
    }


def print_session_type_volume_results(results: dict):
    """
    Pretty-print the output of test_session_type_vs_volume_tier().
    """
    sep = "-" * 58

    print(sep)
    print("  Session Type vs Volume Tier — Chi-Squared Test")
    print(sep)
    print("  H0 : session type and volume tier are independent")
    print("  H1 : session type is associated with volume tier  (non-directional)")
    print(sep)
    print(f"  Sessions analysed    : {len(results['rows'])}")
    print(f"  Degrees of freedom   : {results['degrees_of_freedom']}")
    print(f"  Significance level   : alpha = {results['alpha']}")
    print()

    # Per-type median volumes
    print("  Median volume by session type (threshold for high/low split):")
    for t, med in sorted(results["session_type_medians"].items()):
        print(f"    Type {t} : {med:>10.1f} kg")
    print()

    # Contingency table
    session_types = sorted(results["contingency_table"].keys())
    print(f"  {'Type':<6}  {'High':>8}  {'Low':>8}  {'Total':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for t in session_types:
        hi  = results["contingency_table"][t]["high"]
        lo  = results["contingency_table"][t]["low"]
        print(f"  {t:<6}  {hi:>8}  {lo:>8}  {hi+lo:>8}")
    print()

    print(f"  χ² statistic         : {results['chi2_statistic']:.4f}")
    print(f"  p-value              : {results['p_value']:.4f}")
    print()
    verdict = "REJECT H0" if results["reject_null"] else "FAIL TO REJECT H0"
    print(f"  Verdict  >>  {verdict}")
    print()
    print(f"  {results['conclusion']}")
    print(sep)


def print_rest_gap_results(results: dict):
    """
        Pretty-print the output of test_rest_gap_vs_volume()
    """
    sep = "-" * 56

    print(sep)
    print("  Rest Gap vs Session Volume — Hypothesis Test")
    print(sep)
    print("  H0 : no linear relationship between rest gap and volume")
    print("  H1 : rest gap and session volume are linearly associated  (two-tailed)")
    print(sep)
    print(f"  Sessions analysed    : {results['n']}")
    print(f"  Degrees of freedom   : {results['degrees_of_freedom']}")
    print(f"  Significance level   : alpha = {results['alpha']}")
    print()
    print(f"  Pearson r            : {results['pearson_r']:+.4f}")
    print(f"  t statistic          : {results['t_statistic']:+.4f}")
    print(f"  t critical (±)       : ±{results['t_critical']:.4f}")
    print(f"  p-value (two-tailed) : {results['p_value']:.4f}")
    print()
    verdict = "REJECT H0" if results["reject_null"] else "FAIL TO REJECT H0"
    print(f"  Verdict  >>  {verdict}")
    print()
    print(f"  {results['conclusion']}")
    print(sep)

    print()
    print("  Per-session breakdown (first 20 shown):")
    print(f"  {'Date':<12}  {'Key':<12}  {'Rest days':>9}  {'Volume (kg)':>11}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*11}")
    for p in results["pairs"][:20]:
        print(
            f"  {p['date_str']:<12}  "
            f"{p['key']:<12}  "
            f"{p['gap_days']:>9}  "
            f"{p['volume_kg']:>11.1f}"
        )
    if len(results["pairs"]) > 20:
        print(f"  ... ({len(results['pairs']) - 20} more sessions not shown)")
    print(sep)