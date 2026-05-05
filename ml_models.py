
"""
    predicts the next session's max weight (kg) for each tracked exercise.

    Problem type : Regression
    Target       : next_max_weight_kg  — the heaviest lift in the upcoming session
    Features     : training history features derived from parsed workout logs

    Models compared
        1. Linear Regression  — interpretable baseline
        2. Random Forest      — non-linear, robust to small datasets
        3. XGBoost            — highest capacity, best for tabular data

    Usage (see main.py for full integration):
        from ml_model import build_ml_dataset, train_and_evaluate, print_ml_results, plot_ml_results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datetime import datetime
from collections import defaultdict

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


# exercises to model
TRACKED_EXERCISES = [
    "bench_press",
    "db_shoulder_press",
    "leg_extension",
    "lat_pulldown",
]


# STEP 1 — features
def _compute_session_volume(workout: dict) -> float:
    """Total kg lifted in one session (weight × reps, bodyweight sets = 0)."""
    total = 0.0
    for ex in workout["exercises"]:
        for s in ex["sets"]:
            if not s["bodyweight"] and s["weight_kg"] is not None:
                total += s["weight_kg"] * sum(s["reps"])
    return total


def _max_weight(workout: dict, exercise_name: str):
    """Heaviest non-bodyweight load for one exercise in one workout. None if absent."""
    for ex in workout["exercises"]:
        if ex["name"] != exercise_name:
            continue
        weights = [
            s["weight_kg"]
            for s in ex["sets"]
            if not s["bodyweight"] and s["weight_kg"] is not None and s["weight_kg"] > 0
        ]
        return max(weights) if weights else None
    return None


def build_ml_dataset(data: dict) -> pd.DataFrame:
    """
        Convert the raw workout dict into a flat feature table.

        One row = one (exercise, session) pair where:
        - the exercise was performed that session, AND
        - it was also performed at least once before (so we have lag features)

        Parameters
        ----------
        data : dict
            Output of load_workout_directory().

        Returns
        -------
        pd.DataFrame with columns:
            exercise, date, session_type,
            prev_max_kg, prev2_max_kg, rolling_avg_3_kg,
            days_since_last, sessions_last_7d, sessions_last_28d,
            total_volume_kg, body_weight_kg, weeks_since_start,
            session_type_enc,
            next_max_kg  <- TARGET
    """

    # Sort all workouts chronologically
    workouts = [
        w for w in data.values() if w["date_str"]
    ]
    workouts.sort(key=lambda w: w["date_str"])

    # Precompute: for each date, how many sessions occurred
    date_set = [
        datetime.strptime(w["date_str"], "%Y-%m-%d").date()
        for w in workouts
    ]
    start_date = date_set[0]

    # Per-exercise history: list of (date, max_weight_kg, volume_kg)
    exercise_history: dict[str, list] = defaultdict(list)

    rows = []

    for w in workouts:
        session_date = datetime.strptime(w["date_str"], "%Y-%m-%d").date()
        session_volume = _compute_session_volume(w)
        bw = w["body_weight_kg"]

        # How many sessions in last 7 / 28 days (not counting today)
        sessions_7  = sum(1 for d in date_set if 0 < (session_date - d).days <= 7)
        sessions_28 = sum(1 for d in date_set if 0 < (session_date - d).days <= 28)

        # Encode session type A→0, B→1, C→2
        type_enc = {"A": 0, "B": 1, "C": 2}.get(w["session_type"], -1)

        weeks_since_start = (session_date - start_date).days / 7.0

        for ex_name in TRACKED_EXERCISES:
            max_w = _max_weight(w, ex_name)
            if max_w is None:
                continue  # exercise not done this session

            hist = exercise_history[ex_name]  # previous sessions for this exercise

            if len(hist) == 0:
                # First ever session — no lag features, skip as row but record history
                exercise_history[ex_name].append((session_date, max_w))
                continue

            prev_date, prev_max = hist[-1]
            days_since_last = (session_date - prev_date).days

            prev2_max = hist[-2][1] if len(hist) >= 2 else prev_max

            # Rolling average of last 3 sessions (or fewer)
            last3 = [h[1] for h in hist[-3:]]
            rolling_avg_3 = float(np.mean(last3))

            rows.append({
                "exercise":          ex_name,
                "date":              session_date,
                "session_type":      w["session_type"],
                "session_type_enc":  type_enc,
                "weeks_since_start": weeks_since_start,
                # Lag features
                "prev_max_kg":       prev_max,
                "prev2_max_kg":      prev2_max,
                "rolling_avg_3_kg":  rolling_avg_3,
                "days_since_last":   days_since_last,
                # Frequency features
                "sessions_last_7d":  sessions_7,
                "sessions_last_28d": sessions_28,
                # Load & body
                "total_volume_kg":   session_volume,
                "body_weight_kg":    bw if bw else np.nan,
                # TARGET — current session's max (this becomes "next" for the previous row)
                "next_max_kg":       max_w,
            })

            exercise_history[ex_name].append((session_date, max_w))

    df = pd.DataFrame(rows)

    # Fill missing body weight with forward-filled or median
    df["body_weight_kg"] = df["body_weight_kg"].fillna(df["body_weight_kg"].median())

    return df


# STEP 2 — training and evaluation (Time-Series Cross-Validation)
FEATURE_COLS = [
    "prev_max_kg",
    "prev2_max_kg",
    "rolling_avg_3_kg",
    "days_since_last",
    "sessions_last_7d",
    "sessions_last_28d",
    "total_volume_kg",
    # body_weight_kg excluded: not logged consistently in this dataset (all NaN)
    "session_type_enc",
    "weeks_since_start",
]

TARGET_COL = "next_max_kg"


def _make_models():
    """Return the three models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest":     
            RandomForestRegressor(
                n_estimators= 200,
                max_depth= 6,
                min_samples_leaf= 3,
                random_state= 42,
            ),
        "XGBoost":
            XGBRegressor(
                n_estimators= 200,
                max_depth= 4,
                learning_rate= 0.05,
                subsample= 0.8,
                colsample_bytree= 0.8,
                random_state= 42,
                verbosity= 0,
            ),
    }


def train_and_evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
        Train and evaluate all three models using TimeSeriesSplit cross-validation.

        TimeSeriesSplit is used instead of random k-fold because your data is
        temporal — training on future data to predict the past would be data leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Output of build_ml_dataset().
        n_splits : int
            Number of CV folds (default 5).

        Returns
        -------
        dict with keys:
            results        pd.DataFrame — one row per model with MAE, R², std
            feature_importance  dict[model_name, pd.Series]
            predictions    dict[model_name, list of (actual, predicted)]
            df_used        the filtered DataFrame that was actually used
    """

    # Drop rows with any NaN in features or target
    df_clean = df[FEATURE_COLS + [TARGET_COL, "exercise", "date"]].dropna().copy()
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    if len(df_clean) == 0:
        raise ValueError(
            "ML dataset is empty after cleaning. "
            "Most likely cause: exercise names in TRACKED_EXERCISES do not match "
            "your log files. Check that 'db_shoulder_press', 'bench_press', "
            "'leg_extension', 'lat_pulldown' appear in your .txt files exactly."
        )

    X = df_clean[FEATURE_COLS].values
    y = df_clean[TARGET_COL].values

    # Clamp n_splits so it never exceeds the number of samples
    actual_splits = min(n_splits, len(df_clean) - 1)
    if actual_splits < 2:
        raise ValueError(
            f"Only {len(df_clean)} usable sample(s) found — need at least 3. "
            "Log more sessions or reduce n_splits."
        )
    tscv = TimeSeriesSplit(n_splits=actual_splits)
    models = _make_models()

    summary_rows = []
    feature_importance = {}
    all_predictions = {name: [] for name in models}

    for name, model in models.items():
        fold_maes = []
        fold_r2s  = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_maes.append(mean_absolute_error(y_test, y_pred))
            fold_r2s.append(r2_score(y_test, y_pred))

            # Collect predictions from last fold only (for plotting)
            if test_idx is list(tscv.split(X))[-1][1]:
                all_predictions[name] = list(zip(y_test, y_pred))

        # Re-train on full data for feature importance
        model.fit(X, y)
        y_pred_full = model.predict(X)
        all_predictions[name] = list(zip(y, y_pred_full))

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            fi = pd.Series(np.abs(model.coef_), index=FEATURE_COLS).sort_values(ascending=False)
        else:
            fi = pd.Series(np.zeros(len(FEATURE_COLS)), index=FEATURE_COLS)

        feature_importance[name] = fi

        summary_rows.append({
            "Model":   name,
            "MAE (kg)": round(float(np.mean(fold_maes)), 3),
            "MAE std":  round(float(np.std(fold_maes)),  3),
            "R²":       round(float(np.mean(fold_r2s)),  4),
            "R² std":   round(float(np.std(fold_r2s)),   4),
        })

    results_df = pd.DataFrame(summary_rows).sort_values("MAE (kg)")

    return {
        "results":            results_df,
        "feature_importance": feature_importance,
        "predictions":        all_predictions,
        "df_used":            df_clean,
    }


# STEP 3 — print results
def print_ml_results(ml_output: dict):
    """
        Pretty-print the cross-validation results table.
    """
    sep = "-" * 60
    print(sep)
    print("  ML Model Comparison — Predict Next Session Max Weight (kg)")
    print(sep)
    print(f"  Samples used : {len(ml_output['df_used'])}")
    print(f"  Features     : {len(FEATURE_COLS)}")
    print(f"  Validation   : TimeSeriesSplit (5-fold)")
    print(f"  Target       : next session max weight (kg) per exercise")
    print()
    print(ml_output["results"].to_string(index=False))
    print()

    best = ml_output["results"].iloc[0]["Model"]
    best_mae = ml_output["results"].iloc[0]["MAE (kg)"]
    print(f"  Best model: {best}  (MAE = {best_mae:.3f} kg)")
    print()

    print("  Feature importances (Random Forest):")
    fi = ml_output["feature_importance"].get("Random Forest")
    if fi is not None:
        for feat, imp in fi.items():
            bar = "█" * int(imp * 40)
            print(f"    {feat:<22} {imp:.4f}  {bar}")
    print(sep)


# STEP 4 — plot

def plot_ml_results(ml_output: dict, save_path: str | None = None):
    """
        Three-panel figure:
        Left   — MAE comparison bar chart across models
        Middle — Actual vs predicted scatter (best model)
        Right  — Feature importance (Random Forest)

        Parameters
        ----------
        ml_output : dict
            Output of train_and_evaluate().
        save_path : str or None
            If given, saves PNG to this path.

        Returns
        -------
        matplotlib.figure.Figure
    """
    results_df = ml_output["results"]
    fi_rf      = ml_output["feature_importance"].get("Random Forest")
    preds_rf   = ml_output["predictions"].get("Random Forest", [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.subplots_adjust(wspace=0.38)

    COLORS = ["#4C9BE8", "#F5A623", "#57C26A"]

    # Panel 1: MAE bar chart
    ax1 = axes[0]
    ax1.set_facecolor("white")
    models  = results_df["Model"].tolist()
    maes    = results_df["MAE (kg)"].tolist()
    mae_std = results_df["MAE std"].tolist()

    bars = ax1.bar(
        models, maes,
        color=COLORS[:len(models)],
        width=0.5,
        zorder=2,
        yerr=mae_std,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "#666666"},
    )
    for bar, val in zip(bars, maes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mae_std) * 0.1,
            f"{val:.2f} kg",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax1.set_title("Model Comparison\n(Mean Absolute Error, lower=better)",
                  fontsize=10, fontweight="bold", loc="left", pad=8)
    ax1.set_ylabel("MAE (kg)", fontsize=9)
    ax1.set_ylim(0, max(maes) * 1.35)
    ax1.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#cccccc")
    ax1.spines["bottom"].set_color("#cccccc")
    ax1.tick_params(axis="x", labelsize=8)

    # Panel 2: Actual vs Predicted scatter (Random Forest)s
    ax2 = axes[1]
    ax2.set_facecolor("white")

    if preds_rf:
        actuals   = [p[0] for p in preds_rf]
        predicted = [p[1] for p in preds_rf]
        ax2.scatter(actuals, predicted,
                    alpha=0.45, s=22, color="#4C9BE8", linewidths=0, zorder=2)

        # Perfect prediction line
        mn = min(min(actuals), min(predicted))
        mx = max(max(actuals), max(predicted))
        ax2.plot([mn, mx], [mn, mx], color="#E85D75", linewidth=1.5,
                 linestyle="--", zorder=3, label="Perfect prediction")

        ax2.set_xlabel("Actual max weight (kg)", fontsize=9)
        ax2.set_ylabel("Predicted max weight (kg)", fontsize=9)
        ax2.set_title("Actual vs Predicted\n(Random Forest, full-data fit)",
                      fontsize=10, fontweight="bold", loc="left", pad=8)
        ax2.legend(fontsize=8, frameon=False)

    ax2.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color("#cccccc")
    ax2.spines["bottom"].set_color("#cccccc")

    # Panel 3: Feature importances (Random Forest)
    ax3 = axes[2]
    ax3.set_facecolor("white")

    if fi_rf is not None:
        features = fi_rf.index.tolist()
        importances = fi_rf.values.tolist()
        y_pos = range(len(features))

        ax3.barh(
            list(y_pos), importances,
            color="#57C26A", height=0.6, zorder=2,
        )
        ax3.set_yticks(list(y_pos))
        ax3.set_yticklabels(
            [f.replace("_", " ") for f in features],
            fontsize=8
        )
        ax3.invert_yaxis()
        ax3.set_xlabel("Importance", fontsize=9)
        ax3.set_title("Feature Importance\n(Random Forest)",
                      fontsize=10, fontweight="bold", loc="left", pad=8)

    ax3.xaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax3.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    ax3.spines["left"].set_color("#cccccc")
    ax3.spines["bottom"].set_color("#cccccc")

    fig.suptitle(
        "ML Analysis — Strength Prediction",
        fontsize=13, fontweight="bold", y=1.02,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ML results plot saved: {save_path}")

    return fig


# STEP 5 — predict next session (inference utility)

def predict_next_session(df: pd.DataFrame, exercise: str) -> dict | None:
    """
        Train Random Forest on all available data for one exercise and predict
        the next session's expected max weight.

        This is a simple "what will I lift next time?" utility.

        Parameters
        ----------
        df : pd.DataFrame
            Output of build_ml_dataset().
        exercise : str
            Exercise name, e.g. "bench_press".

        Returns
        -------
        dict with predicted weight and last known weight, or None if insufficient data.
    """
    ex_df = df[df["exercise"] == exercise].dropna(subset=FEATURE_COLS + [TARGET_COL])
    ex_df = ex_df.sort_values("date").reset_index(drop=True)

    if len(ex_df) < 5:
        return None  # not enough history

    X = ex_df[FEATURE_COLS].values
    y = ex_df[TARGET_COL].values

    model = RandomForestRegressor(n_estimators=200, max_depth=6,
                                  min_samples_leaf=3, random_state=42)
    model.fit(X, y)

    # Use the most recent row as the feature vector for prediction
    last_row = ex_df.iloc[-1][FEATURE_COLS].values.reshape(1, -1)
    predicted = float(model.predict(last_row)[0])

    return {
        "exercise":        exercise,
        "last_known_max":  float(ex_df.iloc[-1]["next_max_kg"]),
        "predicted_next":  round(predicted, 2),
        "last_date":       str(ex_df.iloc[-1]["date"]),
    }


def print_next_session_predictions(df: pd.DataFrame):
    """
        Print next-session weight predictions for all tracked exercises.
    """
    sep = "-" * 52
    print(sep)
    print("  Next Session Strength Predictions (Random Forest)")
    print(sep)
    for ex in TRACKED_EXERCISES:
        result = predict_next_session(df, ex)
        if result:
            delta = result["predicted_next"] - result["last_known_max"]
            sign  = "+" if delta >= 0 else ""
            print(f"  {ex:<22}  last: {result['last_known_max']:>6.1f} kg  "
                  f"→  predicted: {result['predicted_next']:>6.2f} kg  "
                  f"({sign}{delta:.2f} kg)")
        else:
            print(f"  {ex:<22}  insufficient data")
    print(sep)

