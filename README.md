# training-program-data-analysis

A personal dataset of fitness program logs. Analysis of this dataset through different metrics.

Parses workout logs (W-NNN-X.txt format) and analyzes training patterns.
workouts2/ folder contains the logs of the fitness program


**Features:**
- Parses weight/rep data from structured text files
- Generates calendar heatmaps, weekly volume charts, and strength progression graphs
- Statistical hypothesis testing: correlation between training frequency and strength gains
- Machine learning models to predict next session max weight for tracked exercises


**EDA:**
* `workout_heatmap.png`: The heatmap of gym sessions over the given timeline
* `weekly_volume.png`: Shows the graph of weekly volume, broken down by session type
* `strength_progression.png`: Represents the lifted weight progression for specific exercises targeting different muscle groups
* `rep_quality_index.png`: Tracks the Rep Quality Index (RQI) over time for each tracked exercise
  * Defined as last_set_reps / first_set_reps (normalized to [0, 1]). It captures how cleanly a load is being lifted (1.0 means no fatigue driven rep drop; lower values indicate progressive failure)
  * Each subplot shows raw per session RQI scatter points, a rolling average with a ±1 SD band, and a statistics box (mean, std, % of perfect sessions)
  * Comparing RQI trends against strength progression reveals whether weight increases reflect true strength gains or overlifting
* `rest_gap_distribution.png`: Analyzes the number of days between consecutive training sessions
  * **Top Section:** A histogram of gap lengths color coded by category (back-to-back, 1 rest day, 2 rest days, long break 4+ days) with median and mean lines
  * **Bottom Section:** A chronological stem chart of individual gaps, highlighting periods of dense training or extended rests


**Hypothesis:**
* H0: no significant positive linear relationship between frequency and strength gain rate
* H1: higher frequency is positively associated with faster strength gain rate


**ML:**
* Task: regression - predict the next session's max weight (kg) for each tracked exercise
* Tracked exercises: bench_press, db_shoulder_press, leg_extension, lat_pulldown
* Models compared: Linear Regression (baseline), Random Forest, XGBoost
* Validation: TimeSeriesSplit cross-validation
* Output: ml_results.png - model comparison, actual vs predicted scatter, feature importances

**Usage:**
```bash
pip install -r requirements.txt
python main.py
```