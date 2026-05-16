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
- workout_heatmap.png the heatmap of gym sessions over the given timeline
- weekly_volume.png shows the graph of weekly volume, each gym session type is shown
- strength_progression.png represents the lifted weight progression for specific exercises that targets different muscle groups
- rep_quality_index.png tracks the Rep Quality Index (RQI) over time for each tracked exercise. 
RQI is defined as last_set_reps / first_set_reps (clamped to [0, 1]) and captures how cleanly a load is being lifted - a value of 1.0 means no fatigue driven rep drop across the session, while lower values indicate progressive failure. 
Each subplot shows raw per session RQI as scatter points, a rolling average with ±1 SD band, and a stats box with mean, std, and percentage of perfect sessions. Comparing RQI trends against the strength progression chart reveals whether weight increases reflect true strength gain or overlifting.
- rest_gap_distribution.png analyzes the number of days between consecutive training sessions. 
The top panel is a histogram of gap lengths colour coded by category (back-to-back, 1 rest day, 2 rest days, long break +4 days), with median and mean lines lying on them. 
The bottom panel is a chronological stem chart of every individual gap, allowing to spot periods of dense training or extended rests throughout the time range.

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