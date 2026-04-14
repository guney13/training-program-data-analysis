# training-program-data-analysis

A personal dataset of fitness program logs. Analysis of this dataset through different metrics.\

Parses workout logs (W-NNN-X.txt format) and analyzes training patterns.\
workouts2/ folder contains the logs of the fitness program\


**Features:**\
- Parses weight/rep data from structured text files\
- Generates calendar heatmaps, weekly volume charts, and strength progression graphs\
- Statistical hypothesis testing: correlation between training frequency and strength gains\


**EDA:**\
workout_heatmap.png the heatmap of gym sessions over the given timeline\
weekly_volume.png shows the graph of weekly volume, each gym session type is shown\
strength_progression.png represents the lifted weight progression for specific exercises that targets different muscle groups\


**Hypothesis:**\
H0: no significant positive linear relationship between frequency and strength gain rate\
H1: higher frequency is positively associated with faster strength gain rate\


**Usage:**
```bash
pip install -r requirements.txt
python main.py