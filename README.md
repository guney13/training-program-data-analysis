# training-program-data-analysis
A personal dataset of fitness program logs. Analysis of this dataset through different metrics.

Parses workout logs (W-NNN-X.txt format) and analyzes training patterns.
workouts2/ folder contains the logs of the fitness program

**Features:**
- Parses weight/rep data from structured text files
- Generates calendar heatmaps, weekly volume charts, and strength progression graphs
- Statistical hypothesis testing: correlation between training frequency and strength gains

**Usage:**
```bash
pip install -r requirements.txt
python main.py