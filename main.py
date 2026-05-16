


import os


def count_txt_files(directory, option= ''):
    return sum(
        1 for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(option + '.txt')
    )
def getNameListOfFiles(directory, option= ''):
    listOfFileNames = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(option + '.txt'):
            listOfFileNames.append(f)

    return listOfFileNames




from log_parser import *
from eda import *

from hypothesis_test import test_frequency_vs_strength, print_hypothesis_results

from ml_models import (
    build_ml_dataset,
    train_and_evaluate,
    print_ml_results,
    plot_ml_results,
    print_next_session_predictions,
)



if __name__ == "__main__":
    directory_of_sessions = 'workouts2'
    print("Number of A sessions:", count_txt_files(directory_of_sessions, option= 'A'))
    print("Number of B sessions:", count_txt_files(directory_of_sessions, option= 'B'))
    print("Number of C sessions:", count_txt_files(directory_of_sessions, option= 'C'))
    print("Number of all sessions:", count_txt_files(directory_of_sessions))


    WORKOUT_DIR = Path("workouts2")
    data = load_workout_directory(WORKOUT_DIR)

    print(f"Loaded {len(data)} workout sessions.\n")
    
    for key in list(data.keys())[:1]:
        w = data[key]
        print(f"{'─'*50}")
        print(f"Key        : {key}")
        print(f"Date       : {w['date_str']}")
        print(f"Day #      : {w['day_number']}  Session: {w['session_type']}")
        print(f"Body weight: {w['body_weight_kg']} kg")
        print(f"kcal eaten : {w['kcal_eaten']}")
        print(f"Exercises  : {len(w['exercises'])}")
        for ex in w["exercises"]:
            print(f"  [{ex['name']}]")
            if ex["cardio_minutes"]:
                print(f"    cardio: {ex['cardio_minutes']} min")
            for s in ex["sets"]:
                wt = "bw" if s["bodyweight"] else f"{s['weight_kg']} kg"
                print(f"    {wt} x {s['reps']}")
        if w["parse_warnings"]:
            print(f"Warnings: {w['parse_warnings']}")
        print()


    fig = plot_workout_heatmap(data, save_path= "workout_heatmap.png")
    #plt.show()

    fig_vol = plot_weekly_volume(data, save_path= "weekly_volume.png")
    #plt.show()

    fig_strength = plot_strength_progression(data, save_path="strength_progression.png")
    #plt.show()

    fig_rqi = plot_rep_quality_index(data, save_path="rep_quality_index.png")
    #plt.show()

    fig_rest = plot_rest_gap_distribution(data, save_path="rest_gap_distribution.png")
    #plt.show()

    results = test_frequency_vs_strength(data)
    print_hypothesis_results(results)


    # applying ml methods
    # 1. Build the feature table from parsed workout data
    ml_df = build_ml_dataset(data)
    print(f"\nML dataset: {len(ml_df)} samples across {ml_df['exercise'].nunique()} exercises\n")
 
    # 2. Train Linear Regression, Random Forest, XGBoost and cross-validate
    ml_output = train_and_evaluate(ml_df, n_splits=5)
 
    # 3. Print comparison table + feature importances
    print_ml_results(ml_output)
 
    # 4. Save the three-panel plot
    fig_ml = plot_ml_results(ml_output, save_path="ml_results.png")
 
    # 5. Print next-session weight predictions for each exercise
    print_next_session_predictions(ml_df)
    #


    '''
    # Aggregate: collect all unique exercise names
    all_exercises = set()
    for w in data.values():
        for ex in w["exercises"]:
            all_exercises.add(ex["name"])

    print(f"\nUnique exercises found ({len(all_exercises)}):")
    for name in sorted(all_exercises):
        print(f"  {name}")

    '''    

    # Report files with parse warnings
    warned = [(k, v["parse_warnings"]) for k, v in data.items() if v["parse_warnings"]]
    print(f"\nFiles with parse warnings: {len(warned)}")
    for k, w in warned:
        print(f"  {k}: {w}")


