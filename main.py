


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

    results = test_frequency_vs_strength(data)
    print_hypothesis_results(results)

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


