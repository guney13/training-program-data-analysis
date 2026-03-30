


import os 

def count_txt_files(directory, option= ''):
    return sum(
        1 for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(option + '.txt')
    )

directory_of_sessions = 'workouts2'
print("Number of A sessions:", count_txt_files(directory_of_sessions, option= 'A'))
print("Number of B sessions:", count_txt_files(directory_of_sessions, option= 'B'))
print("Number of C sessions:", count_txt_files(directory_of_sessions, option= 'C'))

print("Number of all sessions:", count_txt_files(directory_of_sessions))

