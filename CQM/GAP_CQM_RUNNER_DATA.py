import os
import sys
import GAP_CQM

args = sys.argv[1:]

# Specify the directory path
directory_path = args[0]
default_time_limit = int(args[1])

# List all files and directories in the specified path
files_and_dirs = os.listdir(directory_path)
only_files = [
    f
    for f in os.listdir(directory_path)
    if os.path.isfile(os.path.join(directory_path, f))
]

for filename in only_files:
    filepath = f"{directory_path}/{filename}"
    print(f"File path : {filepath}")
    GAP_CQM.GAP_CQM_SOLVER(filepath, default_time_limit)
