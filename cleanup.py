"""
Cleanup Utility

Simple utility to remove files from generated graph directories.
Used for cleaning up intermediate output during development.
"""

import os
import glob

# Path to your folder (update the path if necessary)
folder_path = "generated_graph"

# Find all files in the "generated_graph" folder
files = glob.glob(os.path.join(folder_path, "*"))

# Loop through and delete each file
for file_path in files:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("All files deleted from generated_graph folder.")
import os
import glob

# Path to your folder (update the path if necessary)
folder_path = "generated_graphs"

# Find all files in the "generated_graph" folder
files = glob.glob(os.path.join(folder_path, "*"))

# Loop through and delete each file
for file_path in files:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("All files deleted from generated_graph folder.")
