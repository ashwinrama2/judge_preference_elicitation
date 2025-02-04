import os
import json

# List of folders to iterate through
folders = [
    'evaluate/judge_data_agg'
    ]

def count_json_entries_in_folders(folders):
    for folder in folders:
        print(f"Processing folder: {folder}")
        if not os.path.exists(folder):
            print(f"Folder '{folder}' does not exist. Skipping.")
            continue

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if filename.endswith('.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        print(f"{filename}: {len(data)} entries")
                    else:
                        print(f"{filename}: Not a list, skipping.")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        print()  # Blank line for better readability between folders

if __name__ == "__main__":
    count_json_entries_in_folders(folders)

