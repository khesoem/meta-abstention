import json
import os
from meta_abstention.xcodeeval.utils import execute_code
import logging

# Read logs/logging_2026-08-10-18-45.log into a string
with open('logs/logging_2026-08-10-18-45.log', 'r') as f:
    log_string = f.read()

def run_tests():
    with open('data/code_translation/xcodeeval/selected_python.json', 'r') as f:
        data = json.load(f)

        filtered_data = {}

        for src_uid, item in data.items():
            filtered_submissions = []
            for submission in item['submissions']:
                if f"Failed for {submission['code_uid']}" not in log_string:
                    filtered_submissions.append(submission)
            
            if len(filtered_submissions) >= 6:
                item['submissions'] = filtered_submissions
                filtered_data[src_uid] = item

        with open('data/code_translation/xcodeeval/selected_python_filtered.json', 'w') as f2:
            json.dump(filtered_data, f2, indent=2)