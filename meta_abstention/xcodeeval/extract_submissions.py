import os
import json
import requests
from meta_abstention import config
from meta_abstention.xcodeeval.utils import execute_code
import logging

def _extract_submissions(target_lang: str):
    xCodeEval_path = config.xcodeeval['path']
    code_translation_submissions = os.path.join(xCodeEval_path, "code_translation")

    with open(os.path.join(xCodeEval_path, "unittest_db.json"), "r") as f:
        unittest_db = json.load(f)

    problem_descriptions = {}
    with open(os.path.join(xCodeEval_path, "problem_descriptions.jsonl"), "r") as f:
        for line in f:
            data = json.loads(line)
            problem_descriptions[data['src_uid']] = data

    selected_submissions = {}
    # Read jsonl files from 'validation' and 'test' folders in code_translation_submissions
    for folder in os.listdir(code_translation_submissions):
        if folder in ['validation', 'test']:
            for file in os.listdir(os.path.join(code_translation_submissions, folder)):
                if target_lang != file.split('.')[0].lower():
                    continue
                if file.endswith(".jsonl"):
                    with open(os.path.join(code_translation_submissions, folder, file), "r") as f:
                        for line in f:
                            data = json.loads(line)
                            lang = data['lang_cluster']
                            src_uid = data['src_uid']

                            if target_lang != lang.lower():
                                continue

                            if not src_uid in selected_submissions:
                                selected_submissions[src_uid] = {
                                    'problem_description': problem_descriptions[src_uid],
                                    'submissions': [],
                                    'unittests': unittest_db[src_uid],
                                }
                            selected_submissions[src_uid]['submissions'].append(data)

    return selected_submissions

def _filter_submissions(selected_submissions: dict, min_submissions: int = 6):
    to_be_ignored_problems = set()

    submission_codes = set()

    for src_uid, data in selected_submissions.items():
        if len(data['submissions']) < min_submissions:
            to_be_ignored_problems.add(src_uid)
            continue

        unittests = data['unittests']
        successful_submissions = []
        for submission in data['submissions']:
            if submission['source_code'] in submission_codes:
                continue
            submission_codes.add(submission['source_code'])

            try:
                exec_result = execute_code(submission['lang'], submission['source_code'], unittests)
                logging.info(f"Executed code for {submission['code_uid']} with {len(exec_result['data'])} items")
                if all(item['exec_outcome'] == 'PASSED' for item in exec_result['data']):
                    successful_submissions.append(submission)
                    logging.info(f"Code for {submission['code_uid']} is successful")

            except Exception as e:
                logging.error(f"Error executing code for {submission['code_uid']}: {e}")
        
        if len(successful_submissions) < min_submissions:
            to_be_ignored_problems.add(src_uid)
        
        data['submissions'] = successful_submissions

    selected_submissions = {k: v for k, v in selected_submissions.items() if k not in to_be_ignored_problems}

    return selected_submissions

def run_select_submissions(lang: str, output_path: str, min_submissions: int = 6):
    
    selected_submissions = _extract_submissions(lang)

    selected_submissions = _filter_submissions(selected_submissions, min_submissions)

    with open(output_path, 'w') as f:
        json.dump(selected_submissions, f, indent=2)