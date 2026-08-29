import json
from meta_abstention.xcodeeval.utils import execute_code
import logging

with open('logs/logging_2026-08-28-14-27.log', 'r') as f:
    log_str = f.read()

def execute_translated_code(translated_code_path: str, output_path: str, repeat_execution: bool = False):
    with open(translated_code_path, 'r') as f:
        data = json.load(f)

    for _, item in data.items():
        for submission in item['submissions']:

            for translation in submission['translation']:
                if 'exec_result' in translation and not repeat_execution:
                    continue
                if not f"ERROR Error executing code for {submission['code_uid']}" in log_str:
                    continue
                lang = translation['lang']
                translated_code = translation['translated_code']
                try:
                    exec_result = execute_code(lang, translated_code, item['unittests'], timeout=600)
                    translation['exec_result'] = exec_result
                    logging.info(f'Executed translated code for {submission['code_uid']}')
                except Exception as e:
                    translation['exec_result'] = f'Error: {str(e)}'
                    logging.error(f'Error executing code for {submission['code_uid']} to {lang}: {e}')

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)