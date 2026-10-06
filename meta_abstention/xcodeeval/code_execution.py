import json
from meta_abstention.xcodeeval.utils import execute_code
import logging
from meta_abstention import config as conf
import random
import copy
import os

def execute_translated_code(translated_code_path: str, output_path: str, repeat_execution: bool = False, lang: str = None):
    logging.info(f'Executing translated code for {translated_code_path} with lang {lang}')

    with open(translated_code_path, 'r') as f:
        data = json.load(f)

    for _, item in data.items():
        for submission in item['submissions']:

            for translation in submission['translation']:
                if 'exec_result' in translation and not isinstance(translation['exec_result'], str) and not repeat_execution:
                    continue

                lang = lang if lang else translation['lang']
                translated_code = translation['translated_code']
                try:
                    exec_result = execute_code(lang, translated_code, item['unittests'], max(600, len(item['unittests']) * 5))
                    translation['exec_result'] = exec_result
                    logging.info(f'Executed translated code for {submission['code_uid']} for {lang}')
                except Exception as e:
                    if 'Read timed out' in str(e):
                        translation['exec_result'] = {'data': [{
                                    "exec_outcome": "TIMEOUT_ERROR",
                                    "result": str(e),
                                }]}
                    else:
                        translation['exec_result'] = f'Error: {str(e)}'
                        logging.error(f'Error executing code for {submission['code_uid']} to {lang}: {e}')

                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=4)

def execute_generated_tests(translated_code_path: str, generated_tests_path: str, output_path: str, number_of_considered_generated_tests: int = conf.translation['number-of-considered-generated-tests'], 
    repeat_execution: bool = False, lang: str = None, alternative_type: str = 'developer-made', translation_index: int = 0):

    logging.info(f'Executing generated tests for {generated_tests_path} with lang {lang} and alternative type {alternative_type}')

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            execution_results = json.load(f)
    else:
        execution_results = {}

    with open(translated_code_path, 'r') as f:
        data = json.load(f)

    with open(generated_tests_path, 'r') as f:
        generated_tests = json.load(f)

    rand = random.Random(conf.translation['seed'])
    for _, item in data.items():
        submissions = copy.deepcopy(item['submissions'])

        for submission in item['submissions']:
            test_inputs = generated_tests[submission['code_uid']]
            
            code_uid = submission['code_uid']
            rand.shuffle(submissions)
            if alternative_type == 'developer-made':
                considered_submissions = [s for s in submissions if s['code_uid'] != code_uid][:conf.translation['number-of-perturbations']]
                considered_submissions.append(submission)

            if not code_uid in execution_results:
                execution_results[code_uid] = {}
            
            for executable_submission in considered_submissions:
                s_code_uid = executable_submission['code_uid']
                if s_code_uid in execution_results[code_uid] and not isinstance(execution_results[code_uid][s_code_uid], str) and not repeat_execution:
                    continue

                translation = executable_submission['translation'][translation_index]
                lang = lang if lang else translation['lang']
                translated_code = translation['translated_code']
                try:
                    if len(test_inputs) < number_of_considered_generated_tests:
                        raise ValueError(f'Not enough generated tests for {code_uid}')
                    
                    unittests = [{'input': test_input, 'output': [""]} for test_input in test_inputs]
                    exec_result = execute_code(lang, translated_code, unittests, max(600, len(unittests) * 5))
                    execution_results[code_uid][s_code_uid] = exec_result
                    logging.info(f'Executed generated tests for {code_uid} for {lang}')
                except Exception as e:
                    if 'Read timed out' in str(e):
                        execution_results[code_uid][s_code_uid] = {'data': [{
                                    "exec_outcome": "TIMEOUT_ERROR",
                                    "result": str(e),
                                }]}
                    else:
                        execution_results[code_uid][s_code_uid] = f'Error: {str(e)}'
                        logging.error(f'Error executing code for {code_uid} to {lang}: {e}')

                with open(output_path, 'w') as f:
                    json.dump(execution_results, f, indent=4)