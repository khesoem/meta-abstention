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
        updated = False
        for submission in item['submissions']:

            if not 'translation' in submission or len(submission['translation']) == 0:
                logging.error(f'No translation for {submission['code_uid']}')
                continue

            for translation in submission['translation']:
                if 'exec_result' in translation and not isinstance(translation['exec_result'], str) and not repeat_execution:
                    continue

                lang = lang if lang else translation['lang']
                translated_code = translation['translated_code']
                updated = True
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

        if updated:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

def execute_repaired_code(repaired_code_path: str, output_path: str, repeat_execution: bool = False, lang: str = None):
    logging.info(f'Executing repaired code for {repaired_code_path} with lang {lang}')

    with open(repaired_code_path, 'r') as f:
        data = json.load(f)

    for _, item in data.items():
        updated = False
        for submission in item['submissions']:
            repair = submission.get('repair')
            if not repair or not repair.get('repaired_code'):
                logging.error(f"No repair for {submission['code_uid']}")
                continue

            if 'exec_result' in repair and not isinstance(repair['exec_result'], str) and not repeat_execution:
                continue

            exec_lang = lang if lang else repair.get('lang') or submission.get('lang')
            repaired_code = repair['repaired_code']
            updated = True
            try:
                exec_result = execute_code(exec_lang, repaired_code, item['unittests'], max(600, len(item['unittests']) * 5))
                repair['exec_result'] = exec_result
                logging.info(f"Executed repaired code for {submission['code_uid']} for {exec_lang}")
            except Exception as e:
                if 'Read timed out' in str(e):
                    repair['exec_result'] = {'data': [{
                                "exec_outcome": "TIMEOUT_ERROR",
                                "result": str(e),
                            }]}
                else:
                    repair['exec_result'] = f'Error: {str(e)}'
                    logging.error(f"Error executing repaired code for {submission['code_uid']}: {e}")

        if updated:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)

def execute_generated_tests(translated_code_path: str, generated_tests_path: str, output_path: str, minimum_number_of_tests: int = 1, 
    repeat_execution: bool = False, lang: str = None, alternative_type: str = 'developer-made', translation_index: int = 0):

    logging.info(f'Executing generated tests for {generated_tests_path} on {translated_code_path} with alternative type {alternative_type}')

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                execution_results = json.load(f)
        except Exception as e:
            logging.error(f'Error loading execution results from {output_path}: {e}')
            execution_results = {}
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
            code_uid = submission['code_uid']

            if alternative_type == 'developer-made':
                test_inputs = generated_tests[code_uid]
            elif alternative_type == 'llm-made':
                original_submission_id = code_uid.split('_')[0]
                test_inputs = generated_tests[original_submission_id]
            
            if alternative_type == 'llm-made':
                if f'{original_submission_id}_0' in execution_results and len(execution_results[f'{original_submission_id}_0']) == conf.translation['number-of-perturbations'] + 1:
                    execution_results[code_uid] = execution_results[f'{original_submission_id}_0']
                    continue

            rand.shuffle(submissions)
            if alternative_type == 'developer-made':
                considered_submissions = [s for s in submissions if s['code_uid'] != code_uid][:conf.translation['number-of-perturbations']]
                considered_submissions.append(submission)
            elif alternative_type == 'llm-made':
                considered_submissions = [s for s in submissions if s['code_uid'] != code_uid]
                considered_submissions.append(submission)

            if not code_uid in execution_results:
                execution_results[code_uid] = {}
            
            for executable_submission in considered_submissions:
                s_code_uid = executable_submission['code_uid']

                if not 'translation' in executable_submission or len(executable_submission['translation']) == 0:
                    logging.warning(f'No translation for {s_code_uid}')
                    continue

                if s_code_uid in execution_results[code_uid] and not isinstance(execution_results[code_uid][s_code_uid], str) and not repeat_execution:
                    continue

                if alternative_type == 'llm-made':
                    key = f'{original_submission_id}_0'
                    if (
                        key in execution_results
                        and s_code_uid in execution_results[key]
                        and not isinstance(execution_results[key][s_code_uid], str)
                        and not repeat_execution
                    ):
                        execution_results[code_uid][s_code_uid] = execution_results[key][s_code_uid]
                        logging.info(f'Executed generated tests for {code_uid} for {lang} from original submission {key}')
                        continue

                translation = executable_submission['translation'][translation_index]
                lang = lang if lang else translation['lang']
                translated_code = translation['translated_code']
                try:
                    if len(test_inputs) < minimum_number_of_tests:
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
                
    with open(output_path, 'w') as f:
        json.dump(execution_results, f, indent=4)

def _original_submission_id(code_uid: str) -> str:
    return f"{code_uid.rsplit('_', 1)[0]}_0"

def execute_repaired_generated_tests(repaired_code_path: str, generated_tests_path: str, output_path: str, minimum_number_of_tests: int = 0,
    repeat_execution: bool = False, lang: str = None):

    logging.info(f'Executing generated tests for {generated_tests_path} on repaired code in {repaired_code_path}')

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                execution_results = json.load(f)
        except Exception as e:
            logging.error(f'Error loading execution results from {output_path}: {e}')
            execution_results = {}
    else:
        execution_results = {}

    with open(repaired_code_path, 'r') as f:
        data = json.load(f)

    with open(generated_tests_path, 'r') as f:
        generated_tests = json.load(f)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rand = random.Random(conf.translation['seed'])
    for _, item in data.items():
        submissions = copy.deepcopy(item['submissions'])

        for submission in item['submissions']:
            code_uid = submission['code_uid']
            original_submission_id = _original_submission_id(code_uid)

            if original_submission_id not in generated_tests:
                logging.warning(f'No generated tests for {original_submission_id}')
                continue
            test_inputs = generated_tests[original_submission_id]

            if original_submission_id in execution_results and len(execution_results[original_submission_id]) == conf.translation['number-of-perturbations'] + 1:
                execution_results[code_uid] = execution_results[original_submission_id]
                continue

            rand.shuffle(submissions)
            considered_submissions = [s for s in submissions if s['code_uid'] != code_uid]
            considered_submissions.append(submission)

            if code_uid not in execution_results:
                execution_results[code_uid] = {}

            for executable_submission in considered_submissions:
                s_code_uid = executable_submission['code_uid']
                repair = executable_submission.get('repair') or {}
                repaired_code = repair.get('repaired_code')
                if not repaired_code:
                    logging.warning(f'No repair for {s_code_uid}')
                    continue

                if s_code_uid in execution_results[code_uid] and not isinstance(execution_results[code_uid][s_code_uid], str) and not repeat_execution:
                    continue

                if (
                    original_submission_id in execution_results
                    and s_code_uid in execution_results[original_submission_id]
                    and not isinstance(execution_results[original_submission_id][s_code_uid], str)
                    and not repeat_execution
                ):
                    execution_results[code_uid][s_code_uid] = execution_results[original_submission_id][s_code_uid]
                    logging.info(f'Executed generated tests for {code_uid} on repaired {s_code_uid} from original submission {original_submission_id}')
                    continue

                exec_lang = lang if lang else repair.get('lang') or executable_submission.get('lang')
                try:
                    if len(test_inputs) < minimum_number_of_tests:
                        raise ValueError(f'Not enough generated tests for {code_uid}')

                    unittests = [{'input': test_input, 'output': [""]} for test_input in test_inputs]
                    exec_result = execute_code(exec_lang, repaired_code, unittests, max(600, len(unittests) * 5))
                    execution_results[code_uid][s_code_uid] = exec_result
                    logging.info(f'Executed generated tests for {code_uid} on repaired {s_code_uid}')
                except Exception as e:
                    if 'Read timed out' in str(e):
                        execution_results[code_uid][s_code_uid] = {'data': [{
                                    "exec_outcome": "TIMEOUT_ERROR",
                                    "result": str(e),
                                }]}
                    else:
                        execution_results[code_uid][s_code_uid] = f'Error: {str(e)}'
                        logging.error(f'Error executing repaired code for {code_uid} on {s_code_uid}: {e}')

        with open(output_path, 'w') as f:
            json.dump(execution_results, f, indent=4)

    with open(output_path, 'w') as f:
        json.dump(execution_results, f, indent=4)