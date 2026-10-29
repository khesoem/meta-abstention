import logging
import datetime
import multiprocessing as mp
import meta_abstention.config as conf
import os
from meta_abstention.code_generation.data_manipulation import run as run_data_manipulation
from meta_abstention.code_generation.data_manipulation import run_repair as run_repair
from meta_abstention.code_generation.code_completion import run as run_code_completion
from meta_abstention.code_generation.test_runner import run as run_test_runner
from meta_abstention.code_generation.analyze_results import run as run_analyzer
from meta_abstention.xcodeeval.code_translation import run_translation as run_code_translation
from meta_abstention.xcodeeval.perturbation.llm_perturbation import run_llm_perturbation as run_llm_perturbation
from meta_abstention.xcodeeval.code_execution import execute_translated_code as execute_translated_code
from meta_abstention.xcodeeval.code_execution import execute_generated_tests as execute_generated_tests
from meta_abstention.xcodeeval.confidence_analysis import run_confidence_analysis_for_batch as run_confidence_analysis_for_batch, run_confidence_analysis as run_confidence_analysis
from meta_abstention.xcodeeval.confidence_measurement import compute_similarities as compute_similarities
from meta_abstention.xcodeeval.confidence_measurement import compute_confidence as compute_confidence
from meta_abstention.xcodeeval.extract_submissions import run_select_submissions as run_select_submissions
from meta_abstention.xcodeeval.test_generation import run_test_generation as run_test_generation
from meta_abstention.xcodeeval.data_cleaning import clean_non_translated_submissions as clean_non_translated_submissions

def run_perturbation_by_llms() -> None:
    # pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    pool = mp.Pool(processes=1)
    for model in ['qwen/qwen3-coder-next', 'openai/gpt-4.1-nano', 'deepseek/deepseek-v4-flash', 'meta-llama/llama-3.1-8b-instruct']:
        if 'llama-3.1-8b-instruct' not in model:
            continue

        dir = model.split('/')[1]
        for s_lang in conf.translation['langs'].keys():

            pool.apply_async(
                run_llm_perturbation,
                args=(
                    f'data/code_translation/xcodeeval/original_submissions/{s_lang}-original-submissions.json', 
                    f'data/code_translation/xcodeeval/{dir}/perturbations/{s_lang}-perturbations.json', 
                    model, 
                    5))
    pool.close()
    pool.join()

def _run_code_translation() -> None:
    model = 'qwen/qwen3-coder-next'
    for s_lang in conf.translation['langs'].keys():
        for t_lang in conf.translation['langs'].keys():
            if s_lang == t_lang:
                continue
            dir = model.split('/')[1]
            if os.path.exists(f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json'):
                continue
            new_translations = run_code_translation(f'data/code_translation/xcodeeval/original_submissions/{s_lang}-original-submissions.json', f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', conf.translation['langs'][t_lang], model=model)
            tries = 1
            while new_translations and tries < conf.translation['max-tries']:
                new_translations = run_code_translation(f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', conf.translation['langs'][t_lang], model=model, temp=1.0)
                tries += 1

def _run_code_translation_for_perturbations() -> None:
    for model in ['qwen/qwen3-coder-next', 'openai/gpt-4.1-nano', 'deepseek/deepseek-v4-flash']:
        if 'gpt-4.1-nano' not in model:
            continue

        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue

                dir = model.split('/')[1]
                perturbations_file = f'data/code_translation/xcodeeval/{dir}/perturbations/{s_lang}-perturbations.json'
                perturbations_execution_file = f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-perturbations-to-{t_lang}.json'
                
                new_translations = run_code_translation(perturbations_file, perturbations_execution_file, conf.translation['langs'][t_lang], model=model, get_logprobs=False)
                
                tries = 1
                while new_translations and tries < conf.translation['max-tries']:
                    new_translations = run_code_translation(perturbations_execution_file, perturbations_execution_file, conf.translation['langs'][t_lang], model=model, temp=1.0, get_logprobs=False)
                    tries += 1

def _run_analysis() -> None:
    for n_perturbations in range(1, 6):
        for model in ['qwen3-coder-next', 'llama-3.1-8b-instruct', 'gpt-4.1-nano', 'deepseek-v4-flash']:
            if model != 'gpt-4.1-nano':
                continue

            perturbations_dir = f'data/code_translation/xcodeeval/{model}/translations/{n_perturbations}-perturbations'
            if not os.path.exists(perturbations_dir):
                os.makedirs(perturbations_dir)
            analysis_dir = f'data/code_translation/xcodeeval/{model}/analysis_results/{n_perturbations}-perturbations'
            if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)

            for s_lang in conf.translation['langs'].keys():
                for t_lang in conf.translation['langs'].keys():
                    if s_lang == t_lang:
                        continue
                    compute_confidence(
                        f'data/code_translation/xcodeeval/{model}/similarities/{s_lang}-to-{t_lang}-similarities.json',
                        f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-to-{t_lang}-developer-made-alternative-exec-results.json', 
                        f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', 
                        f'{perturbations_dir}/{s_lang}-to-{t_lang}.json',
                        n_perturbations=n_perturbations)
            
            for s_lang in conf.translation['langs'].keys():
                for t_lang in conf.translation['langs'].keys():
                    if s_lang == t_lang:
                        continue
                    run_confidence_analysis([f'{perturbations_dir}/{s_lang}-to-{t_lang}.json'],
                        output_file=f'{analysis_dir}/{s_lang}-to-{t_lang}.txt')

            run_confidence_analysis_for_batch(perturbations_dir, output_file=f'{analysis_dir}/overall_results.txt')

def _run_analysis_for_perturbations() -> None:
    for n_perturbations in range(5, 6):
        for model in ['qwen3-coder-next', 'gpt-4.1-nano', 'deepseek-v4-flash']:
            # if model != 'deepseek-v4-flash':
            #     continue

            perturbations_dir = f'data/code_translation/xcodeeval/{model}/translations/{n_perturbations}-perturbations'
            if not os.path.exists(perturbations_dir):
                os.makedirs(perturbations_dir)
            analysis_dir = f'data/code_translation/xcodeeval/{model}/analysis_results/{n_perturbations}-perturbations'
            if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)

            for s_lang in conf.translation['langs'].keys():
                for t_lang in conf.translation['langs'].keys():
                    if s_lang == t_lang:
                        continue

                    compute_confidence(
                        f'data/code_translation/xcodeeval/{model}/similarities/{s_lang}-to-{t_lang}-similarities.json',
                        f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-to-{t_lang}-llm-made-alternative-exec-results.json', 
                        f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-perturbations-to-{t_lang}.json', 
                        f'{perturbations_dir}/{s_lang}-perturbations-to-{t_lang}.json',
                        n_perturbations=n_perturbations)


            for s_lang in conf.translation['langs'].keys():
                for t_lang in conf.translation['langs'].keys():
                    if s_lang == t_lang:
                        continue

                    run_confidence_analysis([f'{perturbations_dir}/{s_lang}-perturbations-to-{t_lang}.json'],
                        output_file=f'{analysis_dir}/{s_lang}-perturbations-to-{t_lang}.txt',
                        ignore_token_probability_metrics=model == 'gpt-4.1-nano')

            run_confidence_analysis_for_batch(
                perturbations_dir,
                output_file=f'{analysis_dir}/overall_perturbations_results.txt',
                perturbation=True,
                ignore_token_probability_metrics=model == 'gpt-4.1-nano',
            )

def generate_tests() -> None:
    for model in ['qwen/qwen3-coder-next', 'openai/gpt-4.1-nano', 'deepseek/deepseek-v4-flash']:
        if model != 'openai/gpt-4.1-nano':
            continue

        for s_lang in conf.translation['langs'].keys():
            dir = model.split('/')[1]

            run_test_generation(
                f'data/code_translation/xcodeeval/original_submissions/{s_lang}-original-submissions.json',
                f'data/code_translation/xcodeeval/{dir}/generated_tests/{s_lang}-generated-tests.json', 
                model=model
            )

def run_execute_translated_code() -> None:
    pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    # pool = mp.Pool(processes=1)
    for s_lang in conf.translation['langs'].keys():
        for t_lang in conf.translation['langs'].keys():
            if s_lang == t_lang:
                continue
            pool.apply_async(execute_translated_code, args=(f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', False, conf.translation['langs'][t_lang]))
            # execute_translated_code(f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', lang=conf.translation['langs'][t_lang])
    pool.close()
    pool.join()

def _execute_translated_code_for_perturbations() -> None:
    # pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    pool = mp.Pool(processes=6)
    for model in ['qwen3-coder-next', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        if model != 'gpt-4.1-nano':
            continue

        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue

                perturbations_execution_file = f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-perturbations-to-{t_lang}.json'
                
                if not os.path.exists(perturbations_execution_file):
                    continue
                
                pool.apply_async(
                    execute_translated_code,
                    args=(
                        perturbations_execution_file,
                        perturbations_execution_file,
                        False,
                        conf.translation['langs'][t_lang]))
    pool.close()
    pool.join()

def _execute_generated_tests() -> None:
    # pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    pool = mp.Pool(processes=12)
    for model in ['qwen3-coder-next', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        if model == 'qwen3-coder-next':
            continue

        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue

                pool.apply_async(execute_generated_tests, args=(
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-to-{t_lang}.json',
                    f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-generated-tests.json', 
                    f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-to-{t_lang}-developer-made-alternative-exec-results.json'))
    pool.close()
    pool.join()

def _execute_generated_tests_for_perturbations() -> None:
    # pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    pool = mp.Pool(processes=12)
    for model in ['qwen3-coder-next', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        if model == 'qwen3-coder-next':
            continue

        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue

                pool.apply_async(execute_generated_tests, args=(
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-perturbations-to-{t_lang}.json',
                    f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-generated-tests.json', 
                    f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-to-{t_lang}-llm-made-alternative-exec-results.json'),
                    kwds={'alternative_type': 'llm-made'})
    pool.close()
    pool.join()

def _execute_submission_cleaning() -> None:
    for model in ['qwen3-coder-next', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        if model != 'gpt-4.1-nano':
            continue

        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue
                clean_non_translated_submissions(
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-to-{t_lang}.json',
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-perturbations-to-{t_lang}.json',
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-perturbations-to-{t_lang}.json'
                )

def _execute_similarity_computation(perturbations: bool = False) -> None:
    for model in ['qwen3-coder-next', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue

                compute_similarities(
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}{"-perturbations" if perturbations else ""}-to-{t_lang}.json',
                    f'data/code_translation/xcodeeval/{model}/similarities/{s_lang}{"-perturbations" if perturbations else ""}-to-{t_lang}-similarities.json',
                    0,
                    s_lang,
                    t_lang
                )

def run_translation_experiments() -> None:
    # run_select_submissions('java', 'data/code_translation/xcodeeval/java-original-submissions.json', 6)

    # run_perturbation_by_llms()

    # _run_code_translation()

    # _run_code_translation_for_perturbations()

    # generate_tests()

    # run_execute_translated_code()

    # _execute_translated_code_for_perturbations()

    # _execute_generated_tests()

    # _execute_generated_tests_for_perturbations()

    # _execute_submission_cleaning()

    # _execute_similarity_computation(perturbations=False)
    
    # _execute_similarity_computation(perturbations=True)

    # _run_analysis()

    _run_analysis_for_perturbations()