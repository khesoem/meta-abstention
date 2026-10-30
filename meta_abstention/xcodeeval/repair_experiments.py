import logging
import multiprocessing as mp
import meta_abstention.config as conf
import os
from meta_abstention.code_generation.data_manipulation import run as run_data_manipulation
from meta_abstention.code_generation.data_manipulation import run_repair as run_repair
from meta_abstention.code_generation.code_completion import run as run_code_completion
from meta_abstention.code_generation.test_runner import run as run_test_runner
from meta_abstention.code_generation.analyze_results import run as run_analyzer
from meta_abstention.xcodeeval.code_translation import run_translation as run_code_translation
from meta_abstention.xcodeeval.code_repair import run_repair as run_code_repair
from meta_abstention.xcodeeval.perturbation.llm_perturbation import run_llm_perturbation as run_llm_perturbation
from meta_abstention.xcodeeval.code_execution import execute_translated_code as execute_translated_code
from meta_abstention.xcodeeval.code_execution import execute_generated_tests as execute_generated_tests
from meta_abstention.xcodeeval.code_execution import execute_repaired_generated_tests as execute_repaired_generated_tests
from meta_abstention.xcodeeval.code_execution import execute_repaired_code as execute_repaired_code
from meta_abstention.xcodeeval.confidence_analysis import run_confidence_analysis_for_batch as run_confidence_analysis_for_batch, run_confidence_analysis as run_confidence_analysis
from meta_abstention.xcodeeval.confidence_measurement import compute_similarities as compute_similarities
from meta_abstention.xcodeeval.confidence_measurement import compute_confidence as compute_confidence
from meta_abstention.xcodeeval.extract_submissions import run_select_submissions as run_select_submissions
from meta_abstention.xcodeeval.test_generation import run_test_generation as run_test_generation
from meta_abstention.xcodeeval.test_generation import run_apr_test_generation as run_apr_test_generation
from meta_abstention.xcodeeval.data_cleaning import clean_non_translated_submissions as clean_non_translated_submissions

MODELS = ['qwen/qwen3-coder-next']

def _run_perturbation_by_llms_for_apr() -> None:
    pool = mp.Pool(processes=3)
    for model in MODELS:
        
        output_dir = f'data/apr/xcodeeval/{model.split('/')[1]}/perturbations'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pool.apply_async(
            run_llm_perturbation,
            args=(
                f'data/apr/xcodeeval/repair-submissions.json', 
                f'{output_dir}/perturbed-submissions.json',
                model, 
                5))

    pool.close()
    pool.join()

def _run_repair_submissions() -> None:
    for model in MODELS:

        model_dir = model.split('/')[1]
        perturbations_file = f'data/apr/xcodeeval/{model_dir}/perturbations/perturbed-submissions.json'
        repairs_file = f'data/apr/xcodeeval/{model_dir}/repairs/repaired-submissions.json'

        if not os.path.exists(perturbations_file):
            logging.warning(f"Skipping repair for {model}: {perturbations_file} does not exist")
            continue

        new_repairs = run_code_repair(perturbations_file, repairs_file, model=model)

        tries = 1
        while new_repairs and tries < conf.translation['max-tries']:
            new_repairs = run_code_repair(repairs_file, repairs_file, model=model, temp=0.5 * tries)
            tries += 1

def _run_test_generation() -> None:
    for model in MODELS:
        model_dir = model.split('/')[1]
        repairs_file = f'data/apr/xcodeeval/{model_dir}/repairs/repaired-submissions.json'
        generated_tests_file = f'data/apr/xcodeeval/{model_dir}/generated_tests/generated-tests.json'

        if not os.path.exists(repairs_file):
            logging.warning(f"Skipping test generation for {model}: {repairs_file} does not exist")
            continue

        run_apr_test_generation(repairs_file, generated_tests_file, model)

def _run_generated_test_execution() -> None:
    for model in MODELS:
        model_dir = model.split('/')[1]
        repairs_file = f'data/apr/xcodeeval/{model_dir}/repairs/repaired-submissions.json'
        generated_tests_file = f'data/apr/xcodeeval/{model_dir}/generated_tests/generated-tests.json'
        output_file = f'data/apr/xcodeeval/{model_dir}/generated_tests/repair-exec-results.json'

        if not os.path.exists(repairs_file) or not os.path.exists(generated_tests_file):
            logging.warning(f"Skipping test execution for {model}: repairs or generated tests are missing")
            continue

        execute_repaired_generated_tests(repairs_file, generated_tests_file, output_file)

def _run_original_test_execution() -> None:
    for model in MODELS:
        model_dir = model.split('/')[1]
        repairs_file = f'data/apr/xcodeeval/{model_dir}/repairs/repaired-submissions.json'

        if not os.path.exists(repairs_file):
            logging.warning(f"Skipping original test execution for {model}: {repairs_file} does not exist")
            continue

        execute_repaired_code(repairs_file, repairs_file, False)

def _run_confidence_analysis() -> None:
    for n_perturbations in range(5, 6):
        for model in MODELS:
            model_dir = model.split('/')[1]
            repairs_file = f'data/apr/xcodeeval/{model_dir}/repairs/repaired-submissions.json'
            generated_tests_exec_file = f'data/apr/xcodeeval/{model_dir}/generated_tests/repair-exec-results.json'
            if not os.path.exists(repairs_file) or not os.path.exists(generated_tests_exec_file):
                logging.warning(f"Skipping confidence analysis for {model}: repairs or generated-test execution results are missing")
                continue

            perturbations_dir = f'data/apr/xcodeeval/{model_dir}/repairs/{n_perturbations}-perturbations'
            if not os.path.exists(perturbations_dir):
                os.makedirs(perturbations_dir)
            analysis_dir = f'data/apr/xcodeeval/{model_dir}/analysis_results/{n_perturbations}-perturbations'
            if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)

            scored_repairs_file = f'{perturbations_dir}/repaired-submissions.json'
            compute_confidence(
                f'data/apr/xcodeeval/{model_dir}/similarities/similarities.json',
                generated_tests_exec_file,
                repairs_file,
                scored_repairs_file,
                n_perturbations=n_perturbations,
                use_repair=True,
            )

            run_confidence_analysis(
                [scored_repairs_file],
                output_file=f'{analysis_dir}/repaired-submissions.txt',
                use_repair=True,
            )

def run_apr_experiments() -> None:
    # _run_perturbation_by_llms_for_apr()

    # _run_repair_submissions()

    # _run_test_generation()

    # _run_generated_test_execution()

    # _run_original_test_execution()

    _run_confidence_analysis()