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
from meta_abstention.xcodeeval.code_execution import execute_translated_code as execute_translated_code
from meta_abstention.xcodeeval.code_execution import execute_generated_tests as execute_generated_tests
from meta_abstention.xcodeeval.confidence_analysis import run_confidence_analysis_for_batch as run_confidence_analysis_for_batch, run_confidence_analysis as run_confidence_analysis
from meta_abstention.xcodeeval.confidence_measurement import compute_similarities as compute_similarities
from meta_abstention.xcodeeval.confidence_measurement import compute_confidence as compute_confidence
from meta_abstention.xcodeeval.extract_submissions import run_select_submissions as run_select_submissions
from meta_abstention.xcodeeval.test_generation import run_test_generation as run_test_generation

# logging.basicConfig(filename='logs/logging_{:%Y-%m-%d-%H-%M}.log'.format(datetime.datetime.now()),
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)

def main() -> None:
    # run_select_submissions('java', 'data/code_translation/xcodeeval/java-original-submissions.json', 6)

    # model = 'qwen/qwen3-coder-next'
    # for s_lang in conf.translation['langs'].keys():
    #     for t_lang in conf.translation['langs'].keys():
    #         if s_lang == t_lang:
    #             continue
    #         dir = model.split('/')[1]
    #         if os.path.exists(f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json'):
    #             continue
    #         new_translations = run_code_translation(f'data/code_translation/xcodeeval/original_submissions/{s_lang}-original-submissions.json', f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', conf.translation['langs'][t_lang], model=model)
    #         tries = 1
    #         while new_translations and tries < conf.translation['max-tries']:
    #             new_translations = run_code_translation(f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/{dir}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', conf.translation['langs'][t_lang], model=model, temp=1.0)
    #             tries += 1

    # for model in ['qwen/qwen3-coder-next', 'openai/gpt-4.1-nano', 'deepseek/deepseek-v4-flash']:
    #     if model == 'qwen/qwen3-coder-next':
    #         continue

    #     for s_lang in conf.translation['langs'].keys():
    #         dir = model.split('/')[1]
    #         if os.path.exists(f'data/code_translation/xcodeeval/{dir}/generated_tests/{s_lang}-generated-tests.json'):
    #             continue
    #         run_test_generation(
    #             f'data/code_translation/xcodeeval/original_submissions/{s_lang}-original-submissions.json',
    #             f'data/code_translation/xcodeeval/{dir}/generated_tests/{s_lang}-generated-tests.json', 
    #             model=model
    #         )

    # pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    # # pool = mp.Pool(processes=1)
    # for s_lang in conf.translation['langs'].keys():
    #     for t_lang in conf.translation['langs'].keys():
    #         if s_lang == t_lang:
    #             continue
    #         pool.apply_async(execute_translated_code, args=(f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', False, conf.translation['langs'][t_lang]))
    #         # execute_translated_code(f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', lang=conf.translation['langs'][t_lang])
    # pool.close()
    # pool.join()

    # pool = mp.Pool(processes=conf.translation['num-execution-processes'])
    # # pool = mp.Pool(processes=1)
    # for s_lang in conf.translation['langs'].keys():
    #     for t_lang in conf.translation['langs'].keys():
    #         if s_lang == t_lang:
    #             continue
    #         pool.apply_async(execute_generated_tests, args=(
    #             f'data/code_translation/xcodeeval/qwen3-coder-next/translations/6-perturbations/{s_lang}-to-{t_lang}.json',
    #             f'data/code_translation/xcodeeval/qwen3-coder-next/generated_tests/{s_lang}-generated-tests.json', 
    #             f'data/code_translation/xcodeeval/qwen3-coder-next/generated_tests/{s_lang}-to-{t_lang}-developer-made-alternative-exec-results.json'))
    # pool.close()
    # pool.join()

    # for s_lang in conf.translation['langs'].keys():
    #     for t_lang in conf.translation['langs'].keys():
    #         if s_lang == t_lang:
    #             continue
    #         compute_similarities(f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/translations/6-perturbations/{s_lang}-to-{t_lang}.json', f'data/code_translation/xcodeeval/llama-3.1-8b-instruct/similarities/{s_lang}-to-{t_lang}-similarities.json', 0, s_lang, t_lang)
    
    for model in ['qwen3-coder-next', 'llama-3.1-8b-instruct', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        if model != 'qwen3-coder-next':
            continue
        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue
                compute_confidence(
                    f'data/code_translation/xcodeeval/{model}/similarities/{s_lang}-to-{t_lang}-similarities.json',
                    f'data/code_translation/xcodeeval/{model}/generated_tests/{s_lang}-to-{t_lang}-developer-made-alternative-exec-results.json', 
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-to-{t_lang}.json', 
                    f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-to-{t_lang}.json')
        
        for s_lang in conf.translation['langs'].keys():
            for t_lang in conf.translation['langs'].keys():
                if s_lang == t_lang:
                    continue
                run_confidence_analysis([f'data/code_translation/xcodeeval/{model}/translations/6-perturbations/{s_lang}-to-{t_lang}.json'],
                    output_file=f'data/code_translation/xcodeeval/{model}/analysis_results/{s_lang}-to-{t_lang}.txt')

        run_confidence_analysis_for_batch(f'data/code_translation/xcodeeval/{model}/translations/6-perturbations', output_file=f'data/code_translation/xcodeeval/{model}/analysis_results/overall_results.txt')

    # run_confidence_analysis([f'data/code_translation/xcodeeval/qwen3-coder-next/translations/6-perturbations/python-to-java.json'])
    # run_confidence_analysis_for_batch('data/code_translation/xcodeeval/qwen3-coder-next/translations/6-perturbations')
    

if __name__ == "__main__":
    main()
