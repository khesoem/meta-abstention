import logging
import datetime
import meta_abstention.config as conf

from meta_abstention.code_generation.data_manipulation import run as run_data_manipulation
from meta_abstention.code_generation.data_manipulation import run_repair as run_repair
from meta_abstention.code_generation.code_completion import run as run_code_completion
from meta_abstention.code_generation.test_runner import run as run_test_runner
from meta_abstention.code_generation.analyze_results import run as run_analyzer
from meta_abstention.xcodeeval.dataset_sanity_check import run_tests as run_xcodeeval_dataset_sanity_check
from meta_abstention.xcodeeval.code_translation import run_translation as run_code_translation
from meta_abstention.xcodeeval.code_execution import execute_translated_code as execute_translated_code
from meta_abstention.xcodeeval.confidence_analysis import run_confidence_analysis as run_confidence_analysis
from meta_abstention.xcodeeval.confidence_measurement import compute_similarities as compute_similarities
from meta_abstention.xcodeeval.confidence_measurement import compute_confidence as compute_confidence
from meta_abstention.xcodeeval.extract_submissions import run_select_submissions as run_select_submissions

logging.basicConfig(filename='logs/logging_{:%Y-%m-%d-%H-%M}.log'.format(datetime.datetime.now()),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

def main() -> None:
    run_code_translation('data/code_translation/xcodeeval/cpp-to-java.json', 'data/code_translation/xcodeeval/cpp-to-java.json', 'Java 8')
    # execute_translated_code('data/code_translation/xcodeeval/python-to-java-executed2.json', 'data/code_translation/xcodeeval/python-to-java-executed2.json')
    # run_confidence_analysis('data/code_translation/xcodeeval/python-to-java-executed2.json')
    # compute_similarities('data/code_translation/xcodeeval/python-to-java-executed2.json', 'data/code_translation/xcodeeval/python-to-java-similarities2.json')
    # compute_confidence('data/code_translation/xcodeeval/python-to-java-similarities.json', 'data/code_translation/xcodeeval/python-to-java-executed.json', 'data/code_translation/xcodeeval/python-to-java-executed-3-perturbations.json', n_perturbations=3)
    # run_select_submissions('c++', 'data/code_translation/xcodeeval/cpp-original-submissions.json', 6)

if __name__ == "__main__":
    main()
