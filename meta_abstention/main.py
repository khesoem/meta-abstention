import logging
import datetime
import meta_abstention.config as conf

from meta_abstention.data.data_manipulation import run as run_data_manipulation
from meta_abstention.data.data_manipulation import run_repair as run_repair
from meta_abstention.completion.code_completion import run as run_code_completion
from meta_abstention.evaluation.test_runner import run as run_test_runner
from meta_abstention.analysis.analyze_results import run as run_analyzer

# logging.basicConfig(filename='logs/logging_{:%Y-%m-%d-%H-%M}.log'.format(datetime.datetime.now()),
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)

def main() -> None:
    # run_data_manipulation()
    run_repair(to_be_repaired=['HumanEval/50', 'HumanEval/38', 'HumanEval/32', 'HumanEval/20', 'HumanEval/10'])
    # run_code_completion(model=conf.completion['model'])
    # run_test_runner()
    # run_analyzer()


if __name__ == "__main__":
    main()
