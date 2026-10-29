import logging
import datetime

from meta_abstention.xcodeeval.translation_experiments import run_translation_experiments as run_translation_experiments
from meta_abstention.xcodeeval.repair_experiments import run_apr_experiments as run_apr_experiments

logging.basicConfig(filename='logs/logging_{:%Y-%m-%d-%H-%M}.log'.format(datetime.datetime.now()),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

def main() -> None:
    # _run_translation_experiments()
    run_apr_experiments()

if __name__ == "__main__":
    main()
