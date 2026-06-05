import json
import logging
import os
import subprocess
import tempfile

from datasets import load_dataset

import meta_abstention.config as conf

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds per test execution


def _run_test(prompt: str, generated_code: str, test: str, entry_point: str) -> bool:
    program = prompt + generated_code + '\n\n' + test + '\n\ncheck(' + entry_point + ')'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['python', tmp_path],
            timeout=_TIMEOUT,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("Timeout on %s", entry_point)
        return False
    finally:
        os.unlink(tmp_path)


def run(
    completions_path: str = os.path.join(conf.data['output-dir'], conf.completion['output-file']),
) -> None:
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    task_meta = {
        t['task_id']: {'test': t['test'], 'entry_point': t['entry_point']}
        for t in dataset
    }

    with open(completions_path) as f:
        tasks = json.load(f)

    for task in tasks:
        task_id = task['task_id']
        meta = task_meta.get(task_id)
        if not meta:
            logger.warning("No test metadata found for %s, skipping", task_id)
            continue

        for completion in task['completions']:
            prompt_type = completion['prompt_type']
            logger.info("Testing %s / %s", task_id, prompt_type)
            passed = _run_test(
                completion['prompt'],
                completion['generated_code'],
                meta['test'],
                meta['entry_point'],
            )
            completion['passed'] = passed
            logger.info("  -> %s", "PASS" if passed else "FAIL")

    with open(completions_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    logger.info("Test results written to %s", completions_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
