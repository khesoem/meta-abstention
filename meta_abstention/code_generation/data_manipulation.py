import json
import logging
import os
import re

from datasets import load_dataset

import meta_abstention.config as conf
from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt

logger = logging.getLogger(__name__)

_PARAPHRASE_SYSTEM = "You are a technical writing assistant specializing in Python documentation."

_PARAPHRASE_USER_TEMPLATE = """\
Produce {n} paraphrased variants of the Python code block below. Each variant must:
- Preserve every function present — do not add, remove, or reorder functions.
- Keep all import statements exactly as-is.
- Keep every function signature (name, parameters, return type annotation) exactly as-is.
- Keep all doctest examples (>>> and ...) exactly as-is.
- Rephrase only the natural language description text within each docstring.
- If the program contains more than one function, keep the complete implementation (body) of all functions except the last one exactly as-is. For the last function only, stop after its closing docstring \"\"\" and do not include its implementation.
- Make the {n} variants meaningfully different from one another.

Return a JSON array of {n} strings. No markdown fences, no other text.

{prompt}"""


def _parse_variants(raw: str, n: int) -> list[str]:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    variants = json.loads(raw)
    if len(variants) != n:
        logger.warning("Expected %d variants, got %d", n, len(variants))
    return variants


def _paraphrase_all(adapter: LLMAdapter, original_prompt: str, n_variants: int) -> list[str]:
    messages = [
        Prompt.Message("system", _PARAPHRASE_SYSTEM),
        Prompt.Message("user", _PARAPHRASE_USER_TEMPLATE.format(n=n_variants, prompt=original_prompt)),
    ]
    raw = adapter.get_response(Prompt(messages)).first_content
    return _parse_variants(raw, n_variants)

def _prompt_semantic_equivalence_sanity_check(p1: str, p2: str) -> bool:
    p1_parts, p2_parts = re.split('\'\'\'|"""', p1.strip()), re.split('\'\'\'|"""', p2.strip())
    
    p1_code = '\n'.join(p1_parts[::2]).strip()
    p2_code = '\n'.join(p2_parts[::2]).strip()
    p1_code = p1_code.replace('\n\n\n', '\n\n')
    p2_code = p2_code.replace('\n\n\n', '\n\n')
    
    return p1_code == p2_code or len(p2_code) > len(p1_code) * 2

def run(
    n_variants: int = conf.data['n-variants-per-task'],
    model: str = conf.llm['default-model'],
    output_dir: str = conf.data['output-dir'],
    output_file: str = conf.data['humanevalplus-output-file'],
) -> list[dict]:
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=model)

    results = []
    for task in dataset:
        task_id = task['task_id']
        original = task['prompt']
        logger.info("Paraphrasing %s (%d variants)", task_id, n_variants)
        try:
            variants = _paraphrase_all(adapter, original, n_variants)
        except Exception as e:
            logger.error("Error paraphrasing %s: %s", task_id, e)
            continue
        results.append({
            'task_id': task_id,
            'original_prompt': original,
            'modified_prompts': variants,
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d tasks to %s", len(results), output_path)

    return results

def run_repair(
    n_variants: int = conf.data['n-variants-per-task'],
    model: str = conf.llm['default-model'],
    output_dir: str = conf.data['output-dir'],
    output_file: str = conf.data['humanevalplus-output-file'],
    to_be_repaired: list[str] = [],
) -> list[dict]:

    with open(f'{conf.data["output-dir"]}/{conf.data["humanevalplus-output-file"]}', 'r') as f:
        dataset = json.load(f)
    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=model)

    # for items whose 'task_id' is in to_be_repaired, generate the variants again using the model
    for task_id in to_be_repaired:
        task = next((item for item in dataset if item['task_id'] == task_id), None)
        if task is None:
            logger.error("Task %s not found", task_id)
            continue
        logger.info("Repairing %s", task_id)
        try:
            repaired = _paraphrase_all(adapter, task['original_prompt'], n_variants)
        except Exception as e:
            logger.error("Error repairing %s: %s", task_id, e)
            continue    
        old_item = next((item for item in dataset if item['task_id'] == task_id), None)
        if old_item is None:
            logger.error("Task %s not found", task_id)
            continue
        old_item['modified_prompts'] = repaired

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info("Saved %d tasks to %s", len(dataset), output_path)

    return dataset

if __name__ == "__main__":
    run()
