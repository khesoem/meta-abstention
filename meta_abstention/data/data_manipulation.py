import json
import logging
import os
import re

from datasets import load_dataset

import meta_abstention.config as conf
from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt

logger = logging.getLogger(__name__)

_PARAPHRASE_SYSTEM = (
    "You are a technical writing assistant. You rephrase Python function docstrings "
    "while preserving their exact semantics.\n"
    "Rules:\n"
    "- Keep the function signature (name, parameters, return type annotation) exactly unchanged.\n"
    "- Keep any import statements exactly unchanged.\n"
    "- Keep doctest examples (lines starting with >>> or ...) exactly unchanged.\n"
    "- Rephrase only the natural language description inside the docstring.\n"
    "- Make each variant meaningfully different from the others.\n"
    "- Return a JSON array of {n} strings, each containing one complete paraphrased function. "
    "No markdown fences, no other text."
)


def _parse_variants(raw: str, n: int) -> list[str]:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    variants = json.loads(raw)
    if len(variants) != n:
        logger.warning("Expected %d variants, got %d", n, len(variants))
    return variants


def _paraphrase_all(adapter: LLMAdapter, original_prompt: str, n_variants: int) -> list[str]:
    messages = [
        Prompt.Message("system", _PARAPHRASE_SYSTEM.format(n=n_variants)),
        Prompt.Message("user", original_prompt),
    ]
    prompt = Prompt(messages)
    raw = adapter.get_response(prompt).first_content
    return _parse_variants(raw, n_variants)


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


if __name__ == "__main__":
    run()
