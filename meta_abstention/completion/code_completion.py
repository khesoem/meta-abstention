import json
import logging
import os
import re

import meta_abstention.config as conf
from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt

logger = logging.getLogger(__name__)

_SYSTEM = "You are an expert Python programmer."

_USER_TEMPLATE = """{prompt}

Complete the function body above. Return a JSON object with no markdown fences:
{{
  "code": "<function body only — lines after the closing docstring quote, correctly indented>",
  "confidence": <float from 0.0 to 1.0>
}}"""


def _parse_completion(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    parsed = json.loads(raw)
    return {
        'generated_code': parsed['code'],
        'confidence': float(parsed['confidence']),
    }


def _complete(adapter: LLMAdapter, prompt_text: str) -> dict:
    messages = [
        Prompt.Message("system", _SYSTEM),
        Prompt.Message("user", _USER_TEMPLATE.format(prompt=prompt_text)),
    ]
    raw = adapter.get_response(Prompt(messages)).first_content
    return _parse_completion(raw)


def run(
    model: str = conf.completion['model'],
    input_path: str = os.path.join(conf.data['output-dir'], conf.data['humanevalplus-output-file']),
    output_dir: str = conf.data['output-dir'],
    output_file: str = conf.completion['output-file'],
) -> list[dict]:
    with open(input_path) as f:
        tasks = json.load(f)

    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=model)

    results = []
    for task in tasks:
        task_id = task['task_id']
        prompts = [('original', task['original_prompt'])] + [
            (f'variant_{i}', p) for i, p in enumerate(task['modified_prompts'])
        ]

        completions = []
        for prompt_type, prompt_text in prompts:
            logger.info("Completing %s / %s", task_id, prompt_type)
            try:
                result = _complete(adapter, prompt_text)
                completions.append({'prompt_type': prompt_type, 'prompt': prompt_text, **result})
            except Exception as e:
                logger.error("Error completing %s / %s: %s", task_id, prompt_type, e)

        results.append({'task_id': task_id, 'completions': completions})

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, output_file)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d tasks to %s", len(results), out)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
