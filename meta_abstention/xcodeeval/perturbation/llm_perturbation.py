import json
import copy
import logging
import time
import re
import os
from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt

_PERTURB_SYSTEM = "You are a helpful assistant that rewrites {lang} code into functionally equivalent variants. The code reads and writes I/O from the console."


def _format_example(i: int) -> str:
    return (
        f"<perturbed_code_{i}>\n"
        f"<your_rewritten_code_{i}>\n"
        f"</perturbed_code_{i}>"
    )


def _perturb_user_message(lang: str, perturbation_cnt: int, source_code: str) -> str:
    blocks = "\n".join(_format_example(i) for i in range(1, perturbation_cnt + 1))
    return f"""\
Rewrite the following {lang} code into exactly {perturbation_cnt} different variants. The code reads and writes I/O from the console. Each variant should have the exact same functionality as the given code, meaning that it should have the same output for any given input. Make the variants meaningfully different from one another and from the original code, while preserving functionality exactly.
Print all {perturbation_cnt} rewritten variants, numbered from 1 to {perturbation_cnt}. Do not stop early, do not skip an index, and do not repeat an index. Use this exact format:
{blocks}
Each block must contain one complete program. Do not provide any other text or markdown fences.

Here is the original {lang} code you have to rewrite:
<original_code>
{source_code}
</original_code>
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_perturbed_codes(raw: str, perturbation_cnt: int) -> list[str]:
    raw = _strip_fences(raw)
    found: dict[int, str] = {}
    for match in re.finditer(
        r"<perturbed_code_(\d+)>(.*?)</perturbed_code_\1>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        index = int(match.group(1))
        code = _strip_fences(match.group(2))
        if 1 <= index <= perturbation_cnt and code:
            found[index] = code
    missing = [i for i in range(1, perturbation_cnt + 1) if i not in found]
    if missing:
        raise ValueError(
            f"Expected perturbed programs 1..{perturbation_cnt}, missing {missing}: {raw[-200:]!r}"
        )
    return [found[i] for i in range(1, perturbation_cnt + 1)]


def perturb_source_code(source_code: str, lang: str, model: str, perturbation_cnt: int, temperature: float=0.0) -> list[str]:
    messages = [
        Prompt.Message("system", _PERTURB_SYSTEM.format(lang=lang)),
        Prompt.Message("user", _perturb_user_message(lang, perturbation_cnt, source_code)),
    ]
    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=model)
    raw = adapter.get_response(Prompt(messages, temp=temperature, logprobs=False)).first_content
    return _parse_perturbed_codes(raw, perturbation_cnt)


def run_llm_perturbation(original_submissions_path: str, output_file: str, model: str, perturbation_cnt: int) -> None:
    with open(original_submissions_path, 'r') as f:
        original_submissions = json.load(f)
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            perturbed_submissions = json.load(f)
    else:
        perturbed_submissions = {}
    
    for src_uid, item in original_submissions.items():
        problem_description = item['problem_description']
        unittests = item['unittests']
        for submission in item['submissions']:
            code_uid = submission['code_uid']
            source_code = submission['source_code']
            lang = submission['lang']

            new_src_uid = src_uid + '_' + code_uid

            if new_src_uid in perturbed_submissions:
                continue

            perturbed_submissions[new_src_uid] = {
                'problem_description': problem_description,
                'unittests': unittests,
                'submissions': []
            }

            # deep copy the submission
            submission_copy = copy.deepcopy(submission)
            submission_copy['code_uid'] = code_uid + '_0'
            perturbed_submissions[new_src_uid]['submissions'].append(submission_copy)

            tries = 0
            while tries < 3:
                try:
                    perturbed_source_codes = perturb_source_code(source_code, lang, model, perturbation_cnt, tries * 0.5)
                except Exception as e:
                    tries += 1
                    logging.error(f"Error perturbing source code for {new_src_uid} with {model}: {e}")
                    time.sleep(1)
                    continue

                logging.info(f"Perturbed source code for {new_src_uid} with {model} after {tries} tries")
                break
            if tries == 3:
                logging.error(f"Failed to perturb source code for {new_src_uid} with {model} after 3 tries")
                continue

            for i, perturbed_source_code in enumerate(perturbed_source_codes):
                submission_copy = copy.deepcopy(submission)
                submission_copy['code_uid'] = code_uid + '_' + str(i + 1)
                submission_copy['source_code'] = perturbed_source_code
                perturbed_submissions[new_src_uid]['submissions'].append(submission_copy)

        with open(output_file, 'w') as f:
            json.dump(perturbed_submissions, f, indent=4)