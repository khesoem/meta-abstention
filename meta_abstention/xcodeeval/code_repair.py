import json
import math
import logging
import os
import re

from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt, Response
from meta_abstention import config as conf

_REPAIR_SYSTEM = "You are a helpful assistant that repairs {lang} code so that it correctly solves the given programming problem. The code reads and writes I/O from the console."

_REPAIR_USER_TEMPLATE = """\
Repair the following {lang} code so that it correctly solves the programming problem. The code reads and writes I/O from the console. Your repaired code should implement the problem correctly, meaning that it should produce the correct output for any valid input.
Print your repaired code and also the confidence score you give to your repair. The confidence score should be a number between 0 and 1. The format of the output should be:
<repaired_code>
<your_repaired_code>
</repaired_code>
<confidence>
<your_confidence_score>
</confidence>
Do not provide any other text or markdown fences.

Here is the problem description:
<problem_description>
{description}
</problem_description>

Here is the {lang} code you have to repair:
<original_code>
{source_code}
</original_code>
"""

def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_confidence(text: str) -> float:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not m:
            raise ValueError(f"Could not parse confidence from: {text!r}")
        return float(m.group(0))


def _parse_repair(raw: str) -> tuple[str, dict]:
    raw = raw.strip()

    code_m = re.search(r"<repaired_code>(.*?)</repaired_code>", raw, re.DOTALL | re.IGNORECASE)
    conf_m = re.search(r"<confidence>(.*?)</confidence>", raw, re.DOTALL | re.IGNORECASE)

    if code_m:
        code = code_m.group(1)
    elif re.search(r"</repaired_code>", raw, re.IGNORECASE):
        code = re.split(r"</repaired_code>", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    elif conf_m:
        code = raw[: conf_m.start()]
    else:
        code = raw

    if not conf_m:
        prose = re.search(
            r"(?i)confidence(?:\s*score)?\s*[:=]?\s*([-+]?\d*\.?\d+)",
            raw,
        )
        if not prose:
            raise ValueError(f"Missing confidence in response: {raw[-200:]!r}")
        confidence = float(prose.group(1))
    else:
        confidence = _parse_confidence(conf_m.group(1))

    code = _strip_fences(code)
    if not code:
        raise ValueError("Empty repaired code after parsing")

    return code, {"verbalization": confidence}


def _code_span_in_raw(raw: str) -> tuple[int, int] | None:
    """Character span of the repaired-code body inside the raw response."""
    code_m = re.search(r"<repaired_code>(.*?)</repaired_code>", raw, re.DOTALL | re.IGNORECASE)
    if code_m:
        return code_m.start(1), code_m.end(1)
    conf_m = re.search(r"<confidence>.*?</confidence>", raw, re.DOTALL | re.IGNORECASE)
    if conf_m:
        return 0, conf_m.start()
    return None


def _token_probabilities_for_code(sample: Response.Sample) -> list[float]:
    if not sample.token_logprobs:
        raise ValueError("Response has no token logprobs; cannot compute intrinsic confidence")

    all_probs = [math.exp(lp) for lp in sample.token_logprobs]
    if not sample.tokens or len(sample.tokens) != len(sample.token_logprobs):
        return all_probs

    span = _code_span_in_raw(sample.content)
    if span is None:
        return all_probs

    start, end = span
    probs = []
    pos = 0
    for token, p in zip(sample.tokens, all_probs):
        tok_end = pos + len(token)
        if tok_end > start and pos < end:
            probs.append(p)
        pos = tok_end
    return probs if probs else all_probs


def _intrinsic_confidence(sample: Response.Sample) -> dict[str, float]:
    probs = _token_probabilities_for_code(sample)
    if not probs:
        raise ValueError("No token probabilities available for intrinsic confidence")
    return {
        "average_token_probability": sum(probs) / len(probs),
        "average_token_probability_geometric": math.prod(probs) ** (1/len(probs)),
        "generated_sequence_probability": math.prod(probs),
    }


def _repair_code(adapter: LLMAdapter, description: str, source_code: str, lang: str, temp: float, get_logprobs: bool) -> tuple[str, dict]:
    messages = [
        Prompt.Message("system", _REPAIR_SYSTEM.format(lang=lang)),
        Prompt.Message("user", _REPAIR_USER_TEMPLATE.format(lang=lang, description=description, source_code=source_code)),
    ]
    sample = adapter.get_response(Prompt(messages, logprobs=get_logprobs, temp=temp)).first_sample
    code, confidence = _parse_repair(sample.content)
    if get_logprobs:
        confidence.update(_intrinsic_confidence(sample))
    return code, confidence

def run_repair(selected_problems_path: str, output_file: str, model: str = conf.translation['default-model'], temp: float = conf.translation['default-temp'], get_logprobs: bool = True) -> bool:
    logging.info(f"Running repair for {selected_problems_path} with model {model} and temp {temp}")

    with open(selected_problems_path, 'r') as f:
        data = json.load(f)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=model)
    new_repairs = False
    for _, item in data.items():
        description = item['problem_description']['description']
        for submission in item['submissions']:
            try:
                if 'repair' in submission and submission['repair']:
                    continue

                new_repairs = True

                repaired_code, confidence = _repair_code(
                    adapter,
                    description,
                    submission['source_code'],
                    submission['lang'],
                    temp=temp,
                    get_logprobs=get_logprobs,
                )
                submission['repair'] = {
                    'lang': submission['lang'],
                    'repaired_code': repaired_code,
                    'confidence': confidence,
                }
                logging.info(f"Repaired code for {submission['code_uid']} with confidence {confidence}")
            except Exception as e:
                logging.error(f"Error repairing code for {submission['code_uid']}: {e}")

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    return new_repairs
