import json
import math
import logging
import re

from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt, Response
from meta_abstention import config as conf

_TRANSLATE_SYSTEM = "You are a helpful assistant that translates {source_lang_cluster} code to {target_lang} code. The code should reads and writes I/O from the console."

_TRANSLATE_USER_TEMPLATE = """\
Translate the following {source_lang} code to {target_lang} code. The code reads and writes I/O from the console. Your translated code should have the exact same functionality as the given code, meaning that it should have the same output for any given input.
Print your translated code and also the confidence score you give to your translation. The confidence score should be a number between 0 and 1. The format of the output should be:
<translated_code>
<your_translated_code>
</translated_code>
<confidence>
<your_confidence_score>
</confidence>
Do not provide any other text or markdown fences.

Here is the original {source_lang} code you have to translate:
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


def _parse_translation(raw: str) -> tuple[str, dict]:
    raw = raw.strip()

    # 1) Ideal: both tags
    code_m = re.search(r"<translated_code>(.*?)</translated_code>", raw, re.DOTALL | re.IGNORECASE)
    conf_m = re.search(r"<confidence>(.*?)</confidence>", raw, re.DOTALL | re.IGNORECASE)

    if code_m:
        code = code_m.group(1)
    elif re.search(r"</translated_code>", raw, re.IGNORECASE):
        # Mode A: missing opening tag
        code = re.split(r"</translated_code>", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    elif conf_m:
        # Mode B: code then <confidence>
        code = raw[: conf_m.start()]
    else:
        # Mode C: bare code
        code = raw

    if not conf_m:
        # optional: prose fallback, e.g. "Confidence: 0.9"
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
        raise ValueError("Empty translated code after parsing")

    return code, {"verbalization": confidence}


def _code_span_in_raw(raw: str) -> tuple[int, int] | None:
    """Character span of the translated-code body inside the raw response."""
    code_m = re.search(r"<translated_code>(.*?)</translated_code>", raw, re.DOTALL | re.IGNORECASE)
    if code_m:
        return code_m.start(1), code_m.end(1)
    conf_m = re.search(r"<confidence>.*?</confidence>", raw, re.DOTALL | re.IGNORECASE)
    if conf_m:
        return 0, conf_m.start()
    return None


def _token_probabilities_for_code(sample: Response.Sample) -> list[float]:
    """Per-token probabilities for the generated code (Spiess et al.).

    Uses tokens overlapping the translated-code span when available; otherwise
    falls back to all generated tokens. Probabilities are p(τ_i) = exp(log p).
    """
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
    """Average token probability and generated sequence probability.

    From Spiess et al., "Calibration and Correctness of Language Models for Code":
      p_avg(T) = (1/n) * sum_i p(τ_i)
      p_tot(T) = product_i p(τ_i)
    """
    probs = _token_probabilities_for_code(sample)
    if not probs:
        raise ValueError("No token probabilities available for intrinsic confidence")
    return {
        "average_token_probability": sum(probs) / len(probs),
        "average_token_probability_geometric": math.prod(probs) ** (1/len(probs)),
        "generated_sequence_probability": math.prod(probs),
    }


def _translate_code(adapter: LLMAdapter, source_code: str, source_lang: str, source_lang_cluster: str, target_lang: str) -> tuple[str, dict]:
    messages = [
        Prompt.Message("system", _TRANSLATE_SYSTEM.format(source_lang_cluster=source_lang_cluster, target_lang=target_lang)),
        Prompt.Message("user", _TRANSLATE_USER_TEMPLATE.format(source_lang=source_lang, target_lang=target_lang, source_code=source_code)),
    ]
    sample = adapter.get_response(Prompt(messages, logprobs=True)).first_sample
    code, confidence = _parse_translation(sample.content)
    confidence.update(_intrinsic_confidence(sample))
    return code, confidence

def run_translation(selected_problems_path: str, output_file: str, target_lang: str):
    with open(selected_problems_path, 'r') as f:
        data = json.load(f)

    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=conf.translation['default-model'])
    for _, item in data.items():
        for submission in item['submissions']:
            try:            
                if 'translation' in submission and len(submission['translation']) > 0:
                    continue

                translated_code, confidence = _translate_code(adapter, submission['source_code'], submission['lang'], submission['lang_cluster'], target_lang)
                if not 'translation' in submission:
                    submission['translation'] = []
                
                submission['translation'].append({
                    'lang': target_lang,
                    'translated_code': translated_code,
                    'confidence': confidence
                })
                logging.info(f"Translated code for {submission['code_uid']} to {target_lang} with confidence {confidence}")
            except Exception as e:
                logging.error(f"Error translating code for {submission['code_uid']} to {target_lang}: {e}")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)