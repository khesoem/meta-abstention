import json
import logging
import os
import re
from time import sleep

from meta_abstention.llm.llm_adapter import LLMAdapter
from meta_abstention.llm.invocation import Prompt
from meta_abstention import config as conf
from meta_abstention.xcodeeval.utils import execute_code

_GENERATE_TESTS_SYSTEM = "You are a helpful assistant that generates test inputs for {lang} programs. The code reads and writes I/O from the console."

_GENERATE_TESTS_USER_TEMPLATE = """\
Generate {num_tests} test inputs for the following source code. The code reads from standard input and writes to standard output.
The test inputs should cover various parts of the input space, including corner cases.
The source code should successfully run with each given test input and print an output in a reasonable amount of time (less than {max_runtime_seconds} seconds), with no runtime error.
Follow the input format described in the input specification and the sample inputs.

Print every test input in the following format:
<test_inputs>
<input>
<test input>
</input>
<input>
<test input>
</input>
</test_inputs>
Do not provide any other text or markdown fences.

Here is the problem description:
<description>
{description}
</description>

Here is the input specification:
<input_spec>
{input_spec}
</input_spec>
{notes_section}
Here are sample inputs:
<sample_inputs>
{sample_inputs}
</sample_inputs>

Here is the source code:
<source_code>
{source_code}
</source_code>
"""

_ACCEPTABLE_OUTCOMES = {
    "PASSED",
    "WRONG_ANSWER",
}


def _format_sample_inputs(sample_inputs: list) -> str:
    return "\n".join(f"<input>\n{sample}\n</input>" for sample in sample_inputs)


def _format_notes_section(notes: str | None) -> str:
    if not notes or not notes.strip():
        return "\n"
    return f"""
Here are additional notes:
<notes>
{notes}
</notes>
"""


def _parse_test_inputs(raw: str) -> list[str]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:\w+)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    inputs = []
    for match in re.findall(r"<input>(.*?)</input>", raw, flags=re.DOTALL | re.IGNORECASE):
        text = match.strip("\n")
        if text.strip():
            inputs.append(text)
    if not inputs:
        raise ValueError(f"No test inputs found in response: {raw[-30:]!r}")
    return inputs


def _ran_successfully(item: dict) -> bool:
    if item.get("exec_outcome") not in _ACCEPTABLE_OUTCOMES:
        return False
    result = item.get("result")
    if not isinstance(result, str) or not result.strip():
        return False
    time_consumed = item.get("time_consumed")
    if time_consumed is not None and float(time_consumed) >= conf.translation["max-runtime-seconds"]:
        return False
    return True


def _keep_runnable_inputs(lang: str, source_code: str, test_inputs: list[str]) -> list[str]:
    unittests = [
        {"input": test_input if test_input.endswith("\n") else test_input + "\n", "output": [""]}
        for test_input in test_inputs
    ]
    exec_result = execute_code(
        lang,
        source_code,
        unittests,
        timeout=max(30, len(unittests) * (conf.translation["max-runtime-seconds"] + 1)),
    )
    kept = []
    for test_input, item in zip(test_inputs, exec_result.get("data", [])):
        if _ran_successfully(item):
            kept.append(test_input)
        else:
            logging.info(
                "Dropping test input with outcome %s in %ss",
                item.get("exec_outcome"),
                item.get("time_consumed"),
            )
    return kept


def generate_test_inputs(source_code: str, problem_description: dict, model: str, lang: str, temp: float = 0) -> list[str]:
    num_tests = conf.translation["number-of-generated-tests"]
    messages = [
        Prompt.Message("system", _GENERATE_TESTS_SYSTEM.format(lang=lang)),
        Prompt.Message(
            "user",
            _GENERATE_TESTS_USER_TEMPLATE.format(
                num_tests=num_tests,
                max_runtime_seconds=conf.translation["max-runtime-seconds"],
                description=problem_description.get("description") or "",
                input_spec=problem_description.get("input_spec") or "",
                notes_section=_format_notes_section(problem_description.get("notes")),
                sample_inputs=_format_sample_inputs(problem_description["sample_inputs"]),
                source_code=source_code,
            ),
        ),
    ]
    adapter = LLMAdapter(read_from_cache=True, save_to_cache=True, model=model)
    raw = adapter.get_response(Prompt(messages, temp=temp)).first_content
    test_inputs = _parse_test_inputs(raw)[:num_tests]
    return _keep_runnable_inputs(lang, source_code, test_inputs)


def run_test_generation(original_submission_path: str, output_path: str, model: str):
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            generated_test_inputs = json.load(f)

    else:
        generated_test_inputs = {}

    with open(original_submission_path, "r") as f:
        original_submission = json.load(f)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for _, item in original_submission.items():
        problem_description = item["problem_description"]
        for submission in item["submissions"]:
            if submission["code_uid"] in generated_test_inputs:
                test_inputs = generated_test_inputs[submission["code_uid"]]
            else:
                test_inputs = []
            tries = 0
            while len(test_inputs) < conf.translation["number-of-considered-generated-tests"] and tries < conf.translation["max-tries"]:
                tries += 1
                try:
                    new_test_inputs = generate_test_inputs(
                        submission["source_code"],
                        problem_description,
                        model,
                        submission["lang"],
                        temp=0 if not test_inputs else 1.0,
                    )
                    test_inputs.extend(new_test_inputs)
                except Exception as e:
                    logging.error("Error generating test inputs for %s: %s", submission["code_uid"], e)
                    sleep(1)
            
            generated_test_inputs[submission["code_uid"]] = test_inputs
            logging.info(
                "Generated %s test inputs for %s",
                len(test_inputs),
                submission["code_uid"],
            )

        with open(output_path, "w") as f:
            json.dump(generated_test_inputs, f, indent=4)
