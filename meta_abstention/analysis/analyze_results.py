import json
import os

import meta_abstention.config as conf


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.1%})" if den > 0 else "N/A"


def _load_valid_tasks(completions_path: str) -> list[dict]:
    with open(completions_path) as f:
        tasks = json.load(f)

    valid_tasks = [t for t in tasks if len(t['completions']) == 5]
    print(f"Tasks total: {len(tasks)}  |  valid (5 completions): {len(valid_tasks)}  |  skipped: {len(tasks) - len(valid_tasks)}")
    return valid_tasks


def _generated_body(prompt: str, generated_code: str) -> str:
    prompt = prompt.strip()
    code = generated_code.strip()
    if code.startswith(prompt):
        return code[len(prompt):].strip()
    return code


def run_confidence_consistency(
    completions_path: str = os.path.join(conf.data['output-dir'], conf.completion['output-file']),
) -> None:
    valid_tasks = _load_valid_tasks(completions_path)

    # Per-task: are all 5 confidence values identical?
    task_uniform = {
        t['task_id']: len({c['confidence'] for c in t['completions']}) == 1
        for t in valid_tasks
    }

    # Bucket every completion into one of 4 subsets
    subsets = {
        ('high', True):  [],   # confidence == 100, uniform code across the task's completions
        ('high', False): [],   # confidence == 100, non-uniform code
        ('low',  True):  [],   # confidence < 100,  uniform code
        ('low',  False): [],   # confidence < 100,  non-uniform code
    }

    for task in valid_tasks:
        uniform = task_uniform[task['task_id']]
        for c in task['completions']:
            level = 'high' if c['confidence'] == 100 else 'low'
            subsets[(level, uniform)].append(c['passed'])


    high = subsets[('high', True)] + subsets[('high', False)]
    low  = subsets[('low',  True)] + subsets[('low',  False)]
    total = len(high) + len(low)

    print(f"\nTotal completions: {total}")
    print(f"  Confidence = 100 : {_ratio(len(high), total)}")
    print(f"  Confidence < 100 : {_ratio(len(low),  total)}")

    labels = {
        ('high', True):  "Confidence=100, Uniform code",
        ('high', False): "Confidence=100, Non-uniform code",
        ('low',  True):  "Confidence<100, Uniform code",
        ('low',  False): "Confidence<100, Non-uniform code",
    }

    print("\n--- Per-subset breakdown (all completions of the task share the same confidence) ---")
    for key, label in labels.items():
        items = subsets[key]
        n = len(items)
        n_passed = sum(items)
        print(f"\n  {label}:")
        print(f"    Total  : {n}")
        print(f"    Passed : {_ratio(n_passed, n)}")
        print(f"    Failed : {_ratio(n - n_passed, n)}")


def run_code_consistency(
    completions_path: str = os.path.join(conf.data['output-dir'], conf.completion['output-file']),
) -> None:
    valid_tasks = _load_valid_tasks(completions_path)

    # Per-task: after stripping the prompt from the front of each completion's generated
    # code, are the 5 remaining bodies all identical?
    task_uniform_code = {}
    for t in valid_tasks:
        bodies = {_generated_body(c['prompt'], c['generated_code']) for c in t['completions']}
        task_uniform_code[t['task_id']] = len(bodies) == 1

    # Bucket every completion into one of 4 subsets
    subsets = {
        ('high', True):  [],   # confidence == 100, uniform code across the task's completions
        ('high', False): [],   # confidence == 100, non-uniform code
        ('low',  True):  [],   # confidence < 100,  uniform code
        ('low',  False): [],   # confidence < 100,  non-uniform code
    }

    for task in valid_tasks:
        uniform = task_uniform_code[task['task_id']]
        for c in task['completions']:
            level = 'high' if c['confidence'] == 100 else 'low'
            subsets[(level, uniform)].append(c['passed'])

    high = subsets[('high', True)] + subsets[('high', False)]
    low  = subsets[('low',  True)] + subsets[('low',  False)]
    total = len(high) + len(low)

    print(f"\nTotal completions: {total}")
    print(f"  Confidence = 100 : {_ratio(len(high), total)}")
    print(f"  Confidence < 100 : {_ratio(len(low),  total)}")

    labels = {
        ('high', True):  "Confidence=100, Uniform code",
        ('high', False): "Confidence=100, Non-uniform code",
        ('low',  True):  "Confidence<100, Uniform code",
        ('low',  False): "Confidence<100, Non-uniform code",
    }

    print("\n--- Per-subset breakdown (correctness vs. code consistency across a task's 5 completions) ---")
    for key, label in labels.items():
        items = subsets[key]
        n = len(items)
        n_passed = sum(items)
        print(f"\n  {label}:")
        print(f"    Total  : {n}")
        print(f"    Passed : {_ratio(n_passed, n)}")
        print(f"    Failed : {_ratio(n - n_passed, n)}")


def run() -> None:
    print("=== Confidence-consistency analysis ===")
    run_confidence_consistency()

    print("\n\n=== Code-consistency analysis ===")
    run_code_consistency()
