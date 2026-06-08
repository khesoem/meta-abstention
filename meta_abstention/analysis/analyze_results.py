import json
import os

import meta_abstention.config as conf


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.1%})" if den > 0 else "N/A"


def run(
    completions_path: str = os.path.join(conf.data['output-dir'], conf.completion['output-file']),
) -> None:
    with open(completions_path) as f:
        tasks = json.load(f)

    valid_tasks = [t for t in tasks if len(t['completions']) == 5]
    print(f"Tasks total: {len(tasks)}  |  valid (5 completions): {len(valid_tasks)}  |  skipped: {len(tasks) - len(valid_tasks)}")

    # Per-task: are all 5 confidence values identical?
    task_uniform = {
        t['task_id']: len({c['confidence'] for c in t['completions']}) == 1
        for t in valid_tasks
    }

    # Bucket every completion into one of 4 subsets
    subsets = {
        ('high', True):  [],   # confidence == 100, passed
        ('high', False): [],   # confidence == 100, failed
        ('low',  True):  [],   # confidence < 100,  passed
        ('low',  False): [],   # confidence < 100,  failed
    }

    for task in valid_tasks:
        uniform = task_uniform[task['task_id']]
        for c in task['completions']:
            level = 'high' if c['confidence'] == 100 else 'low'
            subsets[(level, c['passed'])].append(uniform)

    high = subsets[('high', True)] + subsets[('high', False)]
    low  = subsets[('low',  True)] + subsets[('low',  False)]
    total = len(high) + len(low)

    print(f"\nTotal completions: {total}")
    print(f"  Confidence = 100 : {_ratio(len(high), total)}")
    print(f"  Confidence < 100 : {_ratio(len(low),  total)}")

    print(f"\nPass rate within Confidence = 100 : {_ratio(len(subsets[('high', True)]), len(high))}")
    print(f"Pass rate within Confidence < 100 : {_ratio(len(subsets[('low',  True)]), len(low))}")

    labels = {
        ('high', True):  "Confidence=100, Passed",
        ('high', False): "Confidence=100, Failed",
        ('low',  True):  "Confidence<100, Passed",
        ('low',  False): "Confidence<100, Failed",
    }

    print("\n--- Per-subset breakdown (uniform = all 5 completions of the task share the same confidence) ---")
    for key, label in labels.items():
        items = subsets[key]
        n         = len(items)
        n_uniform = sum(items)
        print(f"\n  {label}:")
        print(f"    Total                       : {n}")
        print(f"    Uniform task confidence     : {_ratio(n_uniform, n)}")
        print(f"    Non-uniform task confidence : {_ratio(n - n_uniform, n)}")


if __name__ == "__main__":
    run()
