"""Aggregate code-translation confidence reports into LaTeX tables and charts.

Reads the text reports written by ``confidence_analysis.py`` (one file per
source-target pair, plus ``overall_results.txt``) and writes:

* a table of ECE, Brier score, and Brier skill score
* a table of AUROC
* one chart per language pair (and one for the pooled results) comparing
  every method on those four metrics

Reports sit either directly in the results directory, or in one subdirectory
per perturbation count (``1-perturbations``, ``2-perturbations``, ...). In
the latter layout every count is included. Baseline methods from
``confidence_analysis._BASELINE_METRICS`` — verbalized confidence and token
and sequence probabilities, raw and Platt-scaled, on the original translations
and on translations of perturbed source — do not depend on that count, so
they are emitted once. Every other method is emitted once per count, with
`` (n=k)`` inserted before any Platt or perturbed suffix.

Each metric line is a point estimate. A trailing 95% interval, when still
present on an older line, is ignored. Spearman and Pearson correlations are
ignored. The best value in each table column is bold: higher is better for
AUROC and skill score, lower is better for ECE and Brier.

Generated-test output consistency is reported only for some models (currently
qwen3-coder-next) and is included only when that block is in the reports.

Perturbation reports (``*-perturbations-to-*.txt`` and
``overall_perturbations_results.txt``) are optional and often cover only a
subset of methods. When present, those methods are appended with a
``(Perturbed)`` suffix so they appear alongside the normal metrics. Models
without perturbation reports are left unchanged; missing cells are ``---``.
"""

import argparse
import colorsys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from meta_abstention.xcodeeval.confidence_analysis import PLATT_SUFFIX, _BASELINE_METRICS

HEADER_RE = re.compile(
    r"^===\s+(.+?)\s+\(n=(\d+),\s*accuracy=([\d.]+)%\)\s+===\s*$"
)
METRIC_RE = re.compile(
    r"^\s+(?P<name>AUROC|Brier|Skill Score|ECE \(10 bins\)|Spearman rho|Pearson r)\s*:\s*"
    r"(?P<value>-?\d+\.\d+|nan)"
    r"(?:\s+95% CI\s*\[\s*-?\d+\.\d+,\s*-?\d+\.\d+\])?\s*$"
)

SKIPPED_METRICS = {"Spearman rho", "Pearson r"}
CALIBRATION_METRICS = ("ECE (10 bins)", "Brier", "Skill Score")
HIGHER_IS_BETTER = {"AUROC", "Skill Score"}
PERTURBED_SUFFIX = " (Perturbed)"
# Inserted before the Platt and perturbed suffixes: "Method (n=3) (Platt)".
_COUNT_SUFFIX_RE = re.compile(r" \(n=(\d+)\)$")
_COUNT_DIR_RE = re.compile(r"\d+-perturbations")
BASELINE_METHOD_NAMES = {name for _, name in _BASELINE_METRICS}

LANG_NAMES = {"cpp": "C++", "java": "Java", "python": "Python"}
LANG_HEADER = {"cpp": "C++", "java": "J", "python": "P"}

# Short labels used only in the LaTeX tables. Charts keep the full names.
METHOD_ABBREV = {
    "Simple Verbalized": "Verb",
    "Average Token Probability": "Tok",
    "Average Token Probability Geometric": "Tok-G",
    "Generated Sequence Probability": "Seq",
    "SPUQ CodeBERT Score": "SPUQ-CB",
    "SPUQ CodeBERT Score Reverse": "SPUQ-CB-R",
    "SPUQ CodeBLEU": "SPUQ-BLEU",
    "SPUQ CodeBLEU Reverse": "SPUQ-BLEU-R",
    "SPUQ Unixcoder": "SPUQ-UX",
    "SPUQ Unixcoder Reverse": "SPUQ-UX-R",
    "Average Verbalized": "Avg-Verb",
    "Average Verbalized CodeBERT Score": "Avg-Verb-CB",
    "Average Verbalized CodeBLEU": "Avg-Verb-BLEU",
    "Average Verbalized Unixcoder": "Avg-Verb-UX",
    "Average of Average Token Probability": "Avg-Tok",
    "Average of Average Token Probability CodeBERT Score": "Avg-Tok-CB",
    "Average of Average Token Probability CodeBLEU": "Avg-Tok-BLEU",
    "Average of Average Token Probability Unixcoder": "Avg-Tok-UX",
    "Average of Average Token Probability Geometric": "Avg-Tok-G",
    "Average of Average Token Probability Geometric CodeBERT Score": "Avg-Tok-G-CB",
    "Average of Average Token Probability Geometric CodeBLEU": "Avg-Tok-G-BLEU",
    "Average of Average Token Probability Geometric Unixcoder": "Avg-Tok-G-UX",
    "Average Generated Sequence Probability": "Avg-Seq",
    "Average Generated Sequence Probability CodeBERT Score": "Avg-Seq-CB",
    "Average Generated Sequence Probability CodeBLEU": "Avg-Seq-BLEU",
    "Average Generated Sequence Probability Unixcoder": "Avg-Seq-UX",
    "Generated Test Output Consistency Score": "GTOC",
    "Original Test -4 Output Consistency Score": r"OT$-4$",
    "Original Test -3 Output Consistency Score": r"OT$-3$",
    "Original Test -2 Output Consistency Score": r"OT$-2$",
    "Original Test -1 Output Consistency Score": r"OT$-1$",
    "Original Test 0 Output Consistency Score": "OT0",
    "Original Test 3 Output Consistency Score": "OT3",
    "Original Test 6 Output Consistency Score": "OT6",
    "Original Test 9 Output Consistency Score": "OT9",
    "Original Test 12 Output Consistency Score": "OT12",
}


def abbrev_note(methods: list[str]) -> str:
    note = (
        r"Abbreviations: Verb = simple verbalized; Tok / Tok-G = arithmetic / geometric "
        r"mean token probability; Seq = sequence probability; Avg- = mean over perturbations; "
        r"CB / BLEU / UX = CodeBERT, CodeBLEU, UniXcoder weighting; R = reverse; "
        r"OT$k$ = original-test output consistency on $k$ tests "
        r"($k<0$ uses $1/|k|$ of the tests). "
    )
    if "Generated Test Output Consistency Score" in methods or any(
        base_method_name(m) == "Generated Test Output Consistency Score" for m in methods
    ):
        note += r"GTOC = generated-test output consistency. "
    if any(perturbation_count(m) is not None for m in methods):
        note += (
            r"Suffix n$k$ is the number of source perturbations used for that "
            r"consistency score. "
        )
    if any(is_perturbed_method(m) for m in methods):
        note += (
            r"Suffix -P = same method evaluated on translations of LLM-perturbed "
            r"source code (partial; missing cells are ---). "
        )
    if any(is_platt_method(m) for m in methods):
        note += (
            r"Suffix -Pl = five-fold Platt scaling of that confidence "
            r"(Spiess et al.). "
        )
    note += r"J = Java, P = Python, All = all pairs pooled."
    return note


GROUP_ORDER = (
    "verbalized",
    "token",
    "sequence",
    "spuq",
    "avg_verbalized",
    "avg_token",
    "avg_sequence",
    "gen_test",
    "orig_test",
    "pert_gen_test",
    "pert_orig_test",
    "pert_other",
    "other",
)
GROUP_LABELS = {
    "verbalized": "Simple verbalized",
    "token": "Token probability",
    "sequence": "Sequence probability",
    "spuq": "SPUQ",
    "avg_verbalized": "Average verbalized",
    "avg_token": "Average token probability",
    "avg_sequence": "Average sequence probability",
    "gen_test": "Generated-test consistency",
    "orig_test": "Original-test consistency",
    "pert_gen_test": "Generated-test consistency (perturbed)",
    "pert_orig_test": "Original-test consistency (perturbed)",
    "pert_other": "Other (perturbed)",
    "other": "Other",
}
GROUP_COLORS = {
    "verbalized": "#0072B2",
    "token": "#E69F00",
    "sequence": "#009E73",
    "spuq": "#CC79A7",
    "avg_verbalized": "#56B4E9",
    "avg_token": "#D55E00",
    "avg_sequence": "#882255",
    "gen_test": "#332288",
    "orig_test": "#44AA99",
    "pert_gen_test": "#6A3D9A",
    "pert_orig_test": "#B15928",
    "pert_other": "#A6A6A6",
    "other": "#7F7F7F",
}


def is_perturbed_method(name: str) -> bool:
    return name.endswith(PERTURBED_SUFFIX)


def _without_perturbed(name: str) -> str:
    if is_perturbed_method(name):
        return name[: -len(PERTURBED_SUFFIX)]
    return name


def is_platt_method(name: str) -> bool:
    return _without_perturbed(name).endswith(PLATT_SUFFIX)


def perturbation_count(name: str) -> int | None:
    """Perturbation count encoded in a merged method name, if any."""
    name = _without_perturbed(name)
    if name.endswith(PLATT_SUFFIX):
        name = name[: -len(PLATT_SUFFIX)]
    match = _COUNT_SUFFIX_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def base_method_name(name: str) -> str:
    name = _without_perturbed(name)
    if name.endswith(PLATT_SUFFIX):
        name = name[: -len(PLATT_SUFFIX)]
    return _COUNT_SUFFIX_RE.sub("", name)


def is_baseline_method(name: str) -> bool:
    """True for metrics that do not depend on the perturbation count.

    That is the raw and Platt-scaled baselines from ``confidence_analysis``,
    including the same scores on translations of perturbed source code.
    """
    return base_method_name(name) in BASELINE_METHOD_NAMES


def perturbed_method_name(name: str) -> str:
    if is_perturbed_method(name):
        return name
    return name + PERTURBED_SUFFIX


def with_perturbation_count(name: str, count: int) -> str:
    """Label a count-dependent method with the perturbation count used."""
    if perturbation_count(name) is not None:
        raise ValueError(f"{name!r} already records a perturbation count")
    renamed = f"{base_method_name(name)} (n={count})"
    if is_platt_method(name):
        renamed += PLATT_SUFFIX
    if is_perturbed_method(name):
        renamed += PERTURBED_SUFFIX
    return renamed


def _without_count(name: str) -> str:
    """Drop an `` (n=k)`` label, keeping Platt and perturbed suffixes."""
    restored = base_method_name(name)
    if is_platt_method(name):
        restored += PLATT_SUFFIX
    if is_perturbed_method(name):
        restored += PERTURBED_SUFFIX
    return restored


def method_group(name: str) -> str:
    perturbed = is_perturbed_method(name)
    base = base_method_name(name)
    if base.startswith("SPUQ "):
        group = "spuq"
    elif base.startswith("Average Verbalized"):
        group = "avg_verbalized"
    elif base.startswith("Average of Average Token Probability"):
        group = "avg_token"
    elif base.startswith("Average Generated Sequence Probability"):
        group = "avg_sequence"
    elif base.startswith("Average Token Probability"):
        group = "token"
    elif base == "Generated Sequence Probability":
        group = "sequence"
    elif base == "Simple Verbalized":
        group = "verbalized"
    elif base == "Generated Test Output Consistency Score":
        group = "gen_test"
    elif base.startswith("Original Test "):
        group = "orig_test"
    else:
        group = "other"
    if not perturbed:
        return group
    if group == "gen_test":
        return "pert_gen_test"
    if group == "orig_test":
        return "pert_orig_test"
    return "pert_other"


def pair_sort_key(pair: str) -> tuple:
    if pair == "overall":
        return (1, "")
    return (0, pair)


def latex_pair_label(pair: str) -> str:
    if pair == "overall":
        return "Overall"
    source, target = pair.split("-to-")
    return f"{LANG_NAMES[source]}$\\rightarrow${LANG_NAMES[target]}"


def latex_pair_header(pair: str) -> str:
    if pair == "overall":
        return "All"
    source, target = pair.split("-to-")
    return f"{LANG_HEADER[source]}$\\to${LANG_HEADER[target]}"


def method_abbrev(name: str) -> str:
    base = base_method_name(name)
    try:
        abbrev = METHOD_ABBREV[base]
    except KeyError as exc:
        raise KeyError(f"No table abbreviation for method {name!r}") from exc
    count = perturbation_count(name)
    if count is not None:
        abbrev += f"-n{count}"
    if is_platt_method(name):
        abbrev += "-Pl"
    if is_perturbed_method(name):
        abbrev += "-P"
    return abbrev


def display_pair_label(pair: str) -> str:
    if pair == "overall":
        return "Overall"
    source, target = pair.split("-to-")
    return f"{LANG_NAMES[source]} → {LANG_NAMES[target]}"


def classify_report(stem: str) -> tuple[str, str] | None:
    """Return (kind, pair_key) for a report stem, or None if unrecognized."""
    if stem == "overall_results":
        return "normal", "overall"
    if stem == "overall_perturbations_results":
        return "perturbation", "overall"
    if "-perturbations-to-" in stem:
        return "perturbation", stem.replace("-perturbations-to-", "-to-")
    if re.fullmatch(r"(cpp|java|python)-to-(cpp|java|python)", stem):
        return "normal", stem
    return None


def parse_report(path: Path) -> tuple[dict, list[str], int, float]:
    methods: dict[str, dict[str, float]] = {}
    order: list[str] = []
    current = None
    sample_size = None
    accuracy = None

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        header = HEADER_RE.match(line)
        if header:
            current = header.group(1)
            if current in methods:
                raise ValueError(f"{path}:{line_number}: duplicate method {current!r}")
            methods[current] = {}
            order.append(current)
            sample_size = int(header.group(2))
            accuracy = float(header.group(3))
            continue

        metric = METRIC_RE.match(line)
        if not metric:
            if line.strip():
                raise ValueError(f"{path}:{line_number}: unrecognized line: {line!r}")
            continue
        if current is None:
            raise ValueError(f"{path}:{line_number}: metric before a method header")
        name = metric.group("name")
        if name in SKIPPED_METRICS:
            continue
        methods[current][name] = float(metric.group("value"))

    if not order or sample_size is None or accuracy is None:
        raise ValueError(f"{path}: no method blocks found")

    expected = set(CALIBRATION_METRICS) | {"AUROC"}
    for method, metrics in methods.items():
        missing = expected - metrics.keys()
        if missing:
            raise ValueError(f"{path}: {method!r} is missing {sorted(missing)}")

    return methods, order, sample_size, accuracy


def _check_method_order(reports: dict, label: str) -> list[str]:
    if not reports:
        return []
    first_key = next(iter(reports))
    method_order = reports[first_key]["order"]
    for key, report in reports.items():
        if report["order"] != method_order:
            raise ValueError(
                f"Method list in {label} report {key!r} does not match the other {label} reports"
            )
    return method_order


def _metric_value(data: dict, pair: str, method: str, metric: str) -> float | None:
    methods = data.get(pair)
    if methods is None:
        return None
    method_metrics = methods.get(method)
    if method_metrics is None:
        return None
    return method_metrics.get(metric)


def load_results(results_dir: Path) -> tuple[dict, list[str], dict, dict]:
    normal_reports = {}
    perturbation_reports = {}
    for path in sorted(results_dir.glob("*.txt")):
        classified = classify_report(path.stem)
        if classified is None:
            continue
        kind, key = classified
        methods, order, sample_size, accuracy = parse_report(path)
        report = {
            "methods": methods,
            "order": order,
            "n": sample_size,
            "accuracy": accuracy,
        }
        if kind == "normal":
            normal_reports[key] = report
        else:
            perturbation_reports[key] = report

    if "overall" not in normal_reports:
        raise ValueError(f"No overall_results.txt in {results_dir}")

    pair_keys = [key for key in normal_reports if key != "overall"]
    if not pair_keys:
        raise ValueError(f"No per-pair reports in {results_dir}")

    method_order = _check_method_order(normal_reports, "normal")
    pert_method_order = _check_method_order(perturbation_reports, "perturbation")

    data = {key: dict(report["methods"]) for key, report in normal_reports.items()}
    meta = {
        key: {"n": report["n"], "accuracy": report["accuracy"]}
        for key, report in normal_reports.items()
    }
    pert_meta = {
        key: {"n": report["n"], "accuracy": report["accuracy"]}
        for key, report in perturbation_reports.items()
    }

    if pert_method_order:
        renamed = [perturbed_method_name(name) for name in pert_method_order]
        for pair in data:
            for method in renamed:
                data[pair][method] = None
        for pair, report in perturbation_reports.items():
            if pair not in data:
                raise ValueError(
                    f"Perturbation report {pair!r} has no matching normal report in {results_dir}"
                )
            for name, metrics in report["methods"].items():
                data[pair][perturbed_method_name(name)] = metrics
        method_order = method_order + renamed

    return data, method_order, meta, pert_meta


def _perturbation_count_dirs(results_dir: Path) -> list[Path]:
    dirs = [
        path for path in results_dir.iterdir()
        if path.is_dir() and _COUNT_DIR_RE.fullmatch(path.name)
    ]
    return sorted(dirs, key=lambda path: int(path.name.split("-", 1)[0]))


def _freeze_metrics(metrics: dict | None) -> tuple | None:
    if metrics is None:
        return None
    frozen = []
    for key, value in sorted(metrics.items()):
        frozen.append((key, None if value != value else value))
    return tuple(frozen)


def _baseline_snapshot(data: dict) -> dict:
    snapshot = {}
    for pair, methods in data.items():
        for method, metrics in methods.items():
            if is_baseline_method(method):
                snapshot[(pair, method)] = _freeze_metrics(metrics)
    return snapshot


def _assert_same_baselines(loaded: list[tuple]) -> None:
    ref_count, ref_data, *_ = loaded[0]
    ref = _baseline_snapshot(ref_data)
    for count, data, *_rest in loaded[1:]:
        snapshot = _baseline_snapshot(data)
        if snapshot == ref:
            continue
        missing = sorted(set(ref) - set(snapshot))
        extra = sorted(set(snapshot) - set(ref))
        if missing or extra:
            raise ValueError(
                f"Baseline methods at {count} perturbations do not match "
                f"{ref_count} perturbations (missing {missing[:3]}, extra {extra[:3]})"
            )
        for key, ref_metrics in ref.items():
            if snapshot[key] != ref_metrics:
                pair, method = key
                raise ValueError(
                    f"Baseline {method!r} for {pair} at {count} perturbations "
                    f"differs from the same baseline at {ref_count} perturbations"
                )


def _assert_same_field(loaded: list[tuple], index: int, label: str) -> None:
    ref_count = loaded[0][0]
    ref = loaded[0][index]
    for item in loaded[1:]:
        if item[index] == ref:
            continue
        raise ValueError(
            f"{label} at {item[0]} perturbations does not match {ref_count} perturbations"
        )


def _varying_methods(methods: list[str]) -> list[str]:
    return [method for method in methods if not is_baseline_method(method)]


def _merge_perturbation_counts(loaded: list[tuple]) -> tuple[dict, list[str], dict, dict]:
    """Keep baselines once and repeat every other method for each count.

    ``loaded`` entries are ``(count, data, methods, meta, pert_meta)`` as
    returned by :func:`load_results`, ordered by increasing count.
    """
    _assert_same_baselines(loaded)
    _assert_same_field(loaded, 3, "Translation accuracy")
    _assert_same_field(loaded, 4, "Perturbation accuracy")

    ref_count, ref_data, ref_methods, ref_meta, ref_pert_meta = loaded[0]
    varying = _varying_methods(ref_methods)
    for count, _data, methods, _meta, _pert_meta in loaded[1:]:
        if _varying_methods(methods) != varying:
            raise ValueError(
                f"Non-baseline methods at {count} perturbations do not match "
                f"{ref_count} perturbations"
            )
        if set(_data) != set(ref_data):
            raise ValueError(
                f"Language pairs at {count} perturbations do not match "
                f"{ref_count} perturbations"
            )

    by_count = {count: data for count, data, *_rest in loaded}
    merged_methods = [method for method in ref_methods if is_baseline_method(method)]
    for method in varying:
        for count in by_count:
            merged_methods.append(with_perturbation_count(method, count))

    merged = {}
    for pair in ref_data:
        merged[pair] = {}
        for method in merged_methods:
            if is_baseline_method(method):
                merged[pair][method] = ref_data[pair].get(method)
                continue
            source = by_count[perturbation_count(method)]
            merged[pair][method] = source[pair].get(_without_count(method))
    return merged, merged_methods, ref_meta, ref_pert_meta


def load_results_tree(results_dir: Path) -> tuple[dict, list[str], dict, dict]:
    """Load one results directory, merging every ``k-perturbations`` subdirectory.

    A directory with no such subdirectories is loaded as a single report set,
    which is the layout used before counts were split apart.
    """
    if not results_dir.is_dir():
        raise ValueError(f"No such results directory: {results_dir}")

    count_dirs = _perturbation_count_dirs(results_dir)
    flat_reports = [
        path for path in sorted(results_dir.glob("*.txt"))
        if classify_report(path.stem) is not None
    ]
    if count_dirs and flat_reports:
        names = ", ".join(path.name for path in flat_reports)
        raise ValueError(
            f"{results_dir} has both perturbation-count subdirectories and "
            f"top-level reports ({names})"
        )
    if not count_dirs:
        return load_results(results_dir)

    loaded = []
    for path in count_dirs:
        count = int(path.name.split("-", 1)[0])
        loaded.append((count, *load_results(path)))
    return _merge_perturbation_counts(loaded)


def _cell(value: float | None, metric: str, best_text: str | None) -> str:
    if value is None:
        return "---"
    text = f"{value:.3f}"
    if metric == "Skill Score" and value >= 0:
        text = r"\phantom{-}" + text
    if best_text is not None and text.replace(r"\phantom{-}", "") == best_text:
        text = r"\textbf{" + text + "}"
    return text


def _best_text(values: list[float | None], metric: str) -> str | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    formatted = [f"{value:.3f}" for value in present]
    if metric in HIGHER_IS_BETTER:
        return max(formatted, key=float)
    return min(formatted, key=float)


def _grouped(methods: list[str]) -> list[list[str]]:
    buckets = {name: [] for name in GROUP_ORDER}
    for method in methods:
        buckets[method_group(method)].append(method)
    return [buckets[name] for name in GROUP_ORDER if buckets[name]]


def _column_header(pairs: list[str]) -> str:
    labels = " & ".join(latex_pair_header(pair) for pair in pairs)
    return f"Method & {labels} \\\\"


def _metric_rows(data: dict, pairs: list[str], methods: list[str], metric: str) -> list[str]:
    best_by_pair = {
        pair: _best_text(
            [_metric_value(data, pair, method, metric) for method in methods], metric
        )
        for pair in pairs
    }
    rows = []
    for group_index, group in enumerate(_grouped(methods)):
        if group_index:
            rows.append(r"\addlinespace")
        for method in group:
            cells = " & ".join(
                _cell(_metric_value(data, pair, method, metric), metric, best_by_pair[pair])
                for pair in pairs
            )
            rows.append(f"{method_abbrev(method)} & {cells} \\\\")
    return rows


def _accuracy_sentence(pairs: list[str], meta: dict, pert_meta: dict | None = None) -> str:
    parts = []
    for pair in pairs:
        info = meta[pair]
        parts.append(f"{latex_pair_label(pair)} {info['accuracy']:.2f}\\% ($n={info['n']}$)")
    sentence = "Translation accuracy: " + "; ".join(parts) + "."
    if not pert_meta:
        return sentence
    pert_parts = []
    for pair in pairs:
        info = pert_meta.get(pair)
        if info is None:
            continue
        pert_parts.append(
            f"{latex_pair_label(pair)} {info['accuracy']:.2f}\\% ($n={info['n']}$)"
        )
    if pert_parts:
        sentence += " Perturbation accuracy: " + "; ".join(pert_parts) + "."
    return sentence


def write_tables(
    data: dict, methods: list[str], meta: dict, path: Path, pert_meta: dict | None = None
) -> None:
    pairs = sorted(data, key=pair_sort_key)
    accuracy = _accuracy_sentence(pairs, meta, pert_meta)
    note = abbrev_note(methods)
    header = _column_header(pairs)
    column_spec = "@{}l" + "r" * len(pairs) + "@{}"
    sections = [
        ("ECE (10 bins)", r"ECE (10 bins, lower is better)"),
        ("Brier", r"Brier score (lower is better)"),
        ("Skill Score", r"Skill score (higher is better)"),
    ]

    lines = [
        "% Requires \\usepackage{booktabs,longtable}",
        "% Point estimates. The best value in each column of a block is bold.",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{longtable}{" + column_spec + "}",
        r"\caption{ECE, Brier score, and skill score of confidence methods for code translation. "
        + accuracy
        + " "
        + note
        + r"\label{tab:translation-calibration}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    for index, (metric, title) in enumerate(sections):
        if index:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(pairs) + 1) + r"}{l}{\textit{" + title + r"}} \\")
        lines.append(r"\midrule")
        lines.extend(_metric_rows(data, pairs, methods, metric))

    lines.extend([
        r"\end{longtable}",
        r"}",
        "",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{longtable}{" + column_spec + "}",
        r"\caption{AUROC of confidence methods for code translation. "
        + accuracy
        + r" Higher is better; the best value in each column is bold. "
        + note
        + r"\label{tab:translation-auroc}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ])
    lines.extend(_metric_rows(data, pairs, methods, "AUROC"))
    lines.extend([
        r"\end{longtable}",
        r"}",
        "",
    ])
    path.write_text("\n".join(lines))


def _shades(hex_color: str, count: int) -> list[str]:
    red, green, blue = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
    if count == 1:
        lights = [light]
    else:
        lights = np.linspace(max(0.32, light - 0.16), min(0.72, light + 0.18), count)
    shades = []
    for level in lights:
        r, g, b = colorsys.hls_to_rgb(hue, float(level), sat)
        shades.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return shades


def method_colors(methods: list[str]) -> list[str]:
    colors = {}
    for group in _grouped(methods):
        name = method_group(group[0])
        for method, color in zip(group, _shades(GROUP_COLORS[name], len(group))):
            colors[method] = color
    return [colors[method] for method in methods]


def plot_pair(
    pair: str,
    data: dict,
    methods: list[str],
    meta: dict,
    out_dir: Path,
    pert_meta: dict | None = None,
) -> None:
    metrics = (
        ("AUROC", "AUROC (higher is better)"),
        ("Brier", "Brier (lower is better)"),
        ("Skill Score", "Skill score (higher is better)"),
        ("ECE (10 bins)", "ECE (lower is better)"),
    )
    row_height = 0.38 if len(methods) <= 40 else 0.16
    fig_height = max(13.0, row_height * len(methods) + 2.2)
    fig, axes = plt.subplots(2, 2, figsize=(15, fig_height), sharey=True, constrained_layout=True)
    positions = np.arange(len(methods))
    colors = method_colors(methods)
    hatches = ["///" if is_perturbed_method(method) else None for method in methods]

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = np.array(
            [
                np.nan if (v := _metric_value(data, pair, method, metric)) is None else v
                for method in methods
            ],
            dtype=float,
        )
        bars = ax.barh(positions, values, color=colors, height=0.78)
        for bar, hatch in zip(bars, hatches):
            if hatch is not None:
                bar.set_hatch(hatch)
                bar.set_edgecolor("white")
                bar.set_linewidth(0.4)
        if metric in {"AUROC", "Brier", "ECE (10 bins)"}:
            ax.set_xlim(left=0)
        if metric == "AUROC":
            ax.axvline(0.5, color="#666666", linestyle="--", linewidth=0.8)
        if metric == "Skill Score":
            ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="x", linestyle=":", color="#cccccc")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].set_yticks(positions)
    label_size = 8 if len(methods) <= 28 else 6 if len(methods) > 40 else 7
    axes[0, 0].set_yticklabels(methods, fontsize=label_size)
    axes[0, 0].invert_yaxis()

    handles = [
        Patch(facecolor=GROUP_COLORS[method_group(group[0])], label=GROUP_LABELS[method_group(group[0])])
        for group in _grouped(methods)
    ]
    ncol = 3 if len(handles) <= 9 else 4
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False)
    info = meta[pair]
    title = f"{display_pair_label(pair)}   (n={info['n']}, accuracy={info['accuracy']:.2f}%)"
    if pert_meta and pair in pert_meta:
        pinfo = pert_meta[pair]
        title += f"\nperturbed: n={pinfo['n']}, accuracy={pinfo['accuracy']:.2f}%"
    counts = sorted({c for method in methods if (c := perturbation_count(method)) is not None})
    if counts:
        title += "\nconsistency scores at " + ", ".join(str(c) for c in counts) + " perturbations"
    fig.suptitle(title)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / pair
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate(results_dir: Path, output_dir: Path) -> None:
    data, methods, meta, pert_meta = load_results_tree(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_path = output_dir / "tables.tex"
    write_tables(data, methods, meta, tables_path, pert_meta)
    chart_dir = output_dir / "charts"
    for pair in sorted(data, key=pair_sort_key):
        plot_pair(pair, data, methods, meta, chart_dir, pert_meta)
    print(f"Wrote {tables_path}")
    print(f"Wrote charts to {chart_dir}")
    counts = sorted({c for method in methods if (c := perturbation_count(method)) is not None})
    if counts:
        print(f"Included consistency scores for perturbation counts {counts}")
    if pert_meta:
        print(f"Included perturbation methods for {len(pert_meta)} report(s)")
    else:
        print("No perturbation reports found; wrote normal metrics only")


def main() -> None:
    for model in ['qwen3-coder-next', 'llama-3.1-8b-instruct', 'gpt-4.1-nano', 'deepseek-v4-flash']:
        if model != 'qwen3-coder-next' and model != 'deepseek-v4-flash':
            continue
        aggregate(Path(f"data/code_translation/xcodeeval/{model}/analysis_results"), Path(f"data/code_translation/xcodeeval/{model}/analysis_results/aggregated"))


if __name__ == "__main__":
    main()
