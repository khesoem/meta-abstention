"""Aggregate code-translation confidence reports into LaTeX tables and charts.

Reads the text reports written by ``confidence_analysis.py`` (one file per
source-target pair, plus ``overall_results.txt``) and writes:

* a table of ECE, Brier score, and Brier skill score
* a table of AUROC
* one chart per language pair (and one for the pooled results) comparing
  every method on those four metrics

Each metric line is a point estimate. A trailing 95% interval, when still
present on an older line, is ignored. Spearman and Pearson correlations are
ignored. The best value in each table column is bold: higher is better for
AUROC and skill score, lower is better for ECE and Brier.

Generated-test output consistency is reported only for some models (currently
qwen3-coder-next) and is included only when that block is in the reports.
"""

import argparse
import colorsys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HEADER_RE = re.compile(
    r"^===\s+(.+?)\s+\(n=(\d+),\s*accuracy=([\d.]+)%\)\s+===\s*$"
)
METRIC_RE = re.compile(
    r"^\s+(?P<name>AUROC|Brier|Skill Score|ECE \(10 bins\)|Spearman rho|Pearson r)\s*:\s*"
    r"(?P<value>-?\d+\.\d+)"
    r"(?:\s+95% CI\s*\[\s*-?\d+\.\d+,\s*-?\d+\.\d+\])?\s*$"
)

SKIPPED_METRICS = {"Spearman rho", "Pearson r"}
CALIBRATION_METRICS = ("ECE (10 bins)", "Brier", "Skill Score")
HIGHER_IS_BETTER = {"AUROC", "Skill Score"}

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
    if "Generated Test Output Consistency Score" in methods:
        note += r"GTOC = generated-test output consistency. "
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
    "other": "#7F7F7F",
}


def method_group(name: str) -> str:
    if name.startswith("SPUQ "):
        return "spuq"
    if name.startswith("Average Verbalized"):
        return "avg_verbalized"
    if name.startswith("Average of Average Token Probability"):
        return "avg_token"
    if name.startswith("Average Generated Sequence Probability"):
        return "avg_sequence"
    if name.startswith("Average Token Probability"):
        return "token"
    if name == "Generated Sequence Probability":
        return "sequence"
    if name == "Simple Verbalized":
        return "verbalized"
    if name == "Generated Test Output Consistency Score":
        return "gen_test"
    if name.startswith("Original Test "):
        return "orig_test"
    return "other"


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
    try:
        return METHOD_ABBREV[name]
    except KeyError as exc:
        raise KeyError(f"No table abbreviation for method {name!r}") from exc


def display_pair_label(pair: str) -> str:
    if pair == "overall":
        return "Overall"
    source, target = pair.split("-to-")
    return f"{LANG_NAMES[source]} → {LANG_NAMES[target]}"


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


def load_results(results_dir: Path) -> tuple[dict, list[str], dict]:
    reports = {}
    for path in sorted(results_dir.glob("*.txt")):
        key = "overall" if path.stem == "overall_results" else path.stem
        methods, order, sample_size, accuracy = parse_report(path)
        reports[key] = {
            "methods": methods,
            "order": order,
            "n": sample_size,
            "accuracy": accuracy,
        }

    if "overall" not in reports:
        raise ValueError(f"No overall_results.txt in {results_dir}")

    pair_reports = [reports[key] for key in reports if key != "overall"]
    method_order = pair_reports[0]["order"]
    for key, report in reports.items():
        if report["order"] != method_order:
            raise ValueError(f"Method list in {key} does not match the other reports")

    meta = {key: {"n": report["n"], "accuracy": report["accuracy"]} for key, report in reports.items()}
    data = {key: report["methods"] for key, report in reports.items()}
    return data, method_order, meta


def _cell(value: float, metric: str, best_text: str) -> str:
    text = f"{value:.3f}"
    if metric == "Skill Score" and value >= 0:
        text = r"\phantom{-}" + text
    if text.replace(r"\phantom{-}", "") == best_text:
        text = r"\textbf{" + text + "}"
    return text


def _best_text(values: list[float], metric: str) -> str:
    formatted = [f"{value:.3f}" for value in values]
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
        pair: _best_text([data[pair][method][metric] for method in methods], metric)
        for pair in pairs
    }
    rows = []
    for group_index, group in enumerate(_grouped(methods)):
        if group_index:
            rows.append(r"\addlinespace")
        for method in group:
            cells = " & ".join(
                _cell(data[pair][method][metric], metric, best_by_pair[pair]) for pair in pairs
            )
            rows.append(f"{method_abbrev(method)} & {cells} \\\\")
    return rows


def _accuracy_sentence(pairs: list[str], meta: dict) -> str:
    parts = []
    for pair in pairs:
        info = meta[pair]
        parts.append(f"{latex_pair_label(pair)} {info['accuracy']:.2f}\\% ($n={info['n']}$)")
    return "Translation accuracy: " + "; ".join(parts) + "."


def write_tables(data: dict, methods: list[str], meta: dict, path: Path) -> None:
    pairs = sorted(data, key=pair_sort_key)
    accuracy = _accuracy_sentence(pairs, meta)
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
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AUROC of confidence methods for code translation. "
        + accuracy
        + r" Higher is better; the best value in each column is bold. "
        + note
        + "}",
        r"\label{tab:translation-auroc}",
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ])
    lines.extend(_metric_rows(data, pairs, methods, "AUROC"))
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
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


def plot_pair(pair: str, data: dict, methods: list[str], meta: dict, out_dir: Path) -> None:
    metrics = (
        ("AUROC", "AUROC (higher is better)"),
        ("Brier", "Brier (lower is better)"),
        ("Skill Score", "Skill score (higher is better)"),
        ("ECE (10 bins)", "ECE (lower is better)"),
    )
    fig_height = max(13.0, 0.38 * len(methods) + 2)
    fig, axes = plt.subplots(2, 2, figsize=(15, fig_height), sharey=True, constrained_layout=True)
    positions = np.arange(len(methods))
    colors = method_colors(methods)

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = np.array([data[pair][method][metric] for method in methods])
        ax.barh(positions, values, color=colors, height=0.78)
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
    axes[0, 0].set_yticklabels(methods, fontsize=7 if len(methods) > 28 else 8)
    axes[0, 0].invert_yaxis()

    handles = [
        Patch(facecolor=GROUP_COLORS[method_group(group[0])], label=GROUP_LABELS[method_group(group[0])])
        for group in _grouped(methods)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    info = meta[pair]
    fig.suptitle(f"{display_pair_label(pair)}   (n={info['n']}, accuracy={info['accuracy']:.2f}%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / pair
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate(results_dir: Path, output_dir: Path) -> None:
    data, methods, meta = load_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_path = output_dir / "tables.tex"
    write_tables(data, methods, meta, tables_path)
    chart_dir = output_dir / "charts"
    for pair in sorted(data, key=pair_sort_key):
        plot_pair(pair, data, methods, meta, chart_dir)
    print(f"Wrote {tables_path}")
    print(f"Wrote charts to {chart_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/code_translation/xcodeeval/gpt-4.1-nano/analysis_results"),
        help="Directory of per-pair and overall analysis reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/code_translation/xcodeeval/gpt-4.1-nano/analysis_results/aggregated"),
        help="Directory for tables.tex and charts/",
    )
    args = parser.parse_args()
    aggregate(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
