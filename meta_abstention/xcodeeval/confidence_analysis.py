import json
import copy
import random
import scipy.stats
import os
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import KFold
from scipy import stats
import logging


def expected_calibration_error(confidences, correctness, n_bins=10):
    """ECE with equal-width bins. Use few bins for small n."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == 0 \
               else (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(confidences[mask].mean() - correctness[mask].mean())
    return ece


def skill_score(confidences, correctness):
    """Brier Skill Score vs an unskilled base-rate predictor.

    Following Spiess et al., "Calibration and Correctness of Language Models
    for Code": SS = (B_ref - B_actual) / B_ref, where B_ref = p_r * (1 - p_r)
    and p_r is the empirical correctness rate. Positive SS (perfect = 1) beats
    always predicting the base rate; negative is worse than that baseline.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    p_r = correctness.mean()
    b_ref = p_r * (1.0 - p_r)
    if b_ref == 0:
        return np.nan
    b_actual = brier_score_loss(correctness, confidences)
    return (b_ref - b_actual) / b_ref


def _safe_metric(y, c, fn):
    """Guard AUROC etc. when every label is the same class."""
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return fn(y, c)


def _correlation(confidences, correctness, fn):
    if np.unique(confidences).size < 2 or np.unique(correctness).size < 2:
        return np.nan
    return fn(confidences, correctness).statistic


def calibration_report(confidences, correctness, name="method", output_file=None,
                       n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    n = len(confidences)

    point = {
        "AUROC":         _safe_metric(correctness, confidences, roc_auc_score),
        "Brier":         brier_score_loss(correctness, confidences),
        "Skill Score":   skill_score(confidences, correctness),
        f"ECE ({n_bins} bins)": expected_calibration_error(confidences, correctness, n_bins),
        "Spearman rho":  _correlation(confidences, correctness, stats.spearmanr),
        "Pearson r":     _correlation(confidences, correctness, stats.pearsonr),
    }

    lines = [f"\n=== {name}  (n={n}, accuracy={correctness.mean():.2%}) ==="]
    for k, v in point.items():
        lines.append(f"  {k:15s}: {v:6.3f}")
    report = "\n".join(lines) + "\n"

    if output_file:
        with open(output_file, 'a') as f:
            f.write(report)
    else:
        print(report, end="")

    return point


# Spiess et al., "Calibration and Correctness of Language Models for Code":
# fit a logistic regression to logit(confidence), five folds, each fifth
# scaled by a model fit on the other four fifths. The split is shuffled with
# a fixed seed so repeated runs of the same file stay comparable.
_PLATT_FOLDS = 5
_PLATT_SEED = 0
PLATT_SUFFIX = " (Platt)"

# (confidence dict key, report display name). Raw and Platt-scaled versions
# are both reported.
_BASELINE_METRICS = [
    ('verbalization', 'Simple Verbalized'),
    ('average_token_probability', 'Average Token Probability'),
    ('average_token_probability_geometric', 'Average Token Probability Geometric'),
    ('generated_sequence_probability', 'Generated Sequence Probability'),
]


def _confidence_logits(confidences: np.ndarray) -> np.ndarray:
    """Logit of each confidence, with exact 0 and 1 kept in order.

    Logit is undefined at 0 and 1. Those scores stay strictly below every
    interior score, or strictly above it, so a positive slope cannot swap
    them with a real intermediate value. The gap is one nat beyond the
    nearest interior logit. When the sample is only 0s and 1s, the two
    levels are -1 and +1.
    """
    probabilities = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    if not np.isfinite(probabilities).all():
        raise ValueError("confidence scores must be finite")

    logits = np.empty(probabilities.shape, dtype=float)
    interior = (probabilities > 0.0) & (probabilities < 1.0)
    if interior.any():
        logits[interior] = logit(probabilities[interior])
        if np.any(probabilities == 0.0):
            logits[probabilities == 0.0] = float(logits[interior].min()) - 1.0
        if np.any(probabilities == 1.0):
            logits[probabilities == 1.0] = float(logits[interior].max()) + 1.0
        return logits

    logits[probabilities == 0.0] = -1.0
    logits[probabilities == 1.0] = 1.0
    return logits


def platt_scale(confidences, correctness, n_folds: int = _PLATT_FOLDS,
                seed: int = _PLATT_SEED) -> np.ndarray:
    """Five-fold Platt scaling of a confidence score.

    For each fold, an unregularized logistic regression is fit on the logit
    of the other folds' confidences and applied to the held-out fold. The
    returned scores are out-of-fold probabilities in the original order, so
    calibration metrics are not computed on the data used to fit the scaler.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    n = len(confidences)
    if n != len(correctness):
        raise ValueError("confidences and correctness must have the same length")
    if n == 0:
        return confidences.copy()

    scaled = np.empty(n, dtype=float)
    base_rate = float(correctness.mean())
    logits = _confidence_logits(confidences).reshape(-1, 1)
    n_splits = min(n_folds, n)
    if n_splits < 2 or np.unique(logits).size < 2 or np.unique(correctness).size < 2:
        scaled[:] = base_rate
        return scaled

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(logits):
        y_train = correctness[train_idx]
        x_train = logits[train_idx]
        if np.unique(y_train).size < 2 or np.unique(x_train).size < 2:
            scaled[test_idx] = float(y_train.mean())
            continue
        # C=inf is unregularized MLE: the two Platt parameters are a slope
        # and an intercept on the logit, with no penalty pulling them to 0.
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        model.fit(x_train, y_train)
        scaled[test_idx] = model.predict_proba(logits[test_idx])[:, 1]
    return scaled


def _wants_platt(name: str) -> bool:
    """Baselines and output-consistency scores are reported raw and Platt-scaled."""
    if name in {display for _, display in _BASELINE_METRICS}:
        return True
    return "Output Consistency Score" in name


def _scheduled_reports() -> list[tuple[str, str, str]]:
    """(kind, confidence key, display name) in the order reports are written."""
    items: list[tuple[str, str, str]] = []
    raw_keys: set[str] = set()
    platt_keys: set[str] = set()

    def add(key: str, name: str) -> None:
        if key not in raw_keys:
            items.append(("raw", key, name))
            raw_keys.add(key)
        if _wants_platt(name) and key not in platt_keys:
            items.append(("platt", key, f"{name}{PLATT_SUFFIX}"))
            platt_keys.add(key)

    for key, name in _CONFIDENCE_METRICS:
        add(key, name)
    for key, name in _BASELINE_METRICS:
        add(key, name)
    return items


def _truncate_report(output_file: str | None) -> None:
    """Drop the previous consistency block onward, keeping the preamble."""
    if not output_file or not os.path.exists(output_file):
        return
    with open(output_file, 'r') as f:
        lines = f.readlines()
    kept = []
    for line in lines:
        # Consistency scores, and Platt blocks written after them, are
        # regenerated below. Cut at whichever of those appears first.
        if "Output Consistency Score" in line or PLATT_SUFFIX in line:
            break
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    with open(output_file, 'w') as f:
        f.writelines(kept)
        if kept and not kept[-1].endswith("\n"):
            f.write("\n")


def _report_headers(output_file: str | None) -> set[str]:
    if not output_file or not os.path.exists(output_file):
        return set()
    headers = set()
    with open(output_file, 'r') as f:
        for line in f:
            marker = " (n="
            if line.startswith("=== ") and marker in line:
                headers.add(line[4:line.index(marker)].strip())
    return headers


# (confidence dict key, report display name)
_CONFIDENCE_METRICS = [
    # ('spuq_codebert_score', 'SPUQ CodeBERT Score'),
    # ('spuq_codebert_score_reverse', 'SPUQ CodeBERT Score Reverse'),
    # ('spuq_codebleu', 'SPUQ CodeBLEU'),
    # ('spuq_codebleu_reverse', 'SPUQ CodeBLEU Reverse'),
    # ('spuq_unixcoder', 'SPUQ Unixcoder'),
    # ('spuq_unixcoder_reverse', 'SPUQ Unixcoder Reverse'),
    ('generated_test_output_consistency_score', 'Generated Test Output Consistency Score'),
    ('orginal_test_-4_output_consistency_score', 'Original Test -4 Output Consistency Score'),
    ('orginal_test_-3_output_consistency_score', 'Original Test -3 Output Consistency Score'),
    ('orginal_test_-2_output_consistency_score', 'Original Test -2 Output Consistency Score'),
    ('orginal_test_-1_output_consistency_score', 'Original Test -1 Output Consistency Score'),
    ('orginal_test_0_output_consistency_score', 'Original Test 0 Output Consistency Score'),
    ('orginal_test_3_output_consistency_score', 'Original Test 3 Output Consistency Score'),
    ('orginal_test_6_output_consistency_score', 'Original Test 6 Output Consistency Score'),
    ('orginal_test_9_output_consistency_score', 'Original Test 9 Output Consistency Score'),
    ('orginal_test_12_output_consistency_score', 'Original Test 12 Output Consistency Score'),
    # ('average_verbalized_confidence', 'Average Verbalized'),
    # ('average_verbalized_confidence_codebert_score_weighted', 'Average Verbalized CodeBERT Score'),
    # ('average_verbalized_confidence_codebleu_weighted', 'Average Verbalized CodeBLEU'),
    # ('average_verbalized_confidence_unixcoder_weighted', 'Average Verbalized Unixcoder'),
    # ('average_average_token_probability', 'Average of Average Token Probability'),
    # ('average_average_token_probability_codebert_score_weighted', 'Average of Average Token Probability CodeBERT Score'),
    # ('average_average_token_probability_codebleu_weighted', 'Average of Average Token Probability CodeBLEU'),
    # ('average_average_token_probability_unixcoder_weighted', 'Average of Average Token Probability Unixcoder'),
    # ('average_average_token_probability_geometric', 'Average of Average Token Probability Geometric'),
    # ('average_average_token_probability_geometric_codebert_score_weighted', 'Average of Average Token Probability Geometric CodeBERT Score'),
    # ('average_average_token_probability_geometric_codebleu_weighted', 'Average of Average Token Probability Geometric CodeBLEU'),
    # ('average_average_token_probability_geometric_unixcoder_weighted', 'Average of Average Token Probability Geometric Unixcoder'),
    # ('average_generated_sequence_probability', 'Average Generated Sequence Probability'),
    # ('average_generated_sequence_probability_codebert_score_weighted', 'Average Generated Sequence Probability CodeBERT Score'),
    # ('average_generated_sequence_probability_codebleu_weighted', 'Average Generated Sequence Probability CodeBLEU'),
    # ('average_generated_sequence_probability_unixcoder_weighted', 'Average Generated Sequence Probability Unixcoder'),
]


def run_confidence_analysis(execution_results_paths: list[str], translation_index: int = 0, output_file: str = None):
    _truncate_report(output_file)
    # Baselines already printed above the consistency block are left in place.
    preserved_headers = _report_headers(output_file)
    reports = [
        (kind, key, name)
        for kind, key, name in _scheduled_reports()
        if not (kind == "raw" and name in preserved_headers)
    ]

    logging.info(f"Running confidence analysis for {execution_results_paths}")
    score_keys = list(dict.fromkeys(key for _, key, _ in reports))
    scores = {key: [] for key in score_keys}
    correctness_scores = []

    for execution_results_path in execution_results_paths:
        with open(execution_results_path, 'r') as f:
            execution_results = json.load(f)
        for _, item in execution_results.items():
            for submission in item['submissions']:
                if '_' in submission['code_uid'] and submission['code_uid'].split('_')[1] != '0': # Skip LLM-made submissions
                    continue

                translation = submission['translation'][translation_index]
                conf = translation['confidence']
                for key in score_keys:
                    scores[key].append(conf[key])

                # It is correct if for all items in exec_result['data']['exec_outcome'] are 'PASSED'
                correctness = 1 if all(item['exec_outcome'] == 'PASSED' for item in translation['exec_result']['data']) else 0
                correctness_scores.append(correctness)


    for kind, key, name in reports:
        if kind == "platt":
            logging.info(
                "Platt scaling %s over %d folds (seed=%d)", name, _PLATT_FOLDS, _PLATT_SEED,
            )
            values = platt_scale(scores[key], correctness_scores)
        else:
            values = scores[key]
        calibration_report(values, correctness_scores, name, output_file=output_file)

def run_confidence_analysis_for_batch(execution_results_dir: str, translation_index: int = 0, output_file: str = None, perturbation: bool = False):
    execution_results_paths = []

    for file in os.listdir(execution_results_dir):
        if ('perturbation' in file) != perturbation:
            continue
        execution_results_paths.append(os.path.join(execution_results_dir, file))

    run_confidence_analysis(execution_results_paths, translation_index, output_file=output_file)