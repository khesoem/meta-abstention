import json
import copy
import random
import scipy.stats
import os
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
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
        "Spearman rho":  stats.spearmanr(confidences, correctness).statistic,
        "Pearson r":     stats.pearsonr(confidences, correctness).statistic,
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

# (confidence dict key, report display name)
_CONFIDENCE_METRICS = [
    # ('verbalization', 'Simple Verbalized'),
    # ('average_token_probability', 'Average Token Probability'),
    # ('average_token_probability_geometric', 'Average Token Probability Geometric'),
    # ('generated_sequence_probability', 'Generated Sequence Probability'),
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
    # remove all lines that come after "Output Consistency Score" in the output file
    if output_file:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        with open(output_file, 'w') as f:
            for line in lines:
                if "Output Consistency Score" in line:
                    break
                f.write(line)

    logging.info(f"Running confidence analysis for {execution_results_paths}")
    scores = {key: [] for key, _ in _CONFIDENCE_METRICS}
    correctness_scores = []

    for execution_results_path in execution_results_paths:
        with open(execution_results_path, 'r') as f:
            execution_results = json.load(f)
        for _, item in execution_results.items():
            for submission in item['submissions']:
                translation = submission['translation'][translation_index]
                conf = translation['confidence']
                for key, _ in _CONFIDENCE_METRICS:
                    scores[key].append(conf[key])

                # It is correct if for all items in exec_result['data']['exec_outcome'] are 'PASSED'
                correctness = 1 if all(item['exec_outcome'] == 'PASSED' for item in translation['exec_result']['data']) else 0
                correctness_scores.append(correctness)

    for key, name in _CONFIDENCE_METRICS:
        calibration_report(scores[key], correctness_scores, name, output_file=output_file)

def run_confidence_analysis_for_batch(execution_results_dir: str, translation_index: int = 0, output_file: str = None):
    execution_results_paths = []

    for file in os.listdir(execution_results_dir):
        execution_results_paths.append(os.path.join(execution_results_dir, file))

    run_confidence_analysis(execution_results_paths, translation_index, output_file=output_file)