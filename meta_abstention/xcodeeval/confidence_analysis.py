import json
import copy
import random
import scipy.stats

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats


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
    """Guard AUROC etc. against degenerate bootstrap resamples."""
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return fn(y, c)


def calibration_report(confidences, correctness, name="method",
                       n_bins=10, n_boot=5000, seed=0):
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    n = len(confidences)

    def metrics(c, y):
        return {
            "AUROC":         _safe_metric(y, c, roc_auc_score),
            "Brier":         brier_score_loss(y, c),
            "Skill Score":   skill_score(c, y),
            f"ECE ({n_bins} bins)": expected_calibration_error(c, y, n_bins),
            "Spearman rho":  stats.spearmanr(c, y).statistic,
            "Pearson r":     stats.pearsonr(c, y).statistic,
        }

    point = metrics(confidences, correctness)

    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, v in metrics(confidences[idx], correctness[idx]).items():
            if not np.isnan(v):
                boot[k].append(v)

    print(f"\n=== {name}  (n={n}, accuracy={correctness.mean():.2%}) ===")
    for k, v in point.items():
        arr = np.array(boot[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {k:15s}: {v:6.3f}   95% CI [{lo:6.3f}, {hi:6.3f}]")
    return point, boot

def run_confidence_analysis(execution_results_path: str, translation_index: int = 0):
    with open(execution_results_path, 'r') as f:
        execution_results = json.load(f)

    correct_cnt = 0
    simple_verbalized = []
    average_token_probability = []
    generated_sequence_probability = []
    spuq_codebert_score = []
    spuq_codebert_cosine = []
    spuq_unixcoder = []
    average_verbalized = []
    average_verbalized_codebert_score = []
    average_verbalized_codebert_cosine = []
    average_verbalized_unixcoder = []
    spuq_codebert_score_reverse = []
    spuq_codebert_cosine_reverse = []
    spuq_unixcoder_reverse = []
    correctness_scores = []
    for _, item in execution_results.items():
        for submission in item['submissions']:
            translation = submission['translation'][translation_index]
            conf = translation['confidence']
            simple_verbalized.append(conf['verbalization'])
            average_token_probability.append(conf['average_token_probability'])
            generated_sequence_probability.append(conf['generated_sequence_probability'])
            spuq_codebert_score.append(conf['spuq_codebert_score'])
            spuq_codebert_cosine.append(conf['spuq_codebert_cosine'])
            spuq_unixcoder.append(conf['spuq_unixcoder'])
            average_verbalized.append(conf['average_verbalized_confidence'])
            average_verbalized_codebert_score.append(conf['average_verbalized_confidence_codebert_score_weighted'])
            average_verbalized_codebert_cosine.append(conf['average_verbalized_confidence_codebert_cosine_weighted'])
            average_verbalized_unixcoder.append(conf['average_verbalized_confidence_unixcoder_weighted'])
            spuq_codebert_score_reverse.append(conf['spuq_codebert_score_reverse'])
            spuq_codebert_cosine_reverse.append(conf['spuq_codebert_cosine_reverse'])
            spuq_unixcoder_reverse.append(conf['spuq_unixcoder_reverse'])

            # It is correct if for all items in exec_result['data']['exec_outcome'] are 'PASSED'
            correctness = 1 if all(item['exec_outcome'] == 'PASSED' for item in translation['exec_result']['data']) else 0
            correct_cnt += correctness
            correctness_scores.append(correctness)

    calibration_report(simple_verbalized, correctness_scores, "Simple Verbalized")
    calibration_report(average_token_probability, correctness_scores, "Average Token Probability")
    calibration_report(generated_sequence_probability, correctness_scores, "Generated Sequence Probability")
    calibration_report(spuq_codebert_score, correctness_scores, "SPUQ CodeBERT Score")
    calibration_report(spuq_codebert_score_reverse, correctness_scores, "SPUQ CodeBERT Score Reverse")
    calibration_report(spuq_codebert_cosine, correctness_scores, "SPUQ CodeBERT Cosine")
    calibration_report(spuq_codebert_cosine_reverse, correctness_scores, "SPUQ CodeBERT Cosine Reverse")
    calibration_report(spuq_unixcoder, correctness_scores, "SPUQ Unixcoder")
    calibration_report(spuq_unixcoder_reverse, correctness_scores, "SPUQ Unixcoder Reverse")
    calibration_report(average_verbalized, correctness_scores, "Average Verbalized")
    calibration_report(average_verbalized_codebert_score, correctness_scores, "Average Verbalized CodeBERT Score")
    calibration_report(average_verbalized_codebert_cosine, correctness_scores, "Average Verbalized CodeBERT Cosine")
    calibration_report(average_verbalized_unixcoder, correctness_scores, "Average Verbalized Unixcoder")
