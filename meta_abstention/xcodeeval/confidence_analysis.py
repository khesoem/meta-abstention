import json
import copy
import random
import scipy.stats

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats


def expected_calibration_error(confidences, correctness, n_bins=5):
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


def reliability_table(confidences, correctness, n_bins=5):
    """Print a reliability diagram in text form."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    print(f"  {'bin':>12} {'n':>4} {'mean_conf':>10} {'accuracy':>10}")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == 0 \
               else (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            print(f"  [{lo:.2f},{hi:.2f}] {0:>4}         -          -")
            continue
        print(f"  [{lo:.2f},{hi:.2f}] {mask.sum():>4} "
              f"{confidences[mask].mean():>10.3f} {correctness[mask].mean():>10.3f}")


def _safe_metric(y, c, fn):
    """Guard AUROC etc. against degenerate bootstrap resamples."""
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return fn(y, c)


def calibration_report(confidences, correctness, name="method",
                       n_bins=5, n_boot=5000, seed=0):
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    n = len(confidences)

    def metrics(c, y):
        return {
            "AUROC":         _safe_metric(y, c, roc_auc_score),
            "Brier":         brier_score_loss(y, c),
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
    print("  Reliability:")
    reliability_table(confidences, correctness, n_bins)
    return point, boot


def paired_bootstrap_diff(conf_a, conf_b, correctness, metric_fn,
                          name_a="A", name_b="B", n_boot=5000, seed=0):
    """Paired bootstrap: is method B better than A on the *same* samples?"""
    a = np.asarray(conf_a, dtype=float)
    b = np.asarray(conf_b, dtype=float)
    y = np.asarray(correctness, dtype=int)
    n = len(y)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.sum() == 0 or ys.sum() == n:
            continue
        diffs.append(metric_fn(ys, b[idx]) - metric_fn(ys, a[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value for H0: diff == 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    print(f"\nPaired bootstrap ({metric_fn.__name__}): {name_b} - {name_a}")
    print(f"  mean diff = {diffs.mean():+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   p ≈ {p:.3f}")
    return diffs

def run_confidence_analysis(execution_results_path: str, output_path: str):
    with open(execution_results_path, 'r') as f:
        execution_results = json.load(f)
    
    correct_cnt = 0
    verbalized_confidence_scores = []
    perturbed_confidence_scores = []
    correctness_scores = []
    for _, item in execution_results.items():
        # Deep copy of submissions
        submissions = copy.deepcopy(item['submissions'])
        for submission in item['submissions']:
            translation = submission['translation'][0]
            verbalized_confidence = translation['confidence']['verbalization']
            verbalized_confidence_scores.append(verbalized_confidence)

            # Shuffle submissions
            random.Random(42).shuffle(submissions)
            filtered_submissions = [s for s in submissions if s['code_uid'] != submission['code_uid']][:5]
            # Get average verbalized confidence of the first 5 and verbalized_confidence
            average_verbalized_confidence = (sum([s['translation'][0]['confidence']['verbalization'] for s in filtered_submissions]) + verbalized_confidence) / 6
            perturbed_confidence_scores.append(average_verbalized_confidence)

            # It is correct if for all items in exec_result['data']['exec_outcome'] are 'PASSED'
            correctness = 1 if all(item['exec_outcome'] == 'PASSED' for item in translation['exec_result']['data']) else 0
            correct_cnt += correctness
            correctness_scores.append(correctness)
    
    print(f'Correctness rate: {correct_cnt}')
    calibration_report(verbalized_confidence_scores, correctness_scores, "Verbalized")
    calibration_report(perturbed_confidence_scores, correctness_scores, "Perturbed")

    paired_bootstrap_diff(
        verbalized_confidence_scores, perturbed_confidence_scores,
        correctness_scores, roc_auc_score,
        name_a="Verbalized", name_b="Perturbed",
    )