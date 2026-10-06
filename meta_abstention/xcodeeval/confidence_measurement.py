from meta_abstention.untils.similarity_computation import (
    codebleu_sim,
    codebertscore_sim,
    # codebert_cosine_sim,
    unixcoder_sim,
)
import json
import random
import copy
import os
import logging
from meta_abstention import config as conf

_SIMILARITY_FNS = {
    'codebleu': codebleu_sim,
    'codebertscore': codebertscore_sim,
    # 'codebertcosine': codebert_cosine_sim,
    'unixcoder': unixcoder_sim,
}

# (source_code similarity key, translation similarity key) for each metric.
_METRIC_KEYS = {
    'codebleu': ('code_codebleu', 'translation_codebleu'),
    'codebertscore': ('code_codebertscore', 'translation_codebertscore'),
    # 'codebertcosine': ('code_codebertcosine', 'translation_codebertcosine'),
    'unixcoder': ('code_unixcoder', 'translation_unixcoder'),
}

_REQUIRED_PAIR_KEYS = {key for pair in _METRIC_KEYS.values() for key in pair}

# SPUQ variants: confidence field name, metric, whether to invert source-code similarity,
# and whether to include a self-pair of (weight=1, translation_sim=1).
_SPUQ_VARIANTS = [
    ('spuq_codebleu', 'codebleu', False, False),
    ('spuq_codebert_score', 'codebertscore', False, False),
    # ('spuq_codebert_cosine', 'codebertcosine', False, True),
    ('spuq_unixcoder', 'unixcoder', False, False),
    ('spuq_codebleu_reverse', 'codebleu', True, False),
    ('spuq_codebert_score_reverse', 'codebertscore', True, False),
    # ('spuq_codebert_cosine_reverse', 'codebertcosine', True, False),
    ('spuq_unixcoder_reverse', 'unixcoder', True, False),
]

# (output field prefix, confidence source key). Prefixes get `_codebert_score_weighted` etc.
_AGGREGATED_CONFIDENCE_FIELDS = [
    ('average_verbalized_confidence', 'verbalization'),
    ('average_average_token_probability', 'average_token_probability'),
    ('average_average_token_probability_geometric', 'average_token_probability_geometric'),
    ('average_generated_sequence_probability', 'generated_sequence_probability'),
]

_WEIGHTED_METRIC_SUFFIXES = [
    ('codebleu', 'codebleu'),
    ('codebert_score', 'codebertscore'),
    # ('codebert_cosine', 'codebertcosine'),
    ('unixcoder', 'unixcoder'),
]


_LANG_AWARE_METRICS = {'codebleu', 'codebertscore'}


def _pair_complete(pair_sims: dict) -> bool:
    return _REQUIRED_PAIR_KEYS.issubset(pair_sims.keys())


def _metric_sim(name: str, text_a: str, text_b: str, lang: str) -> float:
    fn = _SIMILARITY_FNS[name]
    if name in _LANG_AWARE_METRICS:
        return fn(text_a, text_b, lang=lang)
    return fn(text_a, text_b)


def _fill_missing_similarities(
    pair_sims: dict,
    code_a: str,
    code_b: str,
    translation_a: str,
    translation_b: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    """Compute only missing metric keys; leave already-present values untouched."""
    result = dict(pair_sims)
    for name, (code_key, translation_key) in _METRIC_KEYS.items():
        if code_key not in result:
            result[code_key] = _metric_sim(name, code_a, code_b, source_lang)
        if translation_key not in result:
            result[translation_key] = _metric_sim(
                name, translation_a, translation_b, target_lang
            )
    return result


def compute_similarities(translations: str, output_path: str, translation_index: int = 0,
        source_lang: str = "java", target_lang: str = "python"):
    logging.info(f"Computing similarities for {translations} to {output_path} with translation index {translation_index} and source lang {source_lang} and target lang {target_lang}")

    # if output_path exists, load existing similarities
    if os.path.exists(output_path):
        with open(output_path, 'r') as s:
            similarities = json.load(s)
    else:
        similarities = {}

    with open(translations, 'r') as t:
        data = json.load(t)
        for _, item in data.items():
            try:
                submissions = item['submissions']
                codes = [s['source_code'] for s in submissions]
                translations_list = [s['translation'][translation_index]['translated_code'] for s in submissions]
                code_uids = [s['code_uid'] for s in submissions]

                new_similarity_computed = False
                for i, uid_i in enumerate(code_uids):
                    similarities.setdefault(uid_i, {})
                    for j, uid_j in enumerate(code_uids):
                        if i == j:
                            continue

                        existing = similarities.get(uid_i, {}).get(uid_j) or {}
                        reverse = similarities.get(uid_j, {}).get(uid_i) or {}
                        # Prefer values already stored for this direction; fill gaps from reverse.
                        pair = {**reverse, **existing}

                        if _pair_complete(pair):
                            if not _pair_complete(existing):
                                similarities[uid_i][uid_j] = pair
                                new_similarity_computed = True
                                logging.info(f"Filled similarities for {uid_i} and {uid_j} from reverse")
                            else:
                                logging.info(f"Similarity for {uid_i} and {uid_j} already computed")
                            continue

                        pair = _fill_missing_similarities(
                            pair,
                            codes[i],
                            codes[j],
                            translations_list[i],
                            translations_list[j],
                            source_lang,
                            target_lang,
                        )
                        similarities[uid_i][uid_j] = pair
                        similarities.setdefault(uid_j, {})[uid_i] = pair
                        new_similarity_computed = True
                        logging.info(f"Computed missing similarities for {uid_i} and {uid_j}")

                if new_similarity_computed:
                    with open(output_path, 'w') as f:
                        json.dump(similarities, f)
            except Exception as e:
                logging.error(f"Error computing similarities for {uid_i} and {uid_j}: {e}")
                continue


def _source_weight(pair_sims: dict, metric: str, reverse: bool) -> float:
    source_key, _ = _METRIC_KEYS[metric]
    weight = pair_sims[source_key]
    return (1 - weight) if reverse else weight


def _spuq_score(similarities: dict, code_uid: str, filtered_submissions: list,
                metric: str, reverse: bool, include_self: bool) -> float:
    _, translation_key = _METRIC_KEYS[metric]
    total_translation = 0.0
    total_source = 0.0
    for other in filtered_submissions:
        pair = similarities[code_uid][other['code_uid']]
        source_sim = _source_weight(pair, metric, reverse)
        total_translation += pair[translation_key] * source_sim
        total_source += source_sim
    if include_self:
        total_source += 1
        total_translation += 1
    return total_translation / total_source


def _confidence_weighted_average(similarities: dict, submission: dict,
                                 filtered_submissions: list, metric: str,
                                 translation_index: int, confidence_key: str) -> float:
    source_key, _ = _METRIC_KEYS[metric]
    code_uid = submission['code_uid']
    own = submission['translation'][translation_index]['confidence'][confidence_key]
    weighted = own
    total_source = 1.0
    for other in filtered_submissions:
        source_sim = similarities[code_uid][other['code_uid']][source_key]
        weighted += other['translation'][translation_index]['confidence'][confidence_key] * source_sim
        total_source += source_sim
    return weighted / total_source


def _add_similarity_based_confidence(similarities: dict, submission: dict, filtered_submissions: list, translation_index: int):
    confidence = submission['translation'][translation_index]['confidence']
    code_uid = submission['code_uid']

    for field, metric, reverse, include_self in _SPUQ_VARIANTS:
        confidence[field] = _spuq_score(
            similarities, code_uid, filtered_submissions, metric, reverse, include_self
        )

    n = len(filtered_submissions) + 1
    for out_prefix, confidence_key in _AGGREGATED_CONFIDENCE_FIELDS:
        total = confidence[confidence_key] + sum(
            other['translation'][translation_index]['confidence'][confidence_key]
            for other in filtered_submissions
        )
        confidence[out_prefix] = total / n

        for suffix, metric in _WEIGHTED_METRIC_SUFFIXES:
            confidence[f'{out_prefix}_{suffix}_weighted'] = _confidence_weighted_average(
                similarities, submission, filtered_submissions, metric,
                translation_index, confidence_key
            )

def _add_generated_test_execution_based_confidence(submission: dict, filtered_submissions: list, translation_index: int, generated_test_execution_results: dict):
    submission_code_uid = submission['code_uid']
    submission_exec_result = generated_test_execution_results[submission_code_uid][submission_code_uid]['data']

    if len(submission_exec_result) <= 1: # Compilation error or Timeout error
        submission['translation'][translation_index]['confidence']['generated_test_output_consistency_score'] = 0
        return
    
    consistent_translations = 0
    valid_other_translations = 0

    for other in filtered_submissions:
        other_code_uid = other['code_uid']
        other_exec_result = generated_test_execution_results[submission_code_uid][other_code_uid]['data']

        if len(other_exec_result) <= 1: # Compilation error or Timeout error
            continue
    
        valid_other_translations += 1

        if len(submission_exec_result) != len(other_exec_result):
            continue

        if [e['result'] for e in submission_exec_result] == [e['result'] for e in other_exec_result]:
            consistent_translations += 1

    if valid_other_translations == 0:
        submission['translation'][translation_index]['confidence']['generated_test_output_consistency_score'] = 0
    else:
        submission['translation'][translation_index]['confidence']['generated_test_output_consistency_score'] = consistent_translations / valid_other_translations

def _add_original_test_execution_based_confidence(submission: dict, filtered_submissions: list, translation_index: int):
    for test_cnt in [-4, -3, -2, -1, 0, 3, 6, 9, 12]:

        submission_exec_result = submission['translation'][translation_index]['exec_result']['data']

        if len(submission_exec_result) <= 1: # Compilation error or Timeout error
            submission['translation'][translation_index]['confidence'][f'orginal_test_{test_cnt}_output_consistency_score'] = 0
            continue

        if test_cnt >= 0:
            selected_indices = random.Random(conf.translation['seed']).sample(range(len(submission_exec_result)), len(submission_exec_result))[:test_cnt]
        else:
            selected_indices = random.Random(conf.translation['seed']).sample(range(len(submission_exec_result)), len(submission_exec_result))[:(len(submission_exec_result) // (-test_cnt))]

        consistent_translations = 0
        valid_other_translations = 0

        for other in filtered_submissions:
            other_exec_result = other['translation'][translation_index]['exec_result']['data']

            if len(other_exec_result) <= 1: # Compilation error or Timeout error
                continue
        
            valid_other_translations += 1

            if len(submission_exec_result) != len(other_exec_result):
                continue

            if [submission_exec_result[i]['result'] for i in selected_indices] == [other_exec_result[i]['result'] for i in selected_indices]:
                consistent_translations += 1


        if valid_other_translations == 0:
            submission['translation'][translation_index]['confidence'][f'orginal_test_{test_cnt}_output_consistency_score'] = 0
        else:
            submission['translation'][translation_index]['confidence'][f'orginal_test_{test_cnt}_output_consistency_score'] = consistent_translations / valid_other_translations

def compute_confidence(similarities_path: str, generated_test_execution_results_path: str, exec_results_path: str, output_path: str, translation_index: int = 0, n_perturbations: int = 5, seed: int = conf.translation['seed']):
    with open(similarities_path, 'r') as s:
        similarities = json.load(s)
    with open(generated_test_execution_results_path, 'r') as e:
        generated_test_execution_results = json.load(e)
    with open(exec_results_path, 'r') as e:
        translation_exec_results = json.load(e)

    rand = random.Random(seed)

    for _, item in translation_exec_results.items():
        submissions = copy.deepcopy(item['submissions'])

        for submission in item['submissions']:
            rand.shuffle(submissions)
            code_uid = submission['code_uid']
            filtered_submissions = [s for s in submissions if s['code_uid'] != code_uid][:n_perturbations]

            _add_similarity_based_confidence(similarities, submission, filtered_submissions, translation_index)

            _add_original_test_execution_based_confidence(submission, filtered_submissions, translation_index)

            _add_generated_test_execution_based_confidence(submission, filtered_submissions, translation_index, generated_test_execution_results)

    with open(output_path, 'w') as e:
        json.dump(translation_exec_results, e, indent=4)
