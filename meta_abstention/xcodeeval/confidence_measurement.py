from meta_abstention.untils.similarity_computation import codebertscore_sim, codebert_cosine_sim, unixcoder_sim
import json
import random
import copy

_SIMILARITY_FNS = {
    'codebertscore': codebertscore_sim,
    'codebertcosine': codebert_cosine_sim,
    'unixcoder': unixcoder_sim,
}

# (source_code similarity key, translation similarity key) for each metric.
_METRIC_KEYS = {
    'codebertscore': ('code_codebertscore', 'translation_codebertscore'),
    'codebertcosine': ('code_codebertcosine', 'translation_codebertcosine'),
    'unixcoder': ('code_unixcoder', 'translation_unixcoder'),
}

# SPUQ variants: confidence field name, metric, whether to invert source-code similarity,
# and whether to include a self-pair of (weight=1, translation_sim=1).
_SPUQ_VARIANTS = [
    ('spuq_codebert_score', 'codebertscore', False, True),
    ('spuq_codebert_cosine', 'codebertcosine', False, True),
    ('spuq_unixcoder', 'unixcoder', False, True),
    ('spuq_codebert_score_reverse', 'codebertscore', True, False),
    ('spuq_codebert_cosine_reverse', 'codebertcosine', True, False),
    ('spuq_unixcoder_reverse', 'unixcoder', True, False),
]

_VERBALIZED_WEIGHTED = [
    ('average_verbalized_confidence_codebert_score_weighted', 'codebertscore'),
    ('average_verbalized_confidence_codebert_cosine_weighted', 'codebertcosine'),
    ('average_verbalized_confidence_unixcoder_weighted', 'unixcoder'),
]


def _pair_similarities(text_a: str, text_b: str) -> dict:
    return {name: fn(text_a, text_b) for name, fn in _SIMILARITY_FNS.items()}


def compute_similarities(translations: str, output_path: str, translation_index: int = 0):
    similarities = {}
    with open(translations, 'r') as t:
        data = json.load(t)
        for _, item in data.items():
            try:
                submissions = item['submissions']
                codes = [s['source_code'] for s in submissions]
                translations_list = [s['translation'][translation_index]['translated_code'] for s in submissions]
                code_uids = [s['code_uid'] for s in submissions]

                for i, uid_i in enumerate(code_uids):
                    similarities.setdefault(uid_i, {})
                    for j, uid_j in enumerate(code_uids):
                        if i == j:
                            continue
                        code_sims = _pair_similarities(codes[i], codes[j])
                        translation_sims = _pair_similarities(translations_list[i], translations_list[j])
                        similarities[uid_i][uid_j] = {
                            f'code_{name}': code_sims[name]
                            for name in _SIMILARITY_FNS
                        } | {
                            f'translation_{name}': translation_sims[name]
                            for name in _SIMILARITY_FNS
                        }

                with open(output_path, 'w') as f:
                    json.dump(similarities, f)
            except Exception:
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


def _verbalized_weighted_average(similarities: dict, submission: dict,
                                 filtered_submissions: list, metric: str, translation_index: int) -> float:
    source_key, _ = _METRIC_KEYS[metric]
    code_uid = submission['code_uid']
    own = submission['translation'][translation_index]['confidence']['verbalization']
    weighted = own
    total_source = 1.0
    for other in filtered_submissions:
        source_sim = similarities[code_uid][other['code_uid']][source_key]
        weighted += other['translation'][translation_index]['confidence']['verbalization'] * source_sim
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
    total_verbalized = confidence['verbalization'] + sum(
        other['translation'][translation_index]['confidence']['verbalization']
        for other in filtered_submissions
    )
    confidence['average_verbalized_confidence'] = total_verbalized / n

    for field, metric in _VERBALIZED_WEIGHTED:
        confidence[field] = _verbalized_weighted_average(
            similarities, submission, filtered_submissions, metric, translation_index
        )


def compute_confidence(similarities_path: str, exec_results_path: str, output_path: str, translation_index: int = 0, n_perturbations: int = 5, seed: int = 42):
    with open(similarities_path, 'r') as s:
        similarities = json.load(s)
    with open(exec_results_path, 'r') as e:
        translation_exec_results = json.load(e)

    rand = random.Random(seed)

    for _, item in translation_exec_results.items():
        submissions = copy.deepcopy(item['submissions'])

        for submission in item['submissions']:
            code_uid = submission['code_uid']
            rand.shuffle(submissions)
            filtered_submissions = [s for s in submissions if s['code_uid'] != code_uid][:n_perturbations]

            _add_similarity_based_confidence(similarities, submission, filtered_submissions, translation_index)

    with open(output_path, 'w') as e:
        json.dump(translation_exec_results, e, indent=4)
