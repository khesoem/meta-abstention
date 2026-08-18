from meta_abstention.untils.similarity_computation import codebertscore_sim, codebert_cosine_sim, unixcoder_sim
import json
import random
import copy

def compute_similarities(translations: str, output_path: str):
    similarities = {}
    with open(translations, 'r') as t:
        data = json.load(t)
        for _, item in data.items():
            try:
                submission_codes = []
                translation_codes = []
                code_uids = []
                for submission in item['submissions']:
                    code_uid = submission['code_uid']
                    code_uids.append(code_uid)
                    code = submission['source_code']
                    translation = submission['translation'][0]['translated_code']
                    submission_codes.append(code)
                    translation_codes.append(translation)
                
                for i, code1 in enumerate(submission_codes):
                    for j, code2 in enumerate(submission_codes):
                        if i == j:
                            continue
                        code_codebertscore = codebertscore_sim(code1, code2)
                        code_codebertcosine = codebert_cosine_sim(code1, code2)
                        codeunixcoder = unixcoder_sim(code1, code2)
                        translation_codebertscore = codebertscore_sim(translation_codes[i], translation_codes[j])
                        translation_codebertcosine = codebert_cosine_sim(translation_codes[i], translation_codes[j])
                        translation_unixcoder = unixcoder_sim(translation_codes[i], translation_codes[j])

                        if code_uids[i] not in similarities:
                            similarities[code_uids[i]] = {}

                        similarities[code_uids[i]][code_uids[j]] = {
                            'code_codebertscore': code_codebertscore,
                            'code_codebertcosine': code_codebertcosine,
                            'code_unixcoder': codeunixcoder,
                            'translation_codebertscore': translation_codebertscore,
                            'translation_codebertcosine': translation_codebertcosine,
                            'translation_unixcoder': translation_unixcoder
                        }
                            
                with open(output_path, 'w') as f:
                    json.dump(similarities, f)
            except Exception as e:
                continue

def _add_similarity_based_confidence(similarities: dict, submission: dict, filtered_submissions: list):
    code_uid = submission['code_uid']

    ### SPUQ method-codebert: average similarity of translations weighted by the similarity of the source codes
    total_translation_similarity = 0
    total_source_code_similarity = 0
    for other_submission in filtered_submissions:
        other_code_uid = other_submission['code_uid']
        source_code_similarity = similarities[code_uid][other_code_uid]['code_codebertscore']
        translation_similarity = similarities[code_uid][other_code_uid]['translation_codebertscore']
        total_translation_similarity += translation_similarity * source_code_similarity
        total_source_code_similarity += source_code_similarity
    total_source_code_similarity += 1
    total_translation_similarity += 1
    submission['translation'][0]['confidence']['spuq_codebert_score'] = total_translation_similarity / total_source_code_similarity

    ### SPUQ method-codebert-cosine: average similarity of translations weighted by the similarity of the source codes
    total_translation_similarity = 0
    total_source_code_similarity = 0
    for other_submission in filtered_submissions:
        other_code_uid = other_submission['code_uid']
        source_code_similarity = similarities[code_uid][other_code_uid]['code_codebertcosine']
        translation_similarity = similarities[code_uid][other_code_uid]['translation_codebertcosine']
        total_translation_similarity += translation_similarity * source_code_similarity
        total_source_code_similarity += source_code_similarity
    total_source_code_similarity += 1
    total_translation_similarity += 1
    submission['translation'][0]['confidence']['spuq_codebert_cosine'] = total_translation_similarity / total_source_code_similarity
    
    ### SPUQ method-unixcoder: average similarity of translations weighted by the similarity of the source codes
    total_translation_similarity = 0
    total_source_code_similarity = 0
    for other_submission in filtered_submissions:
        other_code_uid = other_submission['code_uid']
        source_code_similarity = similarities[code_uid][other_code_uid]['code_unixcoder']
        translation_similarity = similarities[code_uid][other_code_uid]['translation_unixcoder']
        total_translation_similarity += translation_similarity * source_code_similarity
        total_source_code_similarity += source_code_similarity
    total_source_code_similarity += 1
    total_translation_similarity += 1
    submission['translation'][0]['confidence']['spuq_unixcoder'] = total_translation_similarity / total_source_code_similarity

    ### Average verbalized confidence: average of the confidence of the translations
    total_confidence = 0
    total_confidence_codebert_score_weighted = 0
    total_confidence_codebert_cosine_weighted = 0
    total_confidence_unixcoder_weighted = 0
    total_source_code_similarity_codebert = 0
    total_source_code_similarity_codebert_cosine = 0
    total_source_code_similarity_unixcoder = 0
    for other_submission in filtered_submissions:
        other_code_uid = other_submission['code_uid']
        total_confidence += other_submission['translation'][0]['confidence']['verbalization']
        total_confidence_codebert_score_weighted += other_submission['translation'][0]['confidence']['verbalization'] * similarities[code_uid][other_code_uid]['code_codebertscore']
        total_confidence_codebert_cosine_weighted += other_submission['translation'][0]['confidence']['verbalization'] * similarities[code_uid][other_code_uid]['code_codebertcosine']
        total_confidence_unixcoder_weighted += other_submission['translation'][0]['confidence']['verbalization'] * similarities[code_uid][other_code_uid]['code_unixcoder']
        total_source_code_similarity_codebert += similarities[code_uid][other_code_uid]['code_codebertscore']
        total_source_code_similarity_codebert_cosine += similarities[code_uid][other_code_uid]['code_codebertcosine']
        total_source_code_similarity_unixcoder += similarities[code_uid][other_code_uid]['code_unixcoder']
    total_confidence += submission['translation'][0]['confidence']['verbalization']
    total_confidence_codebert_score_weighted += submission['translation'][0]['confidence']['verbalization']
    total_confidence_codebert_cosine_weighted += submission['translation'][0]['confidence']['verbalization']
    total_confidence_unixcoder_weighted += submission['translation'][0]['confidence']['verbalization']
    total_source_code_similarity_codebert += 1
    total_source_code_similarity_codebert_cosine += 1
    total_source_code_similarity_unixcoder += 1
    submission['translation'][0]['confidence']['average_verbalized_confidence'] = total_confidence / (len(filtered_submissions) + 1)
    submission['translation'][0]['confidence']['average_verbalized_confidence_codebert_score_weighted'] = total_confidence_codebert_score_weighted / total_source_code_similarity_codebert
    submission['translation'][0]['confidence']['average_verbalized_confidence_codebert_cosine_weighted'] = total_confidence_codebert_cosine_weighted / total_source_code_similarity_codebert_cosine
    submission['translation'][0]['confidence']['average_verbalized_confidence_unixcoder_weighted'] = total_confidence_unixcoder_weighted / total_source_code_similarity_unixcoder

def compute_confidence(similarities_path: str, translation_exec_results_path: str, output_path: str, n_perturbations: int = 5, seed: int = 42):
    with open(similarities_path, 'r') as s:
        similarities = json.load(s)
    with open(translation_exec_results_path, 'r') as e:
        translation_exec_results = json.load(e)
    
    rand = random.Random(seed)

    for _, item in translation_exec_results.items():
        submissions = copy.deepcopy(item['submissions'])

        for submission in item['submissions']:
            code_uid = submission['code_uid']
            rand.shuffle(submissions)
            filtered_submissions = [s for s in submissions if s['code_uid'] != code_uid][:n_perturbations]

            _add_similarity_based_confidence(similarities, submission, filtered_submissions)
        
    with open(output_path, 'w') as e:
        json.dump(translation_exec_results, e, indent=4)