import json

def clean_non_translated_submissions(original_data_file: str, to_be_cleaned_file: str, output_file: str) -> None:
    original_code_uid_to_translation = {}
    with open(original_data_file, 'r') as f:
        original_data = json.load(f)
        for _, item in original_data.items():
            for submission in item['submissions']:
                original_code_uid_to_translation[submission['code_uid']] = submission['translation']

    code_uid_to_translation = {}
    with open(to_be_cleaned_file, 'r') as f:
        data = json.load(f)
        print(len(data))
        for src_uid, item in data.items():
            if len(item['submissions']) != 6:
                print('too few submissions for', item['problem_description']['src_uid'], item['submissions'][0]['code_uid'], len(item['submissions']))

            selected_submissions = []
            incomplete_submissions = False
            for submission in item['submissions']:
                if not 'translation' in submission or len(submission['translation']) == 0:
                    print('no translation for', submission['code_uid'])
                    if '_0' in submission['code_uid']:
                        submission['translation'] = original_code_uid_to_translation[submission['code_uid'].split('_')[0]]
                        selected_submissions.append(submission)
                    else:
                        incomplete_submissions = True
                else:
                    selected_submissions.append(submission)
            if incomplete_submissions:
                item['submissions'] = selected_submissions
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)