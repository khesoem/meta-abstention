import os

data = {
    'n-variants-per-task': 4,
    'output-dir': 'data',
    'humanevalplus-output-file': 'humanevalplus_modified.json',
}

completion = {
    'output-file': 'humanevalplus_completions.json',
    'model': 'mistralai/codestral-2508',
}

llm = {
    'llm-invocation-cache-dir': 'cache/llm_invocations',
    'api-url': 'https://openrouter.ai/api/v1',
    'openrouter-api-key': os.environ['OPENROUTER_API_KEY_ABSTENTION'],
    'default-temp': 0,
    'default-sample-size': 1,
    'default-improvement-iterations': 0,
    'default-model': 'mistralai/codestral-2508',
    'max-o4-tokens': 10000,
}

exec_eval = {
    'api-url': 'http://localhost:5003/api/execute_code'
}

translation = {
    'num-execution-processes': 6,
    'default-model': 'openai/gpt-4.1-nano',
    'default-temp': 0,
    'langs': {
        'python': 'Python 3',
        'java': 'Java 8',
        'cpp': 'GNU C++11',
    },
    'max-tries': 3,
    'number-of-generated-tests': 40,
    'number-of-considered-generated-tests': 20,
    'max-runtime-seconds': 5,
    'number-of-perturbations': 5,
    'seed': 42,
}

xcodeeval = {
    'path': os.environ['XCODEEVAL_PATH'],
}