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
    'default-model': 'xiaomi/mimo-v2.5',
    'max-o4-tokens': 10000,
}
