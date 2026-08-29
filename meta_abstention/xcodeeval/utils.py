import requests
from meta_abstention import config


def execute_code(language, source_code, unittests, timeout=600):
    payload = {
        "language": language,          # e.g. "Python 3" — must match a runtime_name
        "source_code": source_code,
        "unittests": unittests,
        "compile_cmd": None,                 # None -> use ExecEval's defaults
        "compile_flags": None,
        "execute_cmd": None,
        "execute_flags": None,
        "block_network": True,
        "stop_on_first_fail": False,         # set True if you just want pass/fail
        "use_sanitizer": False,
    }

    r = requests.post(config.exec_eval['api-url'], json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()