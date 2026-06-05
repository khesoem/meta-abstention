# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses Poetry with a local `.venv` (configured in `poetry.toml`). Python 3.13+ required.

```bash
poetry install          # install dependencies
poetry run meta-abstention  # run the CLI entry point
poetry run pytest       # run tests
```

Activate the venv directly with `source .venv/bin/activate` if preferred.

**Required environment variable**: `OPENROUTER_API_KEY` — read at import time in `config.py`. The process will crash without it.

## Architecture

All LLM calls go through **OpenRouter's OpenAI-compatible API** (`https://openrouter.ai/api/v1`), regardless of which underlying model is used (GPT, Gemini, DeepSeek, etc.).

### LLM layer (`meta_abstention/llm/`)

- **`invocation.py`** — three core data classes: `Prompt` (messages + temp + sample_size + model), `Response` (list of `Sample`s), and `Invocation` (prompt + response + timestamp). `Prompt.hash()` uses MD5 over the full JSON serialization to produce the cache key.
- **`llm_adapter.py`** — `LLMAdapter` base class. Owns the `OpenAI` client, and implements cache read/write logic. Cache files live in `cache/llm_invocations/` and are named `{prompt_hash}-{index}.json`.
- **`openai.py`, `gemini.py`, `deepseek_qwen.py`** — thin concrete subclasses, each hardcoding a model string and implementing `get_response(prompt) -> Response`. Adding a new model means subclassing `LLMAdapter` and implementing `get_response`.

### Caching

`LLMAdapter` has two independent flags: `read_from_cache` and `save_to_cache`. Multiple samples for the same prompt hash are stored as separate files (`hash-0.json`, `hash-1.json`, …). Reading only returns the first match.

### Config (`meta_abstention/config.py`)

Single `llm` dict with defaults: temperature 0, sample size 1, 0 improvement iterations, default model `openai/gpt-4.1-nano`. Modify here to change global defaults.

### Data layer (`meta_abstention/data/data_manipulation.py`)

Loads the `evalplus/humanevalplus` HuggingFace dataset, uses the configured LLM to generate N paraphrased variants of each task's `prompt` field (description only — function signature and doctests are left intact), and writes the result to `data/humanevalplus_modified.json`.

Output schema per task:
```json
{ "task_id": "HumanEval/0", "original_prompt": "...", "modified_prompts": ["variant1", ...] }
```

`n-variants-per-task` is controlled by `config.data`. Run the step via `poetry run meta-abstention` or directly with `python -m meta_abstention.data.data_manipulation`.
