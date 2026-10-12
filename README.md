# Gen+ Translator 🧠🚽
World's first bi-drectional brainrot translator.<br>

<i>Youwei Zhen 2024</i>

[Live Production Site](https://brainrot.youweizhen.com/)

## Woah! View the complete Gen+ Translator Repos
- [LLM + Finetuning](https://github.com/AntoDono/GenPlus-Translator)
- [Frontend](https://github.com/AntoDono/GenPlus-Translator-Frontend)
- [Backend](https://github.com/AntoDono/GenPlus-Translator-Backend)

## Backend
This is the backend of Gen+ Translator, built on Python Flask. It serves the fine-tuned model (base model + LoRA adapter in `./adapter`) from [GenPlus-Translator](https://github.com/AntoDono/GenPlus-Translator).

### API
- `POST /translate` with `{"direction": "en-to-slang" | "slang-to-en", "text": "..."}` (max 500 characters) returns `{"translation": "...", "terms": [{"word", "definition", "side", "start", "end"}]}`. Each term is a slang word found on the slang side of the pair: `side` is `"input"` or `"output"`, and `start`/`end` are character offsets into that text. The v1 fields (`Translation-Type`, `User-Input`, `result`, `words`) still work.
- `POST /define` with `{"word": "rizz"}` returns `{"word", "definition", "explanation"}`, where `explanation` is written by the model.
- `GET /vocabulary` returns every term and its definition.
- `GET /health` returns the status and the loaded base model.

Errors return `{"error": {"code", "message"}}` with status 400, 422 or 500.

## Setup

1. Create a virtual environment and install dependencies:
```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```
2. Copy `.env.example` to `.env` (git-ignored) and set:
```.env
MODEL="gpt2-large"       <- base model the adapter was trained on
ADAPTER_PATH="./adapter"
DEVICE="cuda:0"          <- or "cpu"
```
3. Run the server. It listens on `$PORT`, default 5000:
```bash
python main.py
```
