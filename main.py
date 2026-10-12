import json
import os
import re
import threading

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:N matches nvidia-smi

import torch  # noqa: E402
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL = os.getenv("MODEL", "gpt2-large")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./adapter")
DEVICE = os.getenv("DEVICE", "cuda:0")
MAX_INPUT_CHARS = 500

# Prompt format and term matching must match GenPlus-Translator/common.py
TASKS = {
    "en-to-slang": "Rewrite this English text as Gen Z / Gen Alpha brainrot slang.",
    "slang-to-en": "Rewrite this brainrot slang as plain English.",
    "define": "Answer this question about Gen Z / Gen Alpha slang.",
}


def build_prompt(task: str, text: str) -> str:
    instruction = TASKS[task]
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": f"{instruction}\n\n{text}"}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"


class TermMatcher:
    """Finds vocabulary terms in text on word boundaries."""

    def __init__(self, vocab):
        self.by_form = {}
        for term in vocab:
            for form in [term["word"], *term.get("aliases", [])]:
                self.by_form[form.lower()] = term
        forms = sorted(self.by_form, key=len, reverse=True)
        pattern = "|".join(re.escape(f) for f in forms)
        self.regex = re.compile(rf"(?<![\w'])(?:{pattern})(?![\w'])", re.IGNORECASE)

    def find(self, text: str):
        res = []
        for m in self.regex.finditer(text):
            term = self.by_form[m.group(0).lower()]
            res.append(dict(word=term["word"], definition=term["definition"], start=m.start(), end=m.end()))
        return res

    def lookup(self, word: str):
        return self.by_form.get(word.strip().lower())


with open("./vocabulary.json") as f:
    vocab = json.load(f)
matcher = TermMatcher(vocab)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": DEVICE})
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
generate_lock = threading.Lock()  # one GPU, one generation at a time


@torch.no_grad()
def run_model(task: str, text: str) -> str:
    enc = tokenizer(build_prompt(task, text), return_tensors="pt", add_special_tokens=False).to(model.device)
    with generate_lock:
        out = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("\n###")[0].strip()


def error(status: int, code: str, message: str):
    return jsonify(error=dict(code=code, message=message)), status


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return dict(status="ok", model=MODEL)


@app.get("/vocabulary")
def vocabulary():
    terms = sorted(({"word": t["word"], "definition": t["definition"]} for t in vocab), key=lambda t: t["word"].lower())
    return dict(terms=terms)


@app.post("/translate")
def translate():
    data = request.get_json(silent=True) or {}
    # "Translation-Type" / "User-Input" are the v1 field names; still accepted
    direction = (data.get("direction") or data.get("Translation-Type") or "").lower()
    text = (data.get("text") or data.get("User-Input") or "").strip()

    if direction not in ("en-to-slang", "slang-to-en"):
        return error(400, "invalid_direction", 'direction must be "en-to-slang" or "slang-to-en".')
    if not text:
        return error(400, "empty_text", "Enter some text to translate.")
    if len(text) > MAX_INPUT_CHARS:
        return error(422, "text_too_long", f"Text is limited to {MAX_INPUT_CHARS} characters.")

    try:
        translation = run_model(direction, text)
    except Exception as e:
        app.logger.exception(e)
        return error(500, "generation_failed", "The model failed to produce a translation.")

    # Highlight terms on the slang side of the pair
    if direction == "en-to-slang":
        terms = [dict(t, side="output") for t in matcher.find(translation)]
    else:
        terms = [dict(t, side="input") for t in matcher.find(text)]

    v1_words = [[t["word"], t["definition"]] for t in {t["word"]: t for t in terms}.values()]
    return dict(translation=translation, terms=terms, result=translation, words=v1_words)


@app.post("/define")
def define():
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word or len(word) > 60:
        return error(400, "invalid_word", "Send the slang word to explain.")
    term = matcher.lookup(word)
    name = term["word"] if term else word
    try:
        explanation = run_model("define", f'What does "{name}" mean?')
    except Exception as e:
        app.logger.exception(e)
        return error(500, "generation_failed", "The model failed to produce an explanation.")
    return dict(word=name, definition=term["definition"] if term else None, explanation=explanation)


if __name__ == "__main__":
    # AHost passes the port in $PORT
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
