# Gen+ Translator 🧠🚽
World's first bi-drectional brainrot translator.<br>

<i>Youwei Zhen 2024</i>

## Woah! View the complete Gen+ Translator Repos
- [LLM + Finetuning](https://github.com/AntoDono/GenPlus-Translator)
- [Frontend](https://github.com/AntoDono/GenPlus-Translator-Frontend)
- [Backend](https://github.com/AntoDono/GenPlus-Translator-Backend)

## Backend
This is the backend part of Gen+ Translator, built on python flask. This essentially is an server that hosts the model of Gen+ Translator.

## Setup

1. Create virtual environment (optional):
```bash
python -m venv venv
```
2. Activate the environment:
```bash
source ./venv/bin/activate <- linux-based

.\venv\Scripts\activate <- windows
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Create a .env and fill out the following:
```.env
MODEL <- Base model for the PeFT
DEVICE <- CPU for cpu, cuda:0 for GPU
SUPABASE_URL <- Superbase API stuff (optional)
SUPABASE_KEY <- Superbase API stuff (optional)
```
5. Run server:
```bash
python main.py
```
