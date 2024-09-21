from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json

load_dotenv()

model_name = os.getenv("MODEL")
device = os.getenv("DEVICE")
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

superbase = None

if url == None or key == None:
    print("No superbase credentials, skipping superbase init.")
    supabase = create_client(url, key)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, max_length=256, device_map=device)
model = PeftModel.from_pretrained(model, "./adapter", max_length=256).to(device)

def read_json(path: str) -> any:
    try:
        with open(path, 'r') as r:
            data = json.load(r)
        return data
    except Exception as e:
        print(e)
        return None
    
vocab = read_json("./vocabulary.json")

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
        self.stop_id = stops
        StoppingCriteria.__init__(self)

    def __call__(self, input_ids, scores):
        last_id = input_ids.tolist()[-1] 
        for id in stop_words_ids:
            if id == last_id:
                return True
        return False

stop_words_ids = [tokenizer.eos_token_id]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_ids)])

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/translate', methods=['POST'])
@cross_origin(supports_credentials=True)
def translate():
    
    data = request.get_json()
    translate_type = data.get("Translation-Type").lower()
    user_input = data.get("User-Input")
    
    if translate_type.lower() not in ["en-to-slang", "slang-to-en"]:
        return 400
    
    instruction = "Rewrite the following english sentence to slang and identify the words replaced."
    if translate_type == "slang-to-en":
        instruction = "Rewrite the following slang sentence to english and identify the words replaced."
        
    encoded = tokenizer(
        f"{instruction}\nInput: {user_input}\nOutput: ", 
        return_tensors="pt").to(device)
    
    output_ids = model.generate(
        **encoded,
        stopping_criteria=stopping_criteria
    )
    
    try:
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("Output: ")[1]
        output = output.split("Words replaced:")
        translation = output[0]
        words = output[1].split(", ")
        words_with_definition = []
        for word in words:
            words_with_definition.append([word, vocab.get(word.lower())])
    except:
        return 500
    
    return dict(result=translation, words=words_with_definition)    
    
if __name__ == "__main__":
    app.run(debug=False)