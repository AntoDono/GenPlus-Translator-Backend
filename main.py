from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL")
device = os.getenv("DEVICE")
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, max_length=256)
model = PeftModel.from_pretrained(model, "./adapter", max_length=256).to(device)

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
        f"{instruction}\nInput: {user_input}", 
        return_tensors="pt").to(device)
    
    output_ids = model.generate(
        **encoded
    )
    
    try:
        output = tokenizer.decode(output_ids[0]).split("Output: ")[1].split("Words replaced")[0]
    except:
        return 500
    
    return dict(result=output)    
    
if __name__ == "__main__":
    app.run(debug=True)