import os 
from dotenv import load_dotenv

load_dotenv()

def grammatical_correction_model(stt_output):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("grammatical_correction"))
    model = T5ForConditionalGeneration.from_pretrained(os.getenv("grammatical_correction"))
    input_text = 'Fix grammatical errors in this sentence with Spelling and Punctuation: {}'.format(stt_output)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=1024)
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return edited_text
