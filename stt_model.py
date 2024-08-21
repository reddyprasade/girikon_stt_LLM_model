import jiwer
import numpy as np
from datetime import datetime
date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from dotenv import load_dotenv
import pandas as pd
import os 
import torch
import girikon_stt as stt
import grammatical_correction as gc
import text_emotion as te
from database import voice_transcribe_grammatical_correction_emotion

load_dotenv()

def grammatical_correction(transcribe_text):
    results = gc.grammatical_correction_model(transcribe_text)
    return results
    
def emotion_text(text):
    #emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮","confident":" 😎",'other':"O"}
    prediction = te.predict_emotions(text)
    probability = te.get_prediction_proba(text)
    results_text = {'Predications':prediction, 
                    'Max_Probability_emotion':np.max(probability*100)}
    return results_text

def Calculate_WER(reference,hypothesis):
    transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveEmptyStrings()])
    reference = transform(reference)
    hypothesis = transform(hypothesis)
    error = jiwer.wer(reference, hypothesis)
    return error

def girikon_stt_model(voice_path: str):
    model_path = os.getenv('stt_model_path')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = stt.load_model(model_path).to(device)
    
    # Transcribe audio file
    transcribe_result = model.transcribe(voice_path, fp16=False)
    text = transcribe_result["text"]
    
    # Perform grammatical correction
    grammatical_correction_text = grammatical_correction(text)
    
    # Analyze emotion of text
    emotion_text_result = emotion_text(text)
    
    # Calculate WER
    wer = Calculate_WER(text, grammatical_correction_text)
    
    # Create results dictionary
    results = {
        'voice_path': voice_path,
        'transcribe': text,
        'emotion_text': emotion_text_result,
        'wer': wer,
        'grammatical_correction_text': grammatical_correction_text,
        'transcribe_date_time': datetime.now()}
    
    # Store data in database
    sd = voice_transcribe_grammatical_correction_emotion(results)
    if sd:
        print("DB acknowledged Is Success")
    return results




