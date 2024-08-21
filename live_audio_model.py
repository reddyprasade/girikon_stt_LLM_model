import os
import datetime
import requests
import torchaudio
import torch
import numpy as np
from stt_model import girikon_stt_model

def call_center_stt_model(audio_url: str, folder_path='call_center',detected_language='en'or'hi'):
    """
    Download an audio file from a URL, process it into chunks, and return the full transcription using STT model.

    :param audio_url: URL of the audio file
    :param folder_path: Path to the folder where the file will be saved. Defaults to 'call_center'
    :param chunk_duration_seconds: Duration of each chunk in seconds. Defaults to 10 seconds
    :return: Full transcript of the audio
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a timestamped filename
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"audiofile_{timestamp_str}.wav"
    file_path = os.path.join(folder_path, file_name)
    
    try:
        response = requests.get(audio_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved to {file_path}")

        model_path = os.getenv('stt_model_path')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import girikon_stt as stt
        model = stt.load_model(model_path).to(device)
        print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )

        transcribe_result = model.transcribe(file_path, language=detected_language, task='transcribe', temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), best_of=5, beam_size=5, suppress_tokens="-1", condition_on_previous_text=True, fp16=False, compression_ratio_threshold=2.4, logprob_threshold=-1., no_speech_threshold=0.6)
        text = transcribe_result["text"]

        from utils import extract_url as eu 
        url_results = eu.extract_url_text(audio_url)
        url_results['transcript'] = text
    
        return url_results
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")
        return None

# Example usage
# audio_url = 'https://cloudphone.tatateleservices.com/file/recording?callId=1724078757.94332&type=rec&token=N1VwdHVOSGVwSXBaVUlzSTBjQnQxZFVnOUV2S0RFSlkxVGUzbkYzdDI5c3c0UUxwNDZsT2pSNWN2ajdlU0pMcjo6YWIxMjM0Y2Q1NnJ0eXl1dQ%3D%3D'
# print(call_center_stt_model(audio_url))
