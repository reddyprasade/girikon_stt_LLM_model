import joblib,os
from dotenv import load_dotenv
load_dotenv()
# Load Model
pipe_lr = joblib.load(open(os.getenv("emotion_model_path"),'rb'))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
