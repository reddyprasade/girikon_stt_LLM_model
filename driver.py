import logging
import os
import time
import uuid
import io
import re
import google.generativeai as genai
# import socketio
from fastapi import FastAPI,Form, Request,UploadFile,File
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form, Request,UploadFile,File
from fastapi.responses import JSONResponse
import io
from typing import List
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv() 
import os
import json
from datetime import datetime
from prompt import prompt_gen


#from database import db
# client = OpenAI(api_key=os.getenv("OPENAIKEY"))
# from stt_model import girikon_stt_model

load_dotenv()
# counter1 = 0
# SOCKET_PATH = os.getenv("SOCKET_PATH", "/socket.io")
app = FastAPI()
# Initialize chat history
chat_history = []
#  mount static files
# app.mount('/static', StaticFiles(directory='static'), name='static')

from bson import ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def get_Json(response_data):
    return json.loads(json.dumps(response_data, cls=JSONEncoder))
    
#Create a json from an object data
def show_json(obj):
    return json.loads(obj.json())

def assistantAnalysis(questionList, answerList, positionedApplied):
    dataInsert = [{"question": quest, "answer": ans} for quest, ans in zip(questionList, answerList)]
    prompt = """
    As the Salesforce Technical Lead for the company, you are responsible for evaluating the hiring expert and excellent candidate on a scale of 0(completely incorrect) to 10(completely correct). You will need to assess their answers based on the questions asked and check the answer with gpt-3.5-turbo knowledge base before score. Your task also involves analyzing the attached QA JSON Array file and explaining the generated score.
    Make sure to highlight 5 weaknesses and 2 strength in candidate response so that evaluation will be more prominent.
 
    Consider the following for each answer:
    1. Understanding of Question
    2. Technical correctness,
    3. Confident Level
    4. Grammar, 
    5. Fluency
    6. Accuracy of the answer as the metrics for evaluation.
    7. Justify the score in brief explanation.
    8. Text Emotion score on the answer. It will be used for bar graph so need actual emotion score. The emotion score should be in between 1 to 10.
     
     
    Mandatory to provide the output in JSON Array format so that it can be converted into dictionary for python. Never use any other format.
    [{
        \"brief_explanation\": \"\",
        \"question_id\": \"\",
        \"score\": {
            \"understanding\":\"\",
            \"technical\":\"\",
            \"confident\":\"\",
            \"grammar\":\"\",
            \"fluency\":\"\",
            \"ans_acc\":\"\"
            },
        \"emotion_score\": {
            \"fear\": \"\",
            \"angry\": \"\",
            \"sad\": \"\",
            \"happy\": \"\",
            \"neutral\": \"\",
            \"Surprise\": \"\"
        }
    }]
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": prompt},
            {"role": "user", "content": """Consider the list data containing question and answer for the detailed report.
                                        [Data]
                                        {dataInsert}
                                        [/Data]""".format(dataInsert = dataInsert)}
        ]
    )
    return completion.choices[0].message

# # create a Socket.IO server
# sio = socketio.AsyncServer(cors_allowed_origins='*',
#                            async_mode='asgi',
#                            logger=False,
#                            engineio_logger=False,
#                            debug=False,
#                            logger_name="socketio")

# # wrap with ASGI application
# print("SOCKET_PATH", SOCKET_PATH)
# sio_app = socketio.ASGIApp(sio, app, socketio_path=SOCKET_PATH)

# app.mount(SOCKET_PATH, sio_app)


# @sio.event
# def connect(sid, environ, auth=None):
#     # raise ConnectionRefusedError('authentication failed')
#     print("connect ", sid)


# @sio.event
# def disconnect(sid):
#     print("disconnect ", sid)


# @sio.on('heartbeat')
# async def heartbeat(sid, data):
#     print("heartbeat", data)
#     return "PONG"

# @sio.on('audio_stream')
# async def audio_stream(sid, data):
#     # print("sid", data)
#     uid = uuid.uuid4().hex
#     uid =sid
#     file_path = os.path.join(os.getcwd(), 'output', f'_{data["id"]}.wav')
#     with open(file_path, 'ab') as f:  # Append mode
#         f.write(data["blob"])
#     # time.sleep(1)
#     print(file_path)
#     _girikon_stt_model = girikon_stt_model(file_path)
#     print(_girikon_stt_model)
#     return _girikon_stt_model
#     return data


# @sio.event
# def disconnect(sid):
#     print('############################################################# disconnect #############################################################', sid)


# @app.get("/v2")
# def read_main():
#     return {"message": "Hello World"}



@app.post("/ai_interview/createAssistant")
async def createAssistantOpenAI(assistantName : str = Form(...)):
    #if model_name !="gemma":
    assistantName = "Interviewer"
    try:
        assistant = client.beta.assistants.create(
        name=assistantName,
        instructions="""You are an interview expert in all programming languages like Python, C, C++, Java, Javascript, HTML, Scala, Haskell and so on. You are an expert in many tools like Docker, Git, JIRA, Kubernetes, AWS services and so on.\n
    Now you must use all your technical knowledge and as an interviewer with high technical skills, you must ask maximum 10 set of questions. Start the interview with the introduction of the candidate, then with the technical question related to project mentioned in resume and skills mentioned in job description.
    Make sure the follow up question should be asked considering the user response. If the user does not know the answer, you must ask different question. Ask only 9 questions. End the interview with the feedback of the candidate.
    """,
        model="gpt-4-turbo-preview",
        tools=[{"type": "file_search"}])
    except Exception as e:
        print(e)
        return {"message": "Assistant not created. API Key is not valid."}
    return {"message":{"response":"Assistant created successfully.",
                       "assistantID":assistant.id}}



@app.post("/ai_interview/createVectorStore")
async def createVectorStoreDB(files: List[UploadFile] = File(...),
                              assistant_id : str = Form(...)):
    try:
        vector_store = client.beta.vector_stores.create(name="Interviewer",
                                                        expires_after={
                                                                        "anchor": "last_active_at",
                                                                        "days": 365
                                                                    })   
        import datetime
        dirName = str(datetime.datetime.now()).replace("-","").replace(" ","").replace(":","").replace(".","")+"_uploads"
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        for file in files:
            file_path = os.path.join(dirName, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
        listOfFiles = os.listdir(dirName)
        # def getData(filelistData):
        #     with open(os.path.join(dirName, filelistData), "rb") as fileContent:
        #         fileContentData = fileContent.read()
        #     return fileContentData
        file_streams = [open(os.path.join(dirName, path), "rb") for path in listOfFiles]

        client.beta.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store.id, files=file_streams)
        assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        )
    except Exception as e:
        print(e)
        return {"message": "Error in creating the vector store.{e}".format(e = e)}

    return {"message":{"response":"Vector store created successfully.",
                       "assistantID":assistant.id,
                       "vectorStore":vector_store.id}}


@app.post("/ai_interview/getAnswerCHATGPT")
async def getAns(assistantID : str = Form(...),
                 vectorStoreID: str = Form(...),
                 threadID : str = Form(...),
                 userResponse: str = Form(...)):
    if threadID == "null":
        thread = client.beta.threads.create(
        messages=[ { "role": "user", "content": "{userResponse}".format(userResponse = userResponse)} ],
        tool_resources={
            "file_search": {
            "vector_store_ids": [vectorStoreID]
            }
        }
        )
        thread_id = thread.id
    else:
        thread_id = threadID
        pass
    if userResponse != "" or userResponse != None:
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistantID,
            additional_messages=[ { "role": "user", "content": "{userResponse}".format(userResponse = userResponse)} ],
            instructions="""Start the interview by asking introduction of the candidate followed by the technical question and concluded by the feedback of the candidate. There must me 10 question in total and then stop the interview.
            Always generate one question at a time, then on the basis of user response generate the another question.
            Don't generate the answer. Use the user response for the next question.
            If the user do not know the answer then ask different question.
            Consider the resume and job description uploaded in vector store for asking the question. Don't add file name."""
            )
    else:
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistantID,
            additional_messages=[ { "role": "user", "content": "Start the interview."}],
            instructions="""Start the interview by asking introduction of the candidate followed by the technical question and end the interview by asking the feedback of the candidate. There must me 10 question in total for the interview.
            Always generate one question at a time, then on the basis of user response generate the another question.
            Don't generate the answer. Use the user response for the next question.
            If the user do not know the answer then ask different question.
            Consider the resume and job description uploaded in vector store for asking the question. Don't add file name."""
            )
    run_id = run.id
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, 
                                                    run_id=run_id)
        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            print("Run failed:", run_status.last_error)
            break
        time.sleep(2) 
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )
    jsonObjectData = show_json(messages)
    print(jsonObjectData)
    text_data = []
    for message in jsonObjectData['data']:
        for content in message['content']:
            if content['type'] == 'text':
                text_data.append(content['text']['value'])
    requiredText = text_data[0].replace(userResponse,"")
    pattern = r"【\d:\d†\w+\.pdf】"
    requiredText = re.sub(pattern, "", requiredText)
    return {"message":{"gptResponse":requiredText,
                       "threadID":thread_id,}}
                       

@app.post("/ai_interview/deleteAssistantOne")
async def deleteOne(assistID: str = Form(...),
                    vectorID: str = Form(...),
                    threadID: str = Form(...)):
    #messages = client.beta.threads.messages.list(threadID)
    #jsonData = show_json(messages)
    #recordsInsert = {"ChatGPT Interview":jsonData}
    #db["interviewChatGPT"].insert_one(recordsInsert)
    try:
        assistant_object = client.beta.assistants.list()
        assistant_object = show_json(assistant_object)
        assistantID = []
        for asst in assistant_object["data"]:
            assistantID.append(asst["id"])
        if assistID in assistantID:
            client.beta.assistants.delete(assistID)
    except:
        return "Assistant ID not correct or already deleted"
    try:
        client.beta.vector_stores.delete(
                                    vector_store_id=vectorID
                                    )
    except:
        return "Vector Store ID not correct or already deleted"
    try:
        client.beta.threads.delete(threadID)
    except:
        return "Thread ID not correct or already deleted"
    return "SUCCESS"
    


@app.post("/ai_interview/deleteAssistantAll")
async def deleteOne():
    assistant_object = client.beta.assistants.list()
    assistant_object = show_json(assistant_object)
    assistantID = []
    for asst in assistant_object["data"]:
        assistantID.append(asst["id"])
    for val in assistantID:
        client.beta.assistants.delete(val)
    return "Success"


@app.post("/ai_interview/getFiles")
async def fileList(vectorID: str = Form(...)):
    vector_store_files = client.beta.vector_stores.files.list(
    vector_store_id=vectorID
    )
    print(vector_store_files)
    return {"response":show_json(vector_store_files)}

@app.post("/ai_interview/concludeInterview")
async def endInterview(assistID : str = Form(...),
                       threadID: str = Form(...),
                       interviewerName: str = Form(...),
                       positionApplied: str = Form(...),
                       jobCode: str = Form(...)):
    messages = client.beta.threads.messages.list(threadID)
    jsonData = show_json(messages)
    questionList = []
    answerList = []
    for index in range(0,len(jsonData["data"]),2):
        if jsonData["data"][index]["content"][0]["text"]["value"]=="Please start the interview.":
            pass
        else:
            questionList.append(jsonData["data"][index]["content"][0]["text"]["value"])
    for index in range(1,len(jsonData["data"]),2):
        if jsonData["data"][index]["content"][0]["text"]["value"]=="Please start the interview.":
            pass
        else:
            answerList.append(jsonData["data"][index]["content"][0]["text"]["value"])
    questionList = questionList[-1::-1]
    answerList = answerList[-1::-1]
    analysisJson = assistantAnalysis(questionList, answerList, positionApplied)
    # print(1)
    # print(analysisJson)
    # if not analysisJson:
    #     analysisJson = {}
    # else:
    #     print(json.loads(analysisJson.__dict__["content"]))
    #     analysisJson = json.loads(analysisJson.__dict__["content"])
    #     print(analysisJson)
    #generateAnalysisPDF(questionList,answerList,interviewerName,positionApplied,jobCode,analysisJson)
    return {"response":{"interviewerName":interviewerName,
            "positionApplied":positionApplied,
            "jobCode":jobCode,
            "feedback":analysisJson,
            "questionList":questionList,
            "answerList":answerList
            }}
    
@app.post("/ai_interview/codingEvaluation")
async def endInterview(codingQuest: str = Form(...),
                       codingAns: str = Form(...),
                       language: str = Form(...)):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": """You are Senior Technical Recruiter who evaluates coding question of {language} programming language. 
         On a scale of 10 considering 0 as minimum score and 10 as maximum score. 
         Generate the score along with the brief explanation on scoring method.
         [Question]
         {codingQuest}
         [/Question]
         [Answer]
         {codingAns}
         [/Answer]
         
         The generated response must be in below format only. Do not use markdown.
         score:,
         briefExplanation:
         """.format(language = language,codingQuest = codingQuest, codingAns = codingAns)},
        {"role": "user", "content": "Evaluate and generate the response."}
    ]
    )
    output = [val for val in completion.choices[0].message.content.split(":")]
    response = [{"score":output[1][0:3].replace(" ","").replace("\\",""),
                 "explanation":output[-1]}]
    return response


# ###############Commenting the s2t model as this model consuming 56 core cpu##################################
from fastapi import FastAPI, File, UploadFile
import shutil

UPLOAD_DIRECTORY = "uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@app.post("/ai_interview/stt")
def stt_model(voice_path: UploadFile = File(...)):
    """
        AI Interview Model For Speech to Text Model STT which can use auto Correction of Grammar Model and 
        Emotion of the Text Model  
    """
    file_location = os.path.join(UPLOAD_DIRECTORY, voice_path.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(voice_path.file, f)
    from stt_model import girikon_stt_model
    stt_results = girikon_stt_model(file_location)
    print(stt_results)
    return get_Json(stt_results)


@app.post("/ai_interview/emp_interview")
async def emp_interview(skills: str = Form(...),
                  userResponse: str = Form(...)):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])

    # Chat templates
    USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
    MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'
    global chat_history
    if userResponse.lower() == "please start the interview.":
        # Generate the initial prompt based on skills
        initial_prompt = prompt_gen(skills, userResponse)
        chat_history.append(USER_CHAT_TEMPLATE.format(prompt=userResponse))

        # Initial response from the model (interview start)
        response = chat.send_message(initial_prompt)
        print("Line 53: response:", response)
        full_response = ""
        for r in response:
            r.resolve()
            if r.candidates and r.candidates[0].content.parts:
                full_response += r.candidates[0].content.parts[0].text
        chat_history.append(MODEL_CHAT_TEMPLATE.format(prompt=full_response))

        # Append the generated initial prompt to the chat history
        chat_history.append(USER_CHAT_TEMPLATE.format(prompt=initial_prompt))

        # Call the model with the initial prompt to get the first question
        full_prompt = ''.join(chat_history) + '<start_of_turn>model\n'
        response = chat.send_message(full_prompt)
        full_response = ""
        for r in response:
            r.resolve()
            if r.candidates and r.candidates[0].content.parts:
                full_response += r.candidates[0].content.parts[0].text

        full_response = full_response.replace('</end_of_turn>', '').replace('<end_of_turn>', '')
        # Append the model's response (first question) to the chat history
        chat_history.append(MODEL_CHAT_TEMPLATE.format(prompt=full_response))
    elif userResponse.lower() != "please start the interview." or userResponse.lower() != "conclude":
        chat_history.append(USER_CHAT_TEMPLATE.format(prompt=userResponse))
        # Call the model with the accumulated chat history
        full_prompt = ''.join(chat_history) + '<start_of_turn>model\n'
        response = chat.send_message(full_prompt)
        full_response = ""
        for r in response:
            r.resolve()
            if r.candidates and r.candidates[0].content.parts:
                full_response += r.candidates[0].content.parts[0].text
        chat_history.append(MODEL_CHAT_TEMPLATE.format(prompt=full_response))
    else:
        # Generate the initial prompt based on skills
        conclude_prompt = prompt_gen(skills, userResponse)
        chat_history.append(USER_CHAT_TEMPLATE.format(prompt=userResponse))

        # Initial response from the model (interview start)
        response = chat.send_message(conclude_prompt)
        print("Line 53: response:", response)
        full_response = ""
        for r in response:
            r.resolve()
            if r.candidates and r.candidates[0].content.parts:
                full_response += r.candidates[0].content.parts[0].text
        chat_history.append(MODEL_CHAT_TEMPLATE.format(prompt=full_response))

        # Append the generated initial prompt to the chat history
        chat_history.append(USER_CHAT_TEMPLATE.format(prompt=conclude_prompt))

        # Call the model with the initial prompt to get the first question
        full_prompt = ''.join(chat_history) + '<start_of_turn>model\n'
        response = chat.send_message(full_prompt)
        full_response = ""
        for r in response:
            r.resolve()
            if r.candidates and r.candidates[0].content.parts:
                full_response += r.candidates[0].content.parts[0].text

        full_response = full_response.replace('</end_of_turn>', '').replace('<end_of_turn>', '')
        # Append the model's response (first question) to the chat history
        chat_history.append(MODEL_CHAT_TEMPLATE.format(prompt=full_response))

    full_response = full_response.replace('</end_of_turn>', '').replace('<end_of_turn>', '')

    # Convert the string to a dictionary
    response_dict = json.loads(full_response)

    return {"response": response_dict}


@app.post("/ai_interview/stt_indian_only")
def ai_interview_stt(voice_path: str = Form(...)):
    """
        AI Interview Model For Speech to Text Model STT
    """
    import live_audio_model as lam
    results_transcript = lam.call_center_stt_model(audio_url=voice_path,detected_language='hi')
    return results_transcript


# Test Live audio
# audio_url = 'https://cloudphone.tatateleservices.com/file/recording?callId=1724078757.94332&type=rec&token=N1VwdHVOSGVwSXBaVUlzSTBjQnQxZFVnOUV2S0RFSlkxVGUzbkYzdDI5c3c0UUxwNDZsT2pSNWN2ajdlU0pMcjo6YWIxMjM0Y2Q1NnJ0eXl1dQ%3D%3D'
# ai_interview_stt(audio_url)



