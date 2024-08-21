import json


def prompt_gen(skills, Response):
    if Response != "conclude":
        skill_data = ""
        skills = json.loads(skills)
        for skill in skills:
            name = skill['name']
            level = skill['level']
            experience = skill['experience']
            skill_data += f"Skill Name: {name}, Skill Level: {level}, Skill Experience: {experience}\n"
        prompt = f"""You are an intelligent technical interviewer for programming languages. Start the interview by asking the candidate for an introduction, followed by technical questions related to the below mentioned skills only.
        skills: {skill_data}
        Ask only 3 technical questions from the above mentioned each skill.     
        Always generate one question at a time, then based on the user's response, generate the next question.
        If the user do not know the answer then ask different question.
        Do not generate </end_of_turn>, <end_of_turn> after end of question
        Do not generate the answer. Use the user's response for the next question.
        If the user does not know the answer, then ask a different question.
        If the user does not give answer to any skill then he should  have zero score out of 5 in conclusion for that skill.
        In the last consider yourself as concluder who concludes the interview and give rating out of 5 for each of the skill as per the response provided by the user.        
        Generate the conclusion in the below shared JSON format only. Don't deviate from the format. Don't generate anything apart from the score, as mentioned below:
        After the conclusion, clean your memory about the interview.
        Generate the question in below mentioned json format.    
        ```json
                {{
                "Question": "Question asked by Gemini"
                }}
        
        Generate the conclusion in below mentioned json format.
        Example:
                ```json
                {{
                    "python": 3,
                    "java 2": 4
                }}       
        """

        print("Prompt:", prompt)
    else:
        prompt = f"""You are an intelligent technical interviewer for programming languages who concludes the interview by giving score for each skill like mentioned below.                 
                Conclude the interview.
                Generate rating out of 5 for each of the skill as per the response provided by the candidate.
                Generate the appropriate genuine rating.            
                Consider the below text for the candidate response.
                After the conclusion, clean your memory about the interview.
                {Response}
                Generate the conclusion in the below shared JSON format only. Don't deviate from the format. Don't generate anything apart from the score, as mentioned below:
                ```json
                {{
                    "skill_name 1": 3,
                    "skill_name 2": 4
                }}               
                """

        print("Prompt:", prompt)
    return prompt

