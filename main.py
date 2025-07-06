from models.teacher import teach_concept, answer_questions
from models.student import student_asks_questions
import json
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from utils.io_utils import save_conversation

# def save_conversation(topic, lesson, questions, answers):
#     record = {
#         "id": str(uuid4()),
#         "timestamp": str(datetime.now()),
#         "topic": topic,
#         "lesson": lesson,
#         "qa_pairs": [{"question": q, "answer": a} for q, a in zip(questions, answers)]
#     }
#     with open("data/data_logs.json", "w") as f:
#         json.dump(record, f, indent=2)
#     return "data/data_logs.json"
# ========== RUNNING THE PIPELINE ==========


llama_teacher = Ollama(model="llama3:8b", temperature=0.7)
llama_student = Ollama(model="llama3:8b", temperature=0.7)

with open("data/concept_data.json") as f:
    concepts = json.load(f)

for topic in concepts:

    # Step 1: Teacher explains the topic
    lesson = teach_concept(llama_teacher,topic)
    print("\n=== LESSON ===\n", lesson)

    # Step 2: Student asks 10 reflective questions
    past_questions = []
    for i in range(10):
        q = student_asks_questions(llama_student,lesson, past_questions,i+1)
        past_questions.append(q)
        print(f"\nStudent Question {i+1}: {q}")

    # Step 3: Teacher answers strictly from concept_knowledge
    answers = answer_questions(llama_teacher,past_questions, concept_knowledge=lesson)
    # all_answers=[]
    for i, ans in enumerate(answers):
        print(f"\nTeacher Answer {i+1}: {ans}")
        # all_answers.append(ans)

    # Step 4: Save conversation
    filepath = save_conversation(topic, lesson, past_questions, answers)

# # Step 5: Student gets tested
# test_q = "What advantage does RoPE have over fixed sinusoidal encodings?"
# response = test_student(topic, test_q, filepath)
# print("\n=== TEST RESPONSE ===\n", response)