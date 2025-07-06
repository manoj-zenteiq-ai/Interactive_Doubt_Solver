import json
from datetime import datetime
from uuid import uuid4

def save_conversation(topic, lesson, questions, answers):
    record = {
        "id": str(uuid4()),
        "timestamp": str(datetime.now()),
        "topic": topic,
        "lesson": lesson,
        "qa_pairs": [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    }
    with open("data/data_logs.json", "w") as f:
        json.dump(record, f, indent=2)
    return "data/data_logs.json"