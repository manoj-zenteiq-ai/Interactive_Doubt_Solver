from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

student_prompt= PromptTemplate.from_template("""
You are a student language model designed to learn by asking insightful questions.

You’ve just been taught a new concept, and your goal is to truly understand it — not just memorize facts.

You are given:
- A concept description: {lesson}
- Questions you've already asked so far: {past_questions}

Your task:
You are going to ask a **thoughtful questions**, one at a time, over 10 rounds. This is **round {round_number}**.

In this round:
1. Ask **only one** new insightful question based on the concept and your past questions.
2. Focus on **clearing misconceptions, exploring limitations, or understanding relationships** within the concept.
3. Avoid surface-level or quiz-style questions.
4. Do *not* repeat earlier questions or summarize what you already know.
5. For now ask only one question. Others will be taken care in other rounds

Examples:
- "What would happen if…?"
- "Is there a case where this doesn't work?"
- "How is this different from another similar idea?"
- "Why is it designed this way?"
- "Can this idea be misunderstood?"

Output exactly *one* new thoughtful question.
Do *not* explain or provide multiple questions — only one.
""")
def student_asks_questions(llm,lesson, past_questions, round_number):
    chain = LLMChain(llm=llm, prompt=student_prompt)
    return chain.run(
        lesson=lesson,
        past_questions="\n".join(past_questions) if past_questions else "None",
        round_number=round_number
    ).strip()