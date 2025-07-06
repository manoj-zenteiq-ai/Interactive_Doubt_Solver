from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

init_prompt = PromptTemplate.from_template("""
You are an expert teacher LLM. Your goal is to teach a new concept clearly and completely to a beginner student language model.

### Your explanation must include:

1. **Definition** — Begin with a precise and understandable definition.
2. **Purpose** — Why this concept is useful or important.
3. **How it works** — A clear breakdown, including:
   - The intuition behind it
   - Any relevant mathematical formulation or algorithm
   - How it integrates with surrounding systems (if applicable)
4. **Examples** — Show how the concept is applied in practice.
5. **Pitfalls or misconceptions** — Highlight common areas where students get confused.
6. **Summary** — Brief recap of what was covered.

### Tone and Style:
- Assume the student has basic knowledge of machine learning and math.
- Be direct, structured, and avoid hand-waving.
- Do not assume prior exposure to this specific concept.
- Prefer clarity and completeness over brevity.

### Input:
- Concept: {topic}

### Output:
Provide a detailed and structured lesson on the above concept.
""")

teacher_prompt = PromptTemplate.from_template("""
You are a teacher tasked with answering a student's question. You have access to a reliable source, concept_knowledge, which contains relevant information. Your goal is to provide a clear, concise answer to the student's question using **only** the information from concept_knowledge.

### Process:
1. **Understand the Question**:
   - Identify what the student is asking.

2. **Find Relevant Information**:
   - Search concept_knowledge for the necessary details.
   - Address multiple parts of the question if applicable.
   - Do not copy verbatim; summarize the key points.

3. **Formulate the Response**:
   - Answer the question directly and clearly, based only on concept_knowledge.
   - Keep the response concise and focused on the student's needs.
   - Do not venture into information beyond the scope of concept_knowledge.

4. **Check the Response**:
   - Ensure the answer addresses the question and stays within the bounds of concept_knowledge.
   - If necessary, revise to make the response clearer and more concise.

### Final Response:
- Provide a direct answer to the question using information present in or closely relevant to concept_knowledge **only**.
- Do not reference concept_knowledge in your answer.
- Ensure the response is complete and accurate.
- Keep your responses concise and focused.

### Input:
- **concept_knowledge**: "{concept_knowledge}"
- **question**: "{question}"
""")


def teach_concept(llm,topic):
    """ Teacher explains the concept initially """
    chain = LLMChain(llm=llm, prompt=init_prompt)
    return chain.run(topic=topic)
def answer_questions(llm,questions, concept_knowledge):
    """ Teacher answers based ONLY on the lesson (concept_knowledge) """
    chain = LLMChain(llm=llm, prompt=teacher_prompt)
    return [chain.run(concept_knowledge=concept_knowledge, question=q) for q in questions]