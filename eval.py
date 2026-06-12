import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("====================================================")
print("     CUSTOM LLM-AS-A-JUDGE EVALUATION HARNESS")
print("====================================================\n")

# 1. Initialize the Evaluator Model
print("Connecting to Groq Judge Engine...")
judge_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,  # 0.0 temperature ensures objective scoring consistency
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define the Mock RAG Run Output to Validate
rag_sample = {
    "question": "what is ai",
    "answer": "AI stands for Artificial Intelligence. It is basically trying to simulate with a computer something that would match or exceed human intelligence.",
    "contexts": [
        "AI stands for Artificial Intelligence. It is basically trying to simulate with a computer something that would match or exceed human intelligence. Intelligence is generally thought of as the ability to learn, to infer, and to reason."
    ]
}

# 3. Create Factual Faithfulness Prompt Framework (Detecting Hallucinations)
faithfulness_template = """
You are an expert AI quality assurance judge assessing RAG system performance.
Your task is to calculate a FAITHFULNESS SCORE between 0.00 and 1.00.

An answer is faithful if ALL claims made within it are directly supported by the context block.
If the answer makes a claim that is NOT mentioned in the context, penalize the score.

Context Block:
{context}

Generated System Answer:
{answer}

Output your response in this exact format:
SCORE: [numerical value between 0.00 and 1.00]
REASON: [one short sentence explaining the score deduction or approval]
"""

# 4. Create Answer Relevancy Prompt Framework (Detecting Rambling/Irrelevancy)
relevancy_template = """
You are an expert AI quality assurance judge assessing RAG system performance.
Your task is to calculate an ANSWER RELEVANCY SCORE between 0.00 and 1.00.

An answer is relevant if it directly addresses the user's explicit question without rambling or adding unrelated fluff.

User Question:
{question}

Generated System Answer:
{answer}

Output your response in this exact format:
SCORE: [numerical value between 0.00 and 1.00]
REASON: [one short sentence explaining the score deduction or approval]
"""

# Compile prompts
faith_prompt = PromptTemplate(input_variables=["context", "answer"], template=faithfulness_template)
rel_prompt = PromptTemplate(input_variables=["question", "answer"], template=relevancy_template)

print("\nRunning Evaluation Scoring Process...")

# Execute Faithfulness Check
context_str = "\n".join(rag_sample["contexts"])
compiled_faith = faith_prompt.invoke({"context": context_str, "answer": rag_sample["answer"]})
faith_response = judge_llm.invoke(compiled_faith).content

# Execute Relevancy Check
compiled_rel = rel_prompt.invoke({"question": rag_sample["question"], "answer": rag_sample["answer"]})
rel_response = judge_llm.invoke(compiled_rel).content

# Extract numerical scores for visual metrics display
def extract_score(text):
    match = re.search(r"SCORE:\s*([0-9.]+)", text)
    return match.group(1) if match else "N/A"

def extract_reason(text):
    match = re.search(r"REASON:\s*(.*)", text)
    return match.group(1) if match else "Processed successfully."

# 5. Output Metric Dashboards
print("\n================ EVALUATION METRICS ================")
print(f"📊 FAITHFULNESS SCORE : {extract_score(faith_response)} / 1.00")
print(f"   ↳ Explanation     : {extract_reason(faith_response)}")
print("-" * 52)
print(f"📊 ANSWER RELEVANCY   : {extract_score(rel_response)} / 1.00")
print(f"   ↳ Explanation     : {extract_reason(rel_response)}")
print("====================================================\n")