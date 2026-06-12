import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Upgraded production-ready packages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever

# Load environment configurations
load_dotenv()

def extract_video_id(url):
    """
    Extracts the 11-character video ID from various YouTube link layouts.
    """
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def reciprocal_rank_fusion(vector_docs, keyword_docs, k_total=4, c=60):
    """
    Production-level Reciprocal Rank Fusion (RRF) algorithm.
    Fuses two ranked lists (Vector Semantic + Keyword Sparse) mathematically.
    Formula: Score = sum( 1 / (Rank + c) )
    """
    rrf_scores = {}
    
    # Process Dense Vector document ranks
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        if content not in rrf_scores:
            rrf_scores[content] = {"doc": doc, "score": 0.0}
        rrf_scores[content]["score"] += 1.0 / (rank + 1 + c)
        
    # Process Sparse Keyword document ranks
    for rank, doc in enumerate(keyword_docs):
        content = doc.page_content
        if content not in rrf_scores:
            rrf_scores[content] = {"doc": doc, "score": 0.0}
        rrf_scores[content]["score"] += 1.0 / (rank + 1 + c)
        
    # Sort the dictionary items by their accumulated RRF score descending
    sorted_docs = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    
    # Extract and return the top matching document structures
    return [item["doc"] for item in sorted_docs[:k_total]]


# Initialize Application
print("====================================================")
print("      HYBRID YOUTUBE RAG ENGINE")
print("====================================================\n")

video_url = input("Enter YouTube Video URL: ")
video_id = extract_video_id(video_url)

if not video_id:
    print("❌ Invalid YouTube URL provided. Shutting down...")
    exit()

print(f"\nTarget video confirmed! Processing ID: {video_id}")
print("1. Fetching raw transcripts from YouTube API...")

try:
    # Fetch transcript tokens
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id, languages=["en"])
    full_transcript = " ".join(chunk.text for chunk in transcript_list)

    # 2. Text Splitting & Parsing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([full_transcript])
    print(f"✅ Text segment parsing complete. Created: {len(chunks)} document chunks.")

    # 3. Setting Up Parallel Search Engines (Hybrid Mapping)
    print("\n2. Launching Core Indexing Engines...")
    
    # Dense Retriever Setup
    print("   -> Running local Hugging Face Embeddings Engine (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Sparse Retriever Setup
    print("   -> Constructing statistical BM25 Keyword Search indices...")
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5
    
    print("✅ Hybrid Indexing Matrix successfully constructed!")

    # 4. Initialize Modern Cloud LLM Core
    print("\n3. Securing Pipeline Connection to Groq Cloud API...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.1,  # Low temperature forces strict factual generation
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Context ground-locking prompt to eliminate systemic hallucinations
    prompt_template = """
    You are a production-grade generative AI assistant trained to analyze transcripts.
    Answer the question objectively using ONLY the context block provided below.
    If the context does not supply enough clues to generate an answer, say: 'Information not available in video context.'
    Do not infer answers using external facts.

    Context:
    {context}

    Question: 
    {question}

    Factual Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # 5. Interactive Chat Runtime Terminal
    print("\n🚀 DEPLOYMENT COMPLETED: Hybrid RAG System is online!")
    print("----------------------------------------------------------------")
    print("Type your questions below. Type 'exit' or 'quit' to terminate.")
    print("----------------------------------------------------------------")

    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit']:
            print("\nShutting down session context. Goodbye!")
            break
        if not question.strip():
            continue

        print("AI is retrieving and synthesizing data...")

        # Invoke individual retrievers to populate raw candidate lists
        vector_candidates = vector_retriever.invoke(question)
        keyword_candidates = keyword_retriever.invoke(question)
        
        # Fuse scores using Reciprocal Rank Fusion
        fused_context_docs = reciprocal_rank_fusion(vector_candidates, keyword_candidates, k_total=4)
        
        # Structure context elements into the LLM payload
        context_text = "\n\n".join(doc.page_content for doc in fused_context_docs)
        final_prompt = prompt.invoke({"context": context_text, "question": question})
        
        # Execute Generation
        answer = llm.invoke(final_prompt)
        
        print(f"\nAI: {answer.content}")
        print("-" * 40)

except TranscriptsDisabled:
    print("\n❌ Error: Closed captions/transcripts are locked or unavailable for this video link.")
except Exception as e:
    print(f"\n❌ A runtime exception occurred: {e}")