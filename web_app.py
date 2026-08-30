import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# Set up clean web page browser layout
st.set_page_config(page_title="Hybrid YouTube RAG Bot", page_icon="🎬", layout="wide")

st.title("🎬 Production-Grade Hybrid YouTube Chatbot")
st.markdown("---")

# Helper function to process Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(vector_docs, keyword_docs, k_total=4, c=60):
    rrf_scores = {}
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        if content not in rrf_scores:
            rrf_scores[content] = {"doc": doc, "score": 0.0}
        rrf_scores[content]["score"] += 1.0 / (rank + 1 + c)
    for rank, doc in enumerate(keyword_docs):
        content = doc.page_content
        if content not in rrf_scores:
            rrf_scores[content] = {"doc": doc, "score": 0.0}
        rrf_scores[content]["score"] += 1.0 / (rank + 1 + c)
    sorted_docs = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:k_total]]

def extract_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# --- INITIALIZE PERSISTENT STORAGE LAYERS ---
# This ensures data survives the top-to-bottom script reruns
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False
if "vector_retriever" not in st.session_state:
    st.session_state.vector_retriever = None
if "keyword_retriever" not in st.session_state:
    st.session_state.keyword_retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video" not in st.session_state:
    st.session_state.current_video = ""

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Configuration Panel")
video_url = st.sidebar.text_input("Paste YouTube Video URL Here:", value=st.session_state.current_video)
process_button = st.sidebar.button("Build Chatbot Engine", type="primary")

# If the user provides a new URL, force a clean state reset
if video_url != st.session_state.current_video:
    st.session_state.engine_ready = False
    st.session_state.messages = []
    st.session_state.current_video = video_url

# Trigger backend RAG engine assembly when clicked
if process_button and video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.sidebar.error("❌ Invalid YouTube URL layout.")
    else:
        with st.sidebar.spinner("Processing video transcript details..."):
            try:
                # 1. Grab transcript text chunks
                yt_api = YouTubeTranscriptApi()
                transcript_list = yt_api.fetch(video_id, languages=["en"])
                full_transcript = " ".join(chunk.text for chunk in transcript_list)

                # 2. Chunk documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.create_documents([full_transcript])

                # 3. Form search indexes
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                # Save data objects directly to session storage
                st.session_state.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                st.session_state.keyword_retriever = BM25Retriever.from_documents(chunks)
                st.session_state.keyword_retriever.k = 5
                
                # Flip state locks
                st.session_state.engine_ready = True
                st.session_state.messages = [] 
                st.sidebar.success(f"✅ Matrix loaded! {len(chunks)} chunks cached.")
            except TranscriptsDisabled:
                st.sidebar.error(
                    "⚠️ Transcript unavailable for this video. "
                    "Please try another YouTube video with captions."
    )
            except Exception as e:
                st.sidebar.error(f"ERROR: {type(e).__name__}: {e}")

# --- MAIN CHAT INTERFACE WINDOW ---
# We look at st.session_state, NOT the volatile button click state
if st.session_state.engine_ready:
    st.subheader("💭 Conversation Terminal")
    
    # Render historical chat logs from memory
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle incoming user questions
    if question := st.chat_input("Ask something about the video..."):
        # Render question instantly
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Run background RAG execution layers
        with st.spinner("AI is retrieving and synthesizing data..."):
            v_cand = st.session_state.vector_retriever.invoke(question)
            k_cand = st.session_state.keyword_retriever.invoke(question)
            
            # Hybrid search merging
            fused_docs = reciprocal_rank_fusion(v_cand, k_cand, k_total=4)
            context_text = "\n\n".join(doc.page_content for doc in fused_docs)

            # Query LLM Engine via Groq
            llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1, groq_api_key=os.getenv("GROQ_API_KEY"))
            
            prompt_template = """
            You are a production-grade generative AI assistant trained to analyze transcripts.
            Answer the question objectively using ONLY the context block provided below.
            If the context does not supply enough clues to generate an answer, say: 'Information not available in video context.'
            
            Context:
            {context}
            
            Question: 
            {question}
            
            Factual Answer:
            """
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            final_prompt = prompt.invoke({"context": context_text, "question": question})
            
            answer = llm.invoke(final_prompt)
            
            # Render answer and append to session memory logs
            with st.chat_message("assistant"):
                st.markdown(answer.content)
            st.session_state.messages.append({"role": "assistant", "content": answer.content})
else:
    # Landing page helper text
    st.info("👈 Enter a YouTube URL in the left configuration sidebar to launch your interactive RAG session context.")