import os
import re

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from youtube_transcript_api.proxies import WebshareProxyConfig

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid YouTube RAG Bot",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎬 Production-Grade Hybrid YouTube Chatbot")
st.markdown("---")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_video_id(url):
    """
    Extract the 11-character YouTube video ID
    from common YouTube URL formats.
    """

    if not url:
        return None

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"

    match = re.search(pattern, url)

    return match.group(1) if match else None


def reciprocal_rank_fusion(
    vector_docs,
    keyword_docs,
    k_total=4,
    c=60,
):
    """
    Combine dense vector retrieval and sparse BM25
    retrieval using Reciprocal Rank Fusion (RRF).
    """

    rrf_scores = {}

    # --------------------------------------------------------
    # Dense vector search results
    # --------------------------------------------------------

    for rank, doc in enumerate(vector_docs):

        content = doc.page_content

        if content not in rrf_scores:

            rrf_scores[content] = {
                "doc": doc,
                "score": 0.0,
            }

        rrf_scores[content]["score"] += (
            1.0 / (rank + 1 + c)
        )

    # --------------------------------------------------------
    # Sparse BM25 search results
    # --------------------------------------------------------

    for rank, doc in enumerate(keyword_docs):

        content = doc.page_content

        if content not in rrf_scores:

            rrf_scores[content] = {
                "doc": doc,
                "score": 0.0,
            }

        rrf_scores[content]["score"] += (
            1.0 / (rank + 1 + c)
        )

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    sorted_docs = sorted(
        rrf_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return [
        item["doc"]
        for item in sorted_docs[:k_total]
    ]


def create_youtube_api():
    """
    Create the YouTube Transcript API client.

    If Webshare credentials are configured in Streamlit
    Secrets, requests are routed through the configured
    proxy.

    Otherwise, the app uses direct requests.
    """

    proxy_username = os.getenv(
        "WEBSHARE_PROXY_USERNAME"
    )

    proxy_password = os.getenv(
        "WEBSHARE_PROXY_PASSWORD"
    )

    if proxy_username and proxy_password:

        proxy_config = WebshareProxyConfig(
            proxy_username=proxy_username,
            proxy_password=proxy_password,
        )

        return YouTubeTranscriptApi(
            proxy_config=proxy_config
        )

    return YouTubeTranscriptApi()


def get_transcript(video_id):
    """
    Fetch the English transcript for a YouTube video.
    """

    youtube_api = create_youtube_api()

    transcript_list = youtube_api.fetch(
        video_id,
        languages=["en"],
    )

    full_transcript = " ".join(
        chunk.text
        for chunk in transcript_list
    )

    return full_transcript


# ============================================================
# SESSION STATE
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Configuration Panel")

video_url = st.sidebar.text_input(
    "Paste YouTube Video URL Here:",
    value=st.session_state.current_video,
)

process_button = st.sidebar.button(
    "Build Chatbot Engine",
    type="primary",
)


# ============================================================
# RESET STATE WHEN VIDEO CHANGES
# ============================================================

if video_url != st.session_state.current_video:

    st.session_state.engine_ready = False

    st.session_state.vector_retriever = None

    st.session_state.keyword_retriever = None

    st.session_state.messages = []

    st.session_state.current_video = video_url


# ============================================================
# BUILD RAG ENGINE
# ============================================================

if process_button and video_url:

    video_id = extract_video_id(video_url)

    if not video_id:

        st.sidebar.error(
            "❌ Invalid YouTube URL. "
            "Please enter a valid YouTube video link."
        )

    else:

        with st.sidebar.spinner(
            "Processing video transcript..."
        ):

            try:

                # ------------------------------------------------
                # 1. FETCH TRANSCRIPT
                # ------------------------------------------------

                full_transcript = get_transcript(
                    video_id
                )

                if not full_transcript.strip():

                    st.sidebar.error(
                        "⚠️ Transcript unavailable for this video. "
                        "Please try another YouTube video with captions."
                    )

                    st.session_state.engine_ready = False

                else:

                    # ------------------------------------------------
                    # 2. TEXT CHUNKING
                    # ------------------------------------------------

                    text_splitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                        )
                    )

                    chunks = text_splitter.create_documents(
                        [full_transcript]
                    )

                    # ------------------------------------------------
                    # 3. HUGGING FACE EMBEDDINGS
                    # ------------------------------------------------

                    embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )

                    # ------------------------------------------------
                    # 4. FAISS VECTOR STORE
                    # ------------------------------------------------

                    vector_store = FAISS.from_documents(
                        chunks,
                        embeddings,
                    )

                    st.session_state.vector_retriever = (
                        vector_store.as_retriever(
                            search_kwargs={"k": 5}
                        )
                    )

                    # ------------------------------------------------
                    # 5. BM25 RETRIEVER
                    # ------------------------------------------------

                    st.session_state.keyword_retriever = (
                        BM25Retriever.from_documents(
                            chunks
                        )
                    )

                    st.session_state.keyword_retriever.k = 5

                    # ------------------------------------------------
                    # 6. SAVE ENGINE STATE
                    # ------------------------------------------------

                    st.session_state.engine_ready = True

                    st.session_state.messages = []

                    st.sidebar.success(
                        f"✅ Matrix loaded! "
                        f"{len(chunks)} chunks cached."
                    )

            # --------------------------------------------------------
            # TRANSCRIPT ERRORS
            # --------------------------------------------------------

            except TranscriptsDisabled:

                st.sidebar.error(
                    "⚠️ Transcript unavailable for this video. "
                    "Please try another YouTube video with captions."
                )

                st.session_state.engine_ready = False

            except NoTranscriptFound:

                st.sidebar.error(
                    "⚠️ No English transcript is available "
                    "for this video. Please try another video."
                )

                st.session_state.engine_ready = False

            # --------------------------------------------------------
            # OTHER ERRORS
            # --------------------------------------------------------

            except Exception as e:

                error_name = type(e).__name__

                if error_name in (
                    "RequestBlocked",
                    "IpBlocked",
                ):

                    st.sidebar.error(
                        "⚠️ YouTube is currently blocking transcript "
                        "requests from this server. Please try again "
                        "later or use another video."
                    )

                else:

                    st.sidebar.error(
                        "⚠️ We couldn't process the transcript "
                        "for this video. Please try another video."
                    )

                st.session_state.engine_ready = False


# ============================================================
# CHAT INTERFACE
# ============================================================

if st.session_state.engine_ready:

    st.subheader("💭 Conversation Terminal")

    # ------------------------------------------------------------
    # DISPLAY PREVIOUS CHAT HISTORY
    # ------------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # ------------------------------------------------------------
    # USER QUESTION
    # ------------------------------------------------------------

    if question := st.chat_input(
        "Ask something about the video..."
    ):

        # --------------------------------------------------------
        # DISPLAY USER MESSAGE
        # --------------------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # --------------------------------------------------------
        # RAG RETRIEVAL
        # --------------------------------------------------------

        with st.spinner(
            "🔎 Retrieving relevant information..."
        ):

            vector_candidates = (
                st.session_state.vector_retriever.invoke(
                    question
                )
            )

            keyword_candidates = (
                st.session_state.keyword_retriever.invoke(
                    question
                )
            )

            # ----------------------------------------------------
            # RRF HYBRID FUSION
            # ----------------------------------------------------

            fused_docs = reciprocal_rank_fusion(
                vector_candidates,
                keyword_candidates,
                k_total=4,
            )

            # ----------------------------------------------------
            # BUILD CONTEXT
            # ----------------------------------------------------

            context_text = "\n\n".join(
                doc.page_content
                for doc in fused_docs
            )

        # --------------------------------------------------------
        # GROQ API KEY
        # --------------------------------------------------------

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not groq_api_key:

            st.error(
                "❌ GROQ_API_KEY is not configured."
            )

            st.stop()

        # --------------------------------------------------------
        # GROQ LLM
        # --------------------------------------------------------

        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            groq_api_key=groq_api_key,
        )

        # --------------------------------------------------------
        # GROUNDED RAG PROMPT
        # --------------------------------------------------------

        prompt_template = """
You are a production-grade generative AI assistant
trained to analyze YouTube video transcripts.

Answer the user's question using ONLY the context
provided below.

If the context does not contain enough information
to answer the question, say exactly:

"Information not available in video context."

Rules:
- Do not use external facts.
- Do not invent information.
- Do not make unsupported assumptions.
- Answer directly and clearly.
- Use the retrieved context as your only source.

Context:
{context}

Question:
{question}

Factual Answer:
"""

        prompt = PromptTemplate(
            input_variables=[
                "context",
                "question",
            ],
            template=prompt_template,
        )

        final_prompt = prompt.invoke(
            {
                "context": context_text,
                "question": question,
            }
        )

        # --------------------------------------------------------
        # STREAMING RESPONSE
        # --------------------------------------------------------

        with st.chat_message("assistant"):

            try:

                response = st.write_stream(
                    llm.stream(final_prompt)
                )

            except Exception:

                st.error(
                    "⚠️ Sorry, I couldn't generate an answer. "
                    "Please try again."
                )

                response = ""

        # --------------------------------------------------------
        # SAVE COMPLETE RESPONSE
        # --------------------------------------------------------

        if response:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )


# ============================================================
# LANDING PAGE
# ============================================================

else:

    st.info(
        "👈 Enter a YouTube URL in the left "
        "configuration sidebar to launch your "
        "interactive RAG session."
    )