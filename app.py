import os
import gc
import time
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "chroma_db"

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="contexto.io", page_icon="✨", layout="wide")
st.markdown(
    """
    <h1 style='margin-bottom: 0px;'>
        📄✨ Contexto 
        <span style='font-size: 20px; font-weight: normal; color: gray; margin-left: 10px;'>
             PDF Intelligence Chatbot
        </span>
    </h1>
    """, 
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-latest")


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a RAG assistant.

Rules:

Use ONLY the provided context as the source of truth.
Never use outside knowledge.
Never invent facts.
If the user asks:
Explain
Summarize
Simplify
Give key points
Explain in easy words
Compare topics
Then perform the task using ONLY the information found in the context.
You may rephrase information into simpler language.
You may generate examples ONLY if they are directly supported by the context.
If the user asks for:
Exact paragraph
Exact text
Exact definition
Exact wording
Then return the relevant passage exactly as it appears in the context.
Do not paraphrase.
If the answer is partially available in the context:
Answer only the available part.
Clearly mention what information was not found.
If the question appears to contain spelling mistakes, abbreviations, or unclear wording:
First identify the most likely intended topic from the context.
Ask a clarification question.
Example:
"Did you mean Deep Reinforcement Learning?"
Wait for confirmation before answering.
If multiple possible topics match:
Ask the user which topic they mean.

If no relevant information exists in the context:
Respond exactly with:

"I could not find this information in the retrieved document context."

When possible:
Quote relevant lines from the context.
Then provide the explanation.
Be concise, factual, and context-grounded.
""",
    ),
    ("human", """
    Context :{context}

    Question :{question}

     """),
])


def safe_rmtree(path, retries=5, delay=0.3):
    """Retry deletion a few times to dodge transient Windows file locks."""
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(delay)
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    return False


def teardown_vectorstore():
    """
    Properly close/release the current Chroma vectorstore before touching
    the persist directory on disk. On Windows, Chroma keeps file handles
    (sqlite + hnswlib index files) open as long as the client object is
    alive, so deleting the folder while it's still referenced causes
    PermissionError. Calling delete_collection() + dropping references +
    gc.collect() releases those handles first.
    """
    if st.session_state.get("vectorstore") is not None:
        try:
            st.session_state.vectorstore.delete_collection()
        except Exception:
            pass

    st.session_state.vectorstore = None
    st.session_state.retriever = None

    # Force garbage collection so sqlite connections / mmapped index
    # files held by the now-unreferenced Chroma client are actually closed.
    gc.collect()

    safe_rmtree(PERSIST_DIR)


# ---------------------------------------------------------
# Build / Replace vector DB from uploaded PDF
# ---------------------------------------------------------
def build_vectorstore_from_pdf(uploaded_file):
    # Close/clear any existing vectorstore + persisted DB first
    teardown_vectorstore()

    # Save uploaded file to a temp path so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # 1. Load
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        # 3. Build fresh vectorstore
        embedding_model = get_embedding_model()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        return vectorstore, retriever, len(docs), len(chunks)

    finally:
        try:
            os.remove(tmp_path)
        except PermissionError:
            pass


# ---------------------------------------------------------
# Sidebar — Upload PDF
# ---------------------------------------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Process only if it's a new file (avoid reprocessing on every rerun)
        if st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Reading PDF, creating chunks, building vector DB..."):
                vectorstore, retriever, n_pages, n_chunks = build_vectorstore_from_pdf(uploaded_file)

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.vectorstore_ready = True
            st.session_state.processed_filename = uploaded_file.name
            st.session_state.messages = []  # reset chat for new document

            st.success(f"✅ '{uploaded_file.name}' processed!")
            st.caption(f"Pages: {n_pages} | Chunks: {n_chunks}")
        else:
            st.info(f"Using: {uploaded_file.name}")

    if st.session_state.vectorstore_ready:
        if st.button("🗑️ Clear document & chat"):
            teardown_vectorstore()
            st.session_state.vectorstore_ready = False
            st.session_state.processed_filename = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Upload a PDF to start chatting with it. Uploading a new PDF replaces the previous document.")


# ---------------------------------------------------------
# Main chat area
# ---------------------------------------------------------
if not st.session_state.vectorstore_ready:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask something about your PDF...")

    if user_query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(user_query.strip().lower())
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = prompt_template.invoke({
                    "context": context,
                    "question": user_query,
                })

                llm = get_llm()
                response = llm.invoke(prompt)

            st.markdown(response.content)

        st.session_state.messages.append({"role": "assistant", "content": response.content})