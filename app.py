"""
Streamlit UI for the PDF RAG application.

Combines the logic from create_database.py (load PDF -> split -> embed -> store in Chroma)
and main.py (retrieve -> prompt -> LLM answer) into a single interactive app.

Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


PERSIST_DIR = "chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "openai/gpt-oss-120b"

st.set_page_config(page_title="PDF RAG Chat", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm():
    base_llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="conversational",
        max_new_tokens=512,
        temperature=0.7,
    )
    return ChatHuggingFace(llm=base_llm)


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )




def build_vectorstore_from_pdf(pdf_path: str, persist_dir: str = PERSIST_DIR):
    """Load a PDF, split it into chunks, embed, and persist into Chroma."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()

    # Start fresh each time a new book is uploaded so old content doesn't linger
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vectorstore, len(docs), len(chunks)


def load_existing_vectorstore(persist_dir: str = PERSIST_DIR):
    embedding_model = get_embedding_model()
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)


def answer_question(vectorstore, query: str) -> str:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = get_prompt()
    final_prompt = prompt.invoke({"context": context, "question": query})

    llm = get_llm()
    response = llm.invoke(final_prompt)
    return response.content


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None




with st.sidebar:
    st.header("📚 Upload your book")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    build_clicked = st.button("Build knowledge base", type="primary", disabled=uploaded_file is None)

    if build_clicked and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Reading PDF, splitting into chunks, and creating embeddings..."):
            try:
                vectorstore, n_pages, n_chunks = build_vectorstore_from_pdf(tmp_path)
                st.session_state.vectorstore = vectorstore
                st.session_state.book_name = uploaded_file.name
                st.session_state.messages = []  # reset chat for the new book
                st.success(f"Indexed {n_pages} page(s) into {n_chunks} chunk(s).")
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")
            finally:
                os.remove(tmp_path)

    st.divider()

    if st.session_state.vectorstore is None and os.path.exists(PERSIST_DIR):
        if st.button("Load existing knowledge base"):
            with st.spinner("Loading existing Chroma DB..."):
                st.session_state.vectorstore = load_existing_vectorstore()
                st.session_state.book_name = "Previously indexed document"
            st.rerun()

    if st.session_state.book_name:
        st.info(f"Current book: **{st.session_state.book_name}**")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()



st.title("📖 Chat with your PDF")
st.caption("Upload a book in the sidebar, build the knowledge base, then ask questions about it.")

if st.session_state.vectorstore is None:
    st.warning("👈 Upload a PDF and click **Build knowledge base** to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    user_query = st.chat_input("Ask a question about the document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(st.session_state.vectorstore, user_query)
                except Exception as e:
                    answer = f"An error occurred while generating the answer: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
