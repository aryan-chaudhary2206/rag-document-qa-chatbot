import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")
st.title("📄 RAG Document Q&A Chatbot")
st.caption("Ask questions about your PDF and get answers grounded in its contents.")

with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.markdown("---")
    st.header("2. LLM setup")
    st.caption("Add OPENAI_API_KEY to your .env file before asking questions.")
    st.caption("Your PDF is processed in this app session; the vector index is kept in memory.")

def extract_pdf_documents(file_bytes: bytes, filename: str):
    reader = PdfReader(BytesIO(file_bytes))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_number},
                )
            )
    return docs, len(reader.pages)

@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def make_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

def answer_question(question, vector_store, api_key):
    matches = vector_store.similarity_search_with_score(question, k=4)
    if not matches:
        return "I couldn't find relevant information in the uploaded PDF.", []

    context_parts = []
    sources = []
    for doc, score in matches:
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, page {page}]\n{doc.page_content}"
        )
        sources.append({"source": source, "page": page, "score": float(score)})

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You answer questions about documents. Use only the supplied context. "
         "If the context does not contain the answer, say that the uploaded PDF "
         "does not provide enough information. Do not invent facts. When useful, "
         "refer to the source filename and page number."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": "\n\n---\n\n".join(context_parts),
        "question": question,
    })
    return response.content, sources

if uploaded_file:
    try:
        file_bytes = uploaded_file.getvalue()
        docs, page_count = extract_pdf_documents(file_bytes, uploaded_file.name)
        if not docs:
            st.error("No selectable text was found. This may be a scanned PDF; OCR would be needed.")
            st.stop()

        current_signature = f"{uploaded_file.name}:{len(file_bytes)}"
        if st.session_state.get("document_signature") != current_signature:
            with st.spinner("Reading PDF, splitting it into chunks, and building the search index..."):
                chunks = make_chunks(docs)
                embeddings = get_embedding_model()
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.document_signature = current_signature
                st.session_state.document_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)

        st.success(
            f"Ready: **{uploaded_file.name}** · {page_count} pages · "
            f"{st.session_state.chunk_count} text chunks"
        )
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()
else:
    st.info("Upload a text-based PDF from the sidebar to get started.")
    st.markdown("### How it works")
    st.markdown(
        "1. Extract text from the PDF.  \n"
        "2. Split text into overlapping chunks.  \n"
        "3. Convert chunks into embeddings and index them in FAISS.  \n"
        "4. Retrieve relevant chunks for each question.  \n"
        "5. Ask the LLM to answer using the retrieved context."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved source pages"):
                    for source in message["sources"]:
                        st.write(f"- {source['source']} — page {source['page']}")

    question = st.chat_input("Ask a question about your PDF...")
    if question:
        api_key = os.getenv("OPENAI_API_KEY")
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not api_key:
                answer = (
                    "OpenAI API key not found. Add `OPENAI_API_KEY=your_key_here` "
                    "to a `.env` file in the project folder, then restart Streamlit."
                )
                sources = []
                st.warning(answer)
            else:
                try:
                    with st.spinner("Searching the PDF and generating an answer..."):
                        answer, sources = answer_question(
                            question, st.session_state.vector_store, api_key
                        )
                    st.markdown(answer)
                    if sources:
                        with st.expander("Retrieved source pages"):
                            seen = set()
                            for source in sources:
                                item = (source["source"], source["page"])
                                if item not in seen:
                                    st.write(f"- {item[0]} — page {item[1]}")
                                    seen.add(item)
                except Exception as exc:
                    answer = f"Something went wrong while generating the answer: {exc}"
                    sources = []
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
