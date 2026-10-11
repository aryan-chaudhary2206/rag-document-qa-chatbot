# RAG Document Q&A Chatbot

A Streamlit application that lets users upload a text-based PDF, ask questions about it, and receive answers grounded in retrieved passages with source-page references.

## Features

- Upload a PDF through a simple Streamlit interface.
- Extract page text with `pypdf`.
- Split text into overlapping chunks.
- Generate semantic embeddings with Sentence Transformers.
- Index chunks in FAISS and retrieve the most relevant passages.
- Generate context-grounded answers using the OpenAI API (`gpt-4o-mini`).
- Show the filenames and page numbers of retrieved passages.
- Ask the assistant to say when the PDF does not contain enough information.

## Tech stack

Python, Streamlit, LangChain, Sentence Transformers, FAISS, pypdf, OpenAI API.

## Requirements

- Python 3.10 or newer (Python 3.11 is a good choice).
- An OpenAI API key with API access enabled.
- Internet access the first time the embedding model is downloaded.

## Run locally

### 1. Create and enter the project folder

```bash
cd rag_document_qa_chatbot
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your own API key:

```text
OPENAI_API_KEY=your_actual_key
```

Never commit `.env` or expose your API key in source code.

### 5. Start the app

```bash
streamlit run app.py
```

Your browser should open the local Streamlit application. Upload a PDF and ask a question.

## How it works

1. **Load:** Extract text from each PDF page.
2. **Chunk:** Split page text into chunks of 800 characters with 120-character overlap.
3. **Embed:** Convert chunks into vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Index:** Store vectors in a FAISS vector index.
5. **Retrieve:** Search for the four chunks most similar to the user's question.
6. **Generate:** Send the question and retrieved context to the LLM.
7. **Cite:** Display the source filename and page numbers for the retrieved passages.

## Current limitations

- Scanned PDFs without selectable text need OCR, which is not included.
- The FAISS index is held in memory for the current app session; it is not a persistent database.
- Source references identify retrieved pages, but do not guarantee that every generated statement is correct. Always verify important answers against the PDF.
- This initial version processes one uploaded PDF at a time.
- An OpenAI API key and API access are required to generate answers; API usage may incur charges.

## Possible improvements

- Support multiple PDFs at once.
- Add OCR for scanned documents.
- Add tests and more detailed source highlighting.
- Persist the FAISS index and support incremental updates.
- Add configurable retrieval settings and answer-quality evaluation.

## Security

Keep secrets in environment variables or a local `.env` file. The `.gitignore` excludes `.env`. If a key is accidentally published, revoke it and create a new one immediately.
