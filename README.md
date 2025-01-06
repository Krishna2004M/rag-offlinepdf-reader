📑 RAG with Cross-Encoders Re-ranking Demo Application

# RAG PDF Question Answer with Ollama

This is a Streamlit-based application for querying PDF documents using Retrieval-Augmented Generation (RAG). The app processes PDF files, embeds their content into a vector store (using ChromaDB), and uses Ollama's **Llama 2.3, 3B** model to answer user queries based on the document content.

The app also uses **nomic-embed-text:latest** for embedding the text chunks into a vector space.

---

## Features

- **Upload and process PDF documents**: Split content into manageable chunks for semantic search.
- **Vector storage**: Store document embeddings using ChromaDB.
- **AI-powered answers**: Use Ollama's **Llama 2.3, 3B** model for generating answers.
- **Result enhancement**: Improve relevance using cross-encoder re-ranking.

---

## Requirements

- **Python 3.10** or later.
- **Ollama** installed locally ([Ollama installation guide](https://ollama.ai/)).
- **Pulled Ollama models**:
  - **Llama 2.3, 3B** for answering questions:
    ```bash
    ollama pull llama2.3:3b
    ```
  - **nomic-embed-text:latest** for embedding text:
    ```bash
    ollama pull nomic-embed-text:latest
    ```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Krishna2004M/rag-offlinepdf-reader
   cd rag-pdf-qa-ollama

Install dependencies:

   pip install -r requirements.txt
   Install and start Ollama:

   Follow the Ollama installation guide.
   Pull the required models:
        ollama pull llama2.3:3b
        ollama pull nomic-embed-text:latest
    Start the Ollama server:
        ollama serve

   Run the Streamlit app:
        streamlit run app.py

Usage

    1.Upload a PDF file: Use the file uploader in the Streamlit sidebar.
    2.Process the document: Click the "Process" button to split and store the document content in the vector store.
    3.Ask a question: Enter a question related to the document in the text input field and click "Ask".

View results:
    The app retrieves relevant content, re-ranks it, and generates an answer using Ollama's Llama 2.3, 3B model.

Troubleshooting
    Ollama server not running: Ensure you have started the Ollama server using ollama serve.

Model not found: Pull the required models using:

    ollama pull llama2.3:3b
    ollama pull nomic-embed-text:latest
    No documents retrieved: Verify the PDF content was correctly processed and added to the vector store.
    Error with embedding function: Ensure nomic-embed-text:latest is installed and properly configured.
