# Local RAG PDF QA System

This project implements a local Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using open-source tools and local vector storage (Qdrant).

## Features

- Ingest and chunk PDF documents
- Embed text chunks using Sentence Transformers
- Store and search embeddings with Qdrant (local persistent vector DB)
- Retrieve relevant context and generate answers using a transformer-based LLM

## Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **(Optional) Install torch with CUDA support for GPU acceleration.**

## Usage

Run the main script to query your PDF documents:

```sh
python main.py --docs data/ --query "What are the macronutrients?"
```

- `--docs`: Path to the folder containing your PDF files.
- `--query`: The question you want to ask.

## Project Structure

- `main.py` — Entry point for running the RAG pipeline.
- `src/` — Source code for PDF loading, chunking, embedding, vector storage, retrieval, and generation.
- `data/` — Place your PDF files here.
- `qdrant_storage/` — Local persistent storage for Qdrant vector DB.

## References

- [Qdrant](https://qdrant.tech/)
- [Sentence Transformers](https://www.sbert.net/)
- [Transformers](https://huggingface.co/transformers/)





