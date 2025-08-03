import os
from .loader import load_pdf, chunk_text
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    def __init__(self, embed_model='all-MiniLM-L6-v2', gen_model='google/flan-t5-base',
                 storage_path="./qdrant_storage", collection_name="rag_collection"):
        """
        RAG Pipeline that uses Qdrant persistent storage for embeddings.

        Args:
            embed_model (str): Sentence Transformer model for embeddings
            gen_model (str): Text generation model
            storage_path (str): Local folder for persistent Qdrant storage
            collection_name (str): Name of Qdrant collection
        """
        self.embedder = Embedder(embed_model)
        self.generator = Generator(gen_model)
        self.vector_store = VectorStore(384, storage_path=storage_path, collection_name=collection_name)
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.collection_name = collection_name

    def _is_vector_store_empty(self):
        """Check if the vector store collection already contains points."""
        count = self.vector_store.client.count(collection_name=self.collection_name).count
        return count == 0

    def index_document(self, file_path):
        """
        Load a PDF document, split into chunks, embed, and store in Qdrant.
        If data already exists in vector store, skip indexing.
        """
        if not self._is_vector_store_empty():
            print("Vector store already contains data. Skipping indexing.")
            return

        text = load_pdf(file_path)
        chunks = chunk_text(text)
        embeddings = self.embedder.embed(chunks)
        self.vector_store.add(embeddings, chunks)
        print(f"Indexed {len(chunks)} chunks from {file_path} into persistent Qdrant storage.")

    def answer_question(self, query, top_k=3):
        """
        Retrieve context and generate an answer using the generator model.
        """
        retrieved = self.retriever.retrieve(query, top_k)
        context = " ".join([r[0] for r in retrieved])
        return self.generator.generate(context, query)
