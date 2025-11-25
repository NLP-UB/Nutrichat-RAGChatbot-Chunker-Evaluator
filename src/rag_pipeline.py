import os
import glob
from .loader import load_pdf, chunk_text
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator
from .ner_processor import NERProcessor

class RAGPipeline:
    def __init__(self, data_path="data", embed_model='embeddinggemma', gen_model='gpt-oss',
                 storage_path="./qdrant_storage", collection_name="semantic"):
        """
        RAG Pipeline that uses Qdrant persistent storage for embeddings.

        Args:
            data_path (str): Directory containing PDF files for initial indexing
            embed_model (str): Sentence Transformer model for embeddings
            gen_model (str): Text generation model
            storage_path (str): Local folder for persistent Qdrant storage
            collection_name (str): Name of Qdrant collection
        """
        self.embedder = Embedder(embed_model)
        self.generator = Generator(gen_model)
        self.vector_store = VectorStore(768, storage_path=storage_path, collection_name=collection_name)
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.collection_name = collection_name

        # Perform indexing only once at initialization (if data_path provided)
        if data_path and self._is_vector_store_empty():
            self._index_all_pdfs(data_path)
        else:
            print("Using existing Qdrant vector store, skipping indexing.")

    def _is_vector_store_empty(self):
        """Check if the vector store collection already contains points."""
        try:
            count = self.vector_store.client.count(collection_name=self.collection_name).count
            return count == 0
        except Exception:
            return True

    def _index_all_pdfs(self, data_path):
        """
        Load all PDF documents in the directory (including subfolders),
        split into chunks, embed, and store in Qdrant.
        """
        pdf_files = glob.glob(os.path.join(data_path, "**", "*.pdf"), recursive=True)

        if not pdf_files:
            print(f"No PDF files found in directory: {data_path}")
            return

        # 🔹 Kosongkan dulu koleksi biar tidak duplikat
        try:
            self.vector_store.client.delete(
                collection_name=self.collection_name,
                points_selector={"filter": {}}  # hapus semua point
            )
            print(f"Cleared existing data in collection: {self.collection_name}")
        except Exception as e:
            print(f"Warning: could not clear collection ({e})")

        all_chunks = []
        all_embeddings = []

        print(f"Found {len(pdf_files)} PDF(s). Processing...")
        for file_path in pdf_files:
            try:
                print(f"🔹 Reading: {file_path}")
                text = load_pdf(file_path)
                if not text.strip():
                    print(f"Skipping empty PDF: {file_path}")
                    continue

                chunks = chunk_text(text, method=self.collection_name, chunk_size=1000)

                # chunks = self.ner.process_chunks(chunks)

                embeddings = self.embedder.embed_documents(chunks)

                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if all_chunks:
            self.vector_store.add(all_embeddings, all_chunks)
            print(f"Indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs into persistent Qdrant storage.")
        else:
            print("No valid PDF content to index.")

    def answer_question(self, query, top_k=3):
        """
        Retrieve context and generate an answer using the generator model.
        """
        retrieved = self.retriever.retrieve(query, top_k)
        context = " ".join([r[0] for r in retrieved])
        return self.generator.generate(context, query)
    
    def answer_question_with_context(self, query, top_k=3):
        """
        Retrieve context and generate an answer using the generator model.
        Returns:
            [answer_text, list_of_top_texts]
        """
        # Retrieve top-k documents
        retrieved = self.retriever.retrieve(query, top_k)
        
        # Extract only the text parts for context
        top_texts = [r[0] for r in retrieved]
        
        # Combine all texts to feed into the generator
        context = " ".join(top_texts)
        
        # Generate answer
        answer = self.generator.generate(context, query)
        
        # Return answer and top retrieved texts as a 2-element list
        return [answer, top_texts]