import os
import glob
import sys
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.loader import Loader

class Indexer:
    def __init__(self, embedder_name, method_name, data_path="data", storage_path="./qdrant_storage", collection_name="semantic"):
        self.embedder = Embedder(embedder_name)
        self.method_name = method_name
        self.loader = Loader(embedder=self.embedder, method_name=method_name)
        self.data_path = data_path
        self.storage_path = storage_path
        self.collection_name = collection_name
        dimension = self.embedder.get_dimension()
        self.vector_store = VectorStore(dimension, storage_path=storage_path, collection_name=collection_name)


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
                text = self.loader.load_pdf(file_path)
                if not text.strip():
                    print(f"Skipping empty PDF: {file_path}")
                    continue

                chunks = self.loader.chunk_text(text, chunk_size=500)

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
    
    def index(self):
        if self.data_path and self._is_vector_store_empty():
            self._index_all_pdfs(self.data_path)
        else:
            print("Using existing Qdrant vector store, skipping indexing.")

if __name__ == "__main__":
    method_name = sys.argv[1]
    embedder_name = sys.argv[2]
    collection_name_format = f"{method_name}_{embedder_name}"
    indexer = Indexer(embedder_name=embedder_name, method_name=method_name, collection_name=collection_name_format)
    # Perform indexing only once at initialization (if data_path provided)
    indexer.index()