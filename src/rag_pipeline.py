from .loader import load_pdf, chunk_text
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator

class RAGPipeline:
    def __init__(self, embed_model='all-MiniLM-L6-v2', gen_model='google/flan-t5-base'):
        self.embedder = Embedder(embed_model)
        self.generator = Generator(gen_model)
        self.vector_store = VectorStore(384)  # Dimension for MiniLM-L6-v2
        self.retriever = Retriever(self.vector_store, self.embedder)

    def index_document(self, file_path):
        text = load_pdf(file_path)
        chunks = chunk_text(text)
        embeddings = self.embedder.embed(chunks)
        self.vector_store.add(embeddings, chunks)

    def answer_question(self, query, top_k=3):
        retrieved = self.retriever.retrieve(query, top_k)
        context = " ".join([r[0] for r in retrieved])
        return self.generator.generate(context, query)
