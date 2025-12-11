import os
import glob
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator
from index_data import Indexer

class RAGPipeline:
    def __init__(self, embedder: Embedder, indexer: Indexer, format, gen_model='gpt-oss'):
        """
        RAG Pipeline that uses Qdrant persistent storage for embeddings.

        Args:
            data_path (str): Directory containing PDF files for initial indexing
            embed_model (str): Sentence Transformer model for embeddings
            gen_model (str): Text generation model
            storage_path (str): Local folder for persistent Qdrant storage
            collection_name (str): Name of Qdrant collection
        """
        self.indexer = indexer
        self.generator = Generator(model_name=gen_model, format=format)
        self.embedder = embedder
        self.retriever = Retriever(indexer.vector_store, self.embedder)

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