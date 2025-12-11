import re
import numpy as np
from sentence_transformers import util
from .embedder import Embedder
import fitz  # PyMuPDF
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from llama_index.core import Document

class Loader:
    def __init__(self, embedder: Embedder, method_name="recursive"):
        self.embedder = embedder
        self.method_name = method_name # Options: "recursive", "semantic", "doublepass"

    def load_pdf(self, file_path):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def chunk_text(
        self,
        text,
        chunk_size=500,
        separators=["\n\n", "\n", ".", " ", ""],
    ):
        """
        Splits text into chunks based on the selected chunking method.

        Parameters:
            text (str): The input text to split
            chunk_size (int): Max characters per chunk (for non-semantic methods)
            method (str): Chunking method name
                          Options:
                            - "recursive"  : Recursive chunking based on separators
                            - "semantic"   : Embedding-based semantic chunking
                            - "doublepass" : New method for chunking documents
            separators (list): Separators for recursive splitting

        Returns:
            list: List of text chunks
        """

        chunks = []

        # ------------------------------
        # Method 1: Recursive Character Text Splitting
        # ------------------------------
        if self.method_name == "recursive":
            def recursive_split(text, separators):
                if not separators:
                    return [text]
                sep = separators[0]
                if sep and sep in text:
                    parts = text.split(sep)
                else:
                    parts = [text]
                result = []
                for part in parts:
                    if len(part) > chunk_size and len(separators) > 1:
                        result.extend(recursive_split(part, separators[1:]))
                    else:
                        result.append(part.strip())
                return result

            all_parts = recursive_split(text, separators)
            temp = ""
            for part in all_parts:
                if len(temp) + len(part) + 1 <= chunk_size:
                    temp += " " + part
                else:
                    chunks.append(temp.strip())
                    temp = part
            if temp:
                chunks.append(temp.strip())

        # ------------------------------
        # Method 2: Semantic Splitting (Embedding-based)
        # ------------------------------
        elif self.method_name == "semantic":
            sentences = re.split(r'(?<=[.!?]) +', text)
            embeddings = self.embedder.embed_documents(sentences)
            embeddings = np.array(embeddings)

            current_chunk = [sentences[0]]
            current_emb = embeddings[0]
            for i in range(1, len(sentences)):
                sim = util.cos_sim(current_emb, embeddings[i])
                if sim < 0.5 or sum(len(s) for s in current_chunk) > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_emb = embeddings[i]
                else:
                    current_chunk.append(sentences[i])
                    current_emb = (current_emb + embeddings[i]) / 2
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        # ------------------------------
        # Method 3: Double Pass Merging Chunking
        # ------------------------------
        elif self.method_name == "doublepass":
            config = LanguageConfig(language="english", spacy_model="en_core_web_md")
            splitter = SemanticDoubleMergingSplitterNodeParser(
                language_config=config,
                initial_threshold=0.4,
                appending_threshold=0.5,
                merging_threshold=0.5,
                max_chunk_size=chunk_size,
            )
            # Bungkus teks jadi Document
            document = Document(text=text)

            # Split jadi nodes
            nodes = splitter.get_nodes_from_documents([document])

            # Ambil konten tiap node ke dalam chunks
            chunks = [node.get_content() for node in nodes]

        else:
            raise ValueError(f"Invalid chunking method: {self.method_name}. Choose from "
                             f"['character', 'recursive', 'document', 'semantic'].")

        chunks = list(dict.fromkeys(chunks))
        return [c.strip() for c in chunks if c.strip()]