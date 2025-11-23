import re
import numpy as np
from sentence_transformers import util
import fitz  # PyMuPDF
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from llama_index.core import Document

def load_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(
    text,
    chunk_size=500,
    overlap=50,
    method="character",  # Options: "character", "recursive", "document", "semantic", "doublepass"
    separators=["\n\n", "\n", ".", " ", ""],
):
    """
    Splits text into chunks based on the selected chunking method.

    Parameters:
        text (str): The input text to split
        chunk_size (int): Max characters per chunk (for non-semantic methods)
        overlap (int): Overlap between chunks
        method (str): Chunking method name
                      Options:
                        - "character"  : Simple static character chunks
                        - "recursive"  : Recursive chunking based on separators
                        - "document"   : Heuristics for PDFs, Markdown, or code blocks
                        - "semantic"   : Embedding-based semantic chunking
        separators (list): Separators for recursive splitting
        model_name (str): Model for semantic similarity (only used if method="semantic")

    Returns:
        list: List of text chunks
    """

    chunks = []

    # ------------------------------
    # Method 1: Simple Character Splitting
    # ------------------------------
    if method == "character":
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap

    # ------------------------------
    # Method 2: Recursive Character Text Splitting
    # ------------------------------
    elif method == "recursive":
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
    # Method 3: Document-Specific Splitting
    # ------------------------------
    elif method == "document":
        # Basic heuristic for structured docs
        parts = re.split(r"(?:\n\s*\n|```.*?```)", text)
        temp = ""
        for part in parts:
            if len(temp) + len(part) <= chunk_size:
                temp += part + "\n"
            else:
                chunks.append(temp.strip())
                temp = part
        if temp:
            chunks.append(temp.strip())

    # ------------------------------
    # Method 4: Semantic Splitting (Embedding-based)
    # ------------------------------
    elif method == "semantic":
        from .embedder import Embedder
        embedder = Embedder()
        sentences = re.split(r'(?<=[.!?]) +', text)
        embeddings = embedder.embed_documents(sentences)
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

    elif method == "doublepass":
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
        raise ValueError(f"Invalid chunking method: {method}. Choose from "
                         f"['character', 'recursive', 'document', 'semantic'].")

    chunks = list(dict.fromkeys(chunks))
    return [c.strip() for c in chunks if c.strip()]
