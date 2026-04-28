from langchain_ollama import OllamaEmbeddings

class Embedder:
    def __init__(self, model_name="embeddinggemma"):
        self.model = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")

    def embed_query(self, text: str):
        return self.model.embed_query(text)

    def embed_documents(self, texts: list):
        return self.model.embed_documents(texts)

    def embed_text_with_values(self, text: str):
        emb = self.embed_query(text)
        return {"text": text, "embedding": emb}