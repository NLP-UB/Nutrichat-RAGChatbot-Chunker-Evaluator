from langchain_community.embeddings import OllamaEmbeddings

class Embedder:
    def __init__(self, model_name="embeddinggemma"):
        self.model = OllamaEmbeddings(model=model_name, base_url="http://localhost:11435")

    def embed_query(self, text: str):
        # langsung return embedding dari query
        return self.model.embed_query(text)

    def embed_documents(self, texts: list):
        # embed_documents sudah disediakan
        return self.model.embed_documents(texts)
