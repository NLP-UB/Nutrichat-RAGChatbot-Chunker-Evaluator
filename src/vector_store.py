from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import uuid
import os

class VectorStore:
    def __init__(self, dimension, collection_name="test", storage_path="./qdrant_storage",
                 host="localhost", port=6333, prefer_server=True):
        """
        VectorStore using Qdrant. 
        - If prefer_server=True and Qdrant server is running, connect to it.
        - Otherwise, fall back to embedded mode with local RocksDB storage.
        """

        self.collection_name = collection_name
        self.dimension = dimension

        if prefer_server:
            try:
                self.client = QdrantClient(host=host, port=port, timeout=120.0)
                self.client.get_collections()
                print(f"Connected to Qdrant server at {host}:{port}")
            except Exception as e:
                print(f"Could not connect to Qdrant server, falling back to local mode. Reason: {e}")
                os.makedirs(storage_path, exist_ok=True)
                self.client = QdrantClient(path=storage_path)
        else:
            os.makedirs(storage_path, exist_ok=True)
            self.client = QdrantClient(path=storage_path)

        collections = [col.name for col in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )

    def add(self, embeddings, texts, batch_size=256):
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=np.array(emb, dtype=np.float32),
                    payload={"text": txt}
                )
                for emb, txt in zip(batch_embeddings, batch_texts)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

    def search(self, query_embedding, top_k=3):
        query_vector = np.array(query_embedding, dtype=np.float32)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [(hit.payload.get("text", "Text not found"), hit.score) for hit in results]