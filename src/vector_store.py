from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import uuid

class VectorStore:
    def __init__(self, dimension, collection_name="rag_collection"):
        self.client = QdrantClient(":memory:")  # In-memory mode (no server needed)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection if it doesn't exist
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )

        self.text_store = {}  # Map point_id -> text

    def add(self, embeddings, texts):
        points = []
        for emb, text in zip(embeddings, texts):
            point_id = str(uuid.uuid4())
            points.append(PointStruct(id=point_id, vector=np.array(emb, dtype=np.float32)))
            self.text_store[point_id] = text

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k=3):
        query_vector = np.array(query_embedding, dtype=np.float32)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [(self.text_store[hit.id], hit.score) for hit in results]
