import logging

logger = logging.getLogger(__name__)


class VectorSearch:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.embeddings_model = None

    def _ensure_embeddings_model(self):
        if self.embeddings_model is None:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('./model_minilm')

    def add_documents(self, chunks, embeddings=None):
        self.chunks = chunks
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self._ensure_embeddings_model()
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            self.embeddings = self.embeddings_model.encode(chunks)

    def search(self, query, top_k=3):
        if not self.chunks or len(self.embeddings) == 0:
            return []

        self._ensure_embeddings_model()
        query_embedding = self.embeddings_model.encode([query])[0]

        from numpy import dot
        from numpy.linalg import norm

        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = dot(query_embedding, doc_embedding) / (
                    norm(query_embedding) * norm(doc_embedding)
            )
            results.append({
                "chunk": self.chunks[i],
                "similarity": float(similarity),
                "index": i
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
