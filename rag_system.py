import logging
from typing import Dict, Any
from utils import auto_timeit
from llm_client import YandexGPT
from document_processor import DocumentProcessor
from vector_search import VectorSearch

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_search = VectorSearch()
        self.llm = YandexGPT()
        self.chunks = []

    @auto_timeit("rag_load_document")
    def load_and_process_document(self, file_path):
        logger.info(f"Loading document: {file_path}")
        texts = self.document_processor.load_document(file_path)
        logger.info(f"Loaded {len(texts)} text blocks")

        self.chunks = self.document_processor.split_text(texts)
        logger.info(f"Split into {len(self.chunks)} chunks")

        embeddings = self.document_processor.create_embeddings(self.chunks)
        self.vector_search.add_documents(self.chunks, embeddings)

        logger.info("Document processing completed")

    @auto_timeit("rag_ask_question")
    def ask_question(self, question, top_k=3) -> Dict[str, Any]:
        if not self.chunks:
            return {"error": "no doc loaded"}

        logger.info("Searching for relevant chunks")
        relevant_chunks = self.vector_search.search(question, top_k=top_k)

        if not relevant_chunks:
            return {"answer": "No relevant information found in the document."}

        context = "\n".join([
            f"[Chunk {i + 1}]: {chunk['chunk']}"
            for i, chunk in enumerate(relevant_chunks)
        ])

        prompt = f"""Answer strictly based on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY based on the context above
2. If the context lacks information, say "No answer found in document"
3. Be precise and specific
4. Cite the chunk number used

ANSWER:"""

        logger.info("Generating answer with YandexGPT")
        answer = self.llm.generate(prompt)

        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": chunk["chunk"][:500] + "..." if len(chunk["chunk"]) > 500 else chunk["chunk"],
                    "similarity": round(chunk["similarity"], 3),
                    "index": chunk["index"]
                }
                for chunk in relevant_chunks
            ],
            "stats": {
                "total_chunks": len(self.chunks),
                "retrieved_chunks": len(relevant_chunks)
            }
        }

        return result
