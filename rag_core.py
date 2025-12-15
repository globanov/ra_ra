import os
import requests
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from raft_utils import auto_timeit

load_dotenv()

logger = logging.getLogger(__name__)


class YandexGPT:

    def __init__(self, api_key: str = None, folder_id: str = None):
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        if not self.api_key or not self.folder_id:
            logger.warning("YandexGPT API credentials not configured")

    @auto_timeit("yandexgpt_generate")
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.1) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": prompt})

        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": 2000
            },
            "messages": messages
        }

        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]
        except requests.exceptions.RequestException as e:
            return f"YandexGPT request error: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"YandexGPT response parsing error: {str(e)}"


class DocumentProcessor:
    """Process documents and generate embeddings"""

    def __init__(self):
        self.embeddings_model = None
        self.text_splitter = None

    def _ensure_imports(self):
        if self.embeddings_model is None:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('./model_minilm')

        if self.text_splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

    @auto_timeit("document_load")
    def load_document(self, file_path: str) -> List[str]:
        self._ensure_imports()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.lower().endswith('.pdf'):
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                return [doc.page_content for doc in documents]

            elif file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [f.read()]

            else:
                raise ValueError("Only PDF and TXT files are supported")

        except Exception as e:
            raise Exception(f"Document loading failed: {str(e)}")

    def split_text(self, texts: List[str]) -> List[str]:
        self._ensure_imports()
        if not texts:
            return []
        combined_text = "\n".join(texts)
        return self.text_splitter.split_text(combined_text)

    @auto_timeit("create_embeddings")
    def create_embeddings(self, chunks: List[str]):
        self._ensure_imports()
        if not chunks:
            return []
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embeddings_model.encode(chunks, show_progress_bar=True)
        return embeddings


class VectorSearch:
    """Simple cosine-similarity based vector search"""

    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.embeddings_model = None

    def _ensure_embeddings_model(self):
        if self.embeddings_model is None:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('./model_minilm')

    def add_documents(self, chunks: List[str], embeddings=None):
        self.chunks = chunks
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self._ensure_embeddings_model()
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            self.embeddings = self.embeddings_model.encode(chunks)

    @auto_timeit("vector_search")
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
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


class RAGSystem:
    """End-to-end RAG system"""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_search = VectorSearch()
        self.llm = YandexGPT()
        self.chunks = []

    @auto_timeit("rag_load_document")
    def load_and_process_document(self, file_path: str):
        logger.info(f"Loading document: {file_path}")
        texts = self.document_processor.load_document(file_path)
        logger.info(f"Loaded {len(texts)} text blocks")

        self.chunks = self.document_processor.split_text(texts)
        logger.info(f"Split into {len(self.chunks)} chunks")

        embeddings = self.document_processor.create_embeddings(self.chunks)
        self.vector_search.add_documents(self.chunks, embeddings)

        logger.info("Document processing completed")

    @auto_timeit("rag_ask_question")
    def ask_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
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
