import os
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
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

    def load_document(self, file_path):
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

    def split_text(self, texts):
        self._ensure_imports()
        if not texts:
            return []
        combined_text = "\n".join(texts)
        return self.text_splitter.split_text(combined_text)

    def create_embeddings(self, chunks):
        self._ensure_imports()
        if not chunks:
            return []
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.embeddings_model.encode(chunks, show_progress_bar=True)
        return embeddings
