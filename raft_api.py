from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, TypedDict
import os
import tempfile
import logging
import json
from rag_system import RAGSystem
import uvicorn
from raft_utils import auto_timeit

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    rag_system: RAGSystem


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    similarities: List[float]
    total_chunks: int


class DocumentResponse(BaseModel):
    status: str
    chunks_count: int
    document_name: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG system")
    app.state.rag_system = RAGSystem()
    yield
    logger.info("Shutting down RAG system")
    app.state.rag_system = None


class PrettyJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        ).encode("utf-8")


app = FastAPI(
    title="Raft RAG API",
    description="RAG system demo API",
    version="1.0",
    lifespan=lifespan,
    default_response_class=PrettyJSONResponse,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
templates = Jinja2Templates(directory="templates")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


@app.get("/ui")
async def ui_interface(request: Request):
    return templates.TemplateResponse("minimal.html", {"request": request})


@app.post("/upload", response_model=DocumentResponse)
@auto_timeit("api_upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    rag_system = request.app.state.rag_system
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {'.pdf', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
        )

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        rag_system.load_and_process_document(tmp_path)
        return DocumentResponse(
            status="success",
            chunks_count=len(rag_system.chunks),
            document_name=file.filename
        )
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=QuestionResponse)
@auto_timeit("api_ask")
async def ask_question(request: Request, req: QuestionRequest):
    rag_system = request.app.state.rag_system
    if not rag_system or not rag_system.chunks:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a document first.")

    try:
        logger.info(f"Processing question: {req.question[:50]}...")
        result = rag_system.ask_question(
            question=req.question,
            top_k=req.top_k
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        sources = [source["content"] for source in result["sources"]]
        similarities = [source["similarity"] for source in result["sources"]]

        return QuestionResponse(
            answer=result["answer"],
            sources=sources,
            similarities=similarities,
            total_chunks=result["stats"]["total_chunks"]
        )
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(request: Request):
    rag_system = request.app.state.rag_system
    if not rag_system:
        return {"status": "system_not_initialized"}
    return {
        "document_loaded": len(rag_system.chunks) > 0,
        "chunks_count": len(rag_system.chunks) if rag_system.chunks else 0,
        "status": "ready"
    }


@app.get("/health")
async def health_check(request: Request):
    rag_system = request.app.state.rag_system
    return {
        "status": "healthy" if rag_system else "unhealthy",
        "service": "raft_rag_api",
        "version": "1.0"
    }


@app.get("/")
async def root():
    return {
        "project": "Raft RAG API",
        "description": "RAG system demo API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This information",
            "GET /ui": "Web UI interface",
            "POST /upload": "Upload PDF/TXT document",
            "POST /ask": "Ask a question about the document",
            "GET /stats": "System stats",
            "GET /health": "Health check"
        },
        "documentation": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    # Настраиваем логирование ДО запуска
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Устанавливаем уровень для всех логгеров
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("raft_api").setLevel(logging.INFO)
    logging.getLogger("rag_core").setLevel(logging.INFO)
    logging.getLogger("raft_utils").setLevel(logging.INFO)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",  # Логи uvicorn
        access_log=True    # Access логи
    )
