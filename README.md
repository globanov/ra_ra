# RAG demo

deployed: https://ra-ra.onrender.com/ui

## Architecture

```mermaid
graph TB
    subgraph "External Services"
        YC[Yandex Cloud API]
    end

    subgraph "API Layer"
        API[FastAPI]
    end

    subgraph "Core Logic Layer"
        RS[RAG System]
        DP[Document Processor<br/>LangChain Text Splitter]
        VS[Vector Search<br/>Sentence Transformers]
        LLM[LLM Client<br/>YandexGPT]
    end

    subgraph "Supporting Layer"
        UTILS[Utils]
        TEMP[Templates]
    end

    subgraph "Data/Models"
        DOCS[Documents]
        VEC[Embeddings Model]
    end

    API --> RS
    RS --> DP
    RS --> VS
    RS --> LLM
    DP --> DOCS
    VS --> VEC
    LLM --> YC

    RS -.-> UTILS
    DP -.-> UTILS
    LLM -.-> UTILS
    API -.-> TEMP

    style API fill:#e1f5fe
    style RS fill:#f3e5f5
    style DP fill:#e8f5e8
    style VS fill:#fff3e0
    style LLM fill:#ffebee
```