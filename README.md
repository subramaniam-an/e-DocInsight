# e-DocInsight

A powerful document processing and retrieval system built with Streamlit, SQLLite, Milvus, and MinIO. Upload PDF documents, process them (chunking, embedding, summarization), and search through them using natural language queries.

## Project Structure

```
.
├── doc_retriever/         # Application code
│   ├── api/               # Streamlit UI and API layer
│   ├── config/            # Configuration settings and environment
│   ├── core/              # Core document processing logic
│   ├── models/            # Data models (Document dataclass)
│   ├── services/          # Service classes (MinIO, Milvus, embeddings, summarizer)
│   └── utils/             # Utility functions (e.g., text chunker)
├── data/                  # Local data directories (uploads, cache, etc.)
├── tests/                 # Pytest test suite
├── .env.example           # Example environment variables
├── .env                   # Your environment variables (not committed)
├── requirements.in        # Primary dependencies (uses pip-tools)
├── requirements.txt       # Pinned dependencies
├── run.py                 # Main entry point (starts Streamlit app)
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose setup
└── README.md              # Project documentation
```

## Setup

1. **Clone & enter the repo**

```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Create & activate a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**

This project uses pip-tools, so install/update pinned versions:

```bash
pip install pip-tools
pip-compile requirements.in  # regenerates requirements.txt if needed
pip-sync #installs the necessaries in requirements.txt
pip install -r requirements.txt # Optional: Also can use this instead of pip-sync
```

4. **Configure environment variables**

```bash
cp .env.example .env
# Then open .env and set:
# OPENAI_API_KEY, MILVUS_HOST, MILVUS_PORT, MINIO_* settings, etc.
```

## Running the Application

```bash
streamlit run run.py
```

Open your browser at http://localhost:8501/.

## Features

- PDF document upload and processing (storage in MinIO)
- Text chunking, summarization, and metadata extraction
- Vector embeddings stored in Milvus with per-chunk `document_id`
- Two-tab UI:
  1. **Add Documents** – upload and process documents
  2. **Search Documents** – pick a document, enter a query, and see ranked results
- Side-bar UI:
  1. **Clear Data** - clears the data in milvus, minio, sqllite databases. 

## Testing

Run the pytest suite:

```bash
pytest
```

## Docker Support

Start all services (MinIO, Milvus) and the app:

```bash
docker-compose up
```

## License

All rights reserved. This project is provided for demonstration purposes only and is not licensed for redistribution or commercial use.