"""
Core document processing service that orchestrates the document processing pipeline.
"""
import uuid
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import re
import json
from tqdm import tqdm

from doc_retriever.config.settings import (
    CHUNK_SIZE,
    MAX_RETRIES,
    RETRY_DELAY,
    BATCH_SIZE
)
from doc_retriever.models.document import Document
from doc_retriever.services.embedding_service import EmbeddingService
from doc_retriever.services.summarizer_service import SummarizerService
from doc_retriever.services.vector_store_service import VectorStoreService
from doc_retriever.services.storage_service import StorageService

class DocumentProcessor:
    """Service for processing documents through the pipeline."""
    
    def __init__(self):
        """Initialize the document processor with required services."""
        self.embedding_service = EmbeddingService()
        self.summarizer_service = SummarizerService()
        self.vector_store_service = VectorStoreService()
        self.storage_service = StorageService()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on system resources
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for file metadata."""
        conn = sqlite3.connect("file_metadata.db")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                file_extension TEXT,
                file_hash TEXT UNIQUE,
                size INTEGER,
                uploaded_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _file_exists_in_db(self, file_hash: str) -> bool:
        """Check if a file with the given hash exists in the database."""
        conn = sqlite3.connect("file_metadata.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files WHERE file_hash=?", (file_hash,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def _insert_metadata(self, file_name: str, file_extension: str, file_hash: str, file_size: int):
        """Insert file metadata into the database."""
        conn = sqlite3.connect("file_metadata.db")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO files (file_name, file_extension, file_hash, size, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (file_name, file_extension, file_hash, file_size, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def process_document(self, file_bytes: bytes, file_name: str) -> Document:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_bytes: Document content as bytes
            file_name: Name of the file
            
        Returns:
            Processed Document object
            
        Raises:
            Exception: If document processing fails
        """
        try:
            # Calculate file hash
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            file_extension = Path(file_name).suffix[1:]  # Remove leading dot
            
            # Check if file already exists
            if self._file_exists_in_db(file_hash):
                raise ValueError("File already exists in the database")
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Upload to MinIO processing folder
            processing_path = self.storage_service.upload_to_processing(file_bytes, file_name, document_id)
            
            # Extract text from PDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_chunks = []
            total_pages = len(doc)
            
            print(f"Processing document with {total_pages} pages...")
            for i, page in enumerate(tqdm(doc, desc="Extracting text")):
                text = page.get_text()
                chunks = self._split_text(text)
                text_chunks.extend(chunks)
                del text
                del chunks
            
            metadata = {
                "filename": file_name,
                "file_size": len(file_bytes),
                "mime_type": "application/pdf",
                "page_count": total_pages,
                "created_at": datetime.now().isoformat(),
                "storage_location": "minio"
            }
            
            # Process chunks in batches
            processed_chunks = []
            previous_summary = None
            previous_end_context = None
            
            print("Processing chunks...")
            for i in tqdm(range(0, len(text_chunks), BATCH_SIZE), desc="Processing chunks"):
                batch = text_chunks[i:i + BATCH_SIZE]
                batch_results = self._process_chunk_batch(
                    batch,
                    previous_summary,
                    previous_end_context
                )
                processed_chunks.extend(batch_results)
                
                # Update context for next batch
                if batch_results:
                    previous_summary = batch_results[-1]["summary"]
                    previous_end_context = batch_results[-1]["end_context"]
                
                # Store chunks in MinIO
                for idx, chunk in enumerate(batch_results, start=i+1):
                    chunk_data = {
                        "rewritten_text": chunk["rewritten_text"],
                        "summary": chunk["summary"],
                        "end_context": chunk["end_context"]
                    }
                    # Convert chunk_data to JSON string for valid parsing
                    chunk_json = json.dumps(chunk_data)
                    self.storage_service.upload_chunk(
                        file_name,
                        idx,
                        chunk_json,
                        document_id
                    )
                
                del batch
                del batch_results
            
            # Generate embeddings for rewritten chunks in batches
            print("Generating embeddings...")
            embeddings = []
            for i in tqdm(range(0, len(processed_chunks), BATCH_SIZE), desc="Generating embeddings"):
                batch = [chunk["rewritten_text"] for chunk in processed_chunks[i:i + BATCH_SIZE]]
                batch_embeddings = self.embedding_service.get_embeddings_batch(batch)
                embeddings.extend(batch_embeddings)
                del batch
                del batch_embeddings
            
            # Store in vector database in batches
            print("Storing in vector database...")
            for i in tqdm(range(0, len(processed_chunks), BATCH_SIZE), desc="Storing chunks"):
                batch_chunks = processed_chunks[i:i + BATCH_SIZE]
                batch_embeddings = embeddings[i:i + BATCH_SIZE]
                self.vector_store_service.store_chunks(
                    document_id,
                    batch_chunks,
                    batch_embeddings,
                    metadata
                )
                del batch_chunks
                del batch_embeddings
            
            # Move file to processed folder
            self.storage_service.move_to_processed(processing_path)
            
            # Save metadata to database
            self._insert_metadata(file_name, file_extension, file_hash, len(file_bytes))
            
            # Create document object
            document = Document(
                id=document_id,
                filename=file_name,
                content="\n".join(text_chunks),
                chunks=[chunk["rewritten_text"] for chunk in processed_chunks],
                metadata=metadata,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_size=len(file_bytes),
                mime_type="application/pdf",
                page_count=total_pages,
                summary="\n".join(chunk["summary"] for chunk in processed_chunks)
            )
            
            return document
            
        except Exception as e:
            print(f"Failed to process document: {str(e)}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _process_chunk_batch(
        self,
        chunks: List[str],
        previous_summary: Optional[str] = None,
        previous_end_context: Optional[str] = None
    ) -> List[Dict]:
        """Process a batch of chunks in parallel."""
        futures = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            future = self.executor.submit(
                self.summarizer_service.process_chunk,
                chunk,
                previous_summary,
                previous_end_context
            )
            futures.append(future)
        
        return [future.result() for future in futures]
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using optimized regex-based splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Use regex for more efficient chunk boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > CHUNK_SIZE:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def __del__(self):
        """Cleanup when the processor is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def search_documents(self, query: str, document_id: str = None) -> List[Dict]:
        """
        Search documents using the query.
        
        Args:
            query: The search query
            document_id: Optional document ID to search within
            
        Returns:
            List of search results
        """
        try:
            # Get vector store service
            vector_store = VectorStoreService()
            
            # Generate query embedding
            query_embedding = self.embedding_service.get_embeddings(query)
            
            # Search in vector store
            results = vector_store.search(
                query_embedding=query_embedding,
                query_text=query,
                document_id=document_id
            )
            
            return results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def search_query(self, query: str, matched_records: List[str]):
        """Search using the query and matched records."""
        return self.summarizer_service.search_query(query, matched_records)
    
    def clear_database(self):
        """Clear all data from the SQLite database."""
        try:
            conn = sqlite3.connect("file_metadata.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM files")
            conn.commit()
            print("Successfully cleared all data from the database")
        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            raise
        finally:
            conn.close()

