"""
Service for managing document storage in Minio.
"""
from typing import List, Dict
from minio import Minio
from doc_retriever.config.settings import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    MINIO_PROCESSED_FOLDER
)
import io

class MinioService:
    """Service for managing document storage in Minio."""
    
    def __init__(self):
        """Initialize the Minio client."""
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Set to True if using HTTPS
        )
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Ensure the bucket exists."""
        if not self.client.bucket_exists(MINIO_BUCKET_NAME):
            self.client.make_bucket(MINIO_BUCKET_NAME)
    
    def list_documents(self) -> List[Dict]:
        """
        List all documents in the Minio bucket's processed folder.
        
        Returns:
            List of dictionaries containing document metadata
        """
        documents = []
        # List all PDF objects and extract document_id from key
        objects = self.client.list_objects(
            MINIO_BUCKET_NAME,
            prefix=MINIO_PROCESSED_FOLDER + "/",
            recursive=True
        )
        for obj in objects:
            filename = obj.object_name.split('/')[-1]
            if not filename.endswith('.pdf'):
                continue
            # Filename format: {document_id}_{original_filename}.pdf
            parts = filename.split('_', 1)
            if len(parts) == 2:
                document_id, original_filename = parts
            else:
                document_id = filename
                original_filename = filename
            documents.append({
                "id": document_id,
                "filename": original_filename,
                "size": obj.size,
                "last_modified": obj.last_modified,
                "path": obj.object_name
            })
        return documents
    
    def get_document(self, document_id: str) -> bytes:
        """
        Get a document from Minio.
        
        Args:
            document_id: The document ID
            
        Returns:
            Document content as bytes
        """
        # List and find PDF by key prefix
        objects = self.client.list_objects(
            MINIO_BUCKET_NAME,
            prefix=MINIO_PROCESSED_FOLDER + "/",
            recursive=True
        )
        for obj in objects:
            filename = obj.object_name.split('/')[-1]
            if filename.startswith(f"{document_id}_"):
                response = self.client.get_object(MINIO_BUCKET_NAME, obj.object_name)
                return response.read()
        return None
    
    def store_document(self, document_id: str, filename: str, content: bytes) -> bool:
        """
        Store a document in Minio.
        
        Args:
            document_id: The document ID (UUID)
            filename: The original filename
            content: The document content as bytes
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            Exception: If document already exists
        """
        # Construct object name embedding document_id
        object_name = f"{MINIO_PROCESSED_FOLDER}/{document_id}_{filename}"
        
        # Check if document already exists
        try:
            self.client.stat_object(MINIO_BUCKET_NAME, object_name)
            raise Exception(f"Document {filename} already exists in {MINIO_PROCESSED_FOLDER}")
        except Exception as e:
            if "NoSuchKey" not in str(e):
                raise
        
        # Store PDF content
        self.client.put_object(
            MINIO_BUCKET_NAME,
            object_name,
            io.BytesIO(content),
            len(content),
            content_type='application/pdf'
        )
        return True
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from Minio.
        
        Args:
            document_id: The document ID
            
        Returns:
            True if successful, False otherwise
        """
        # Find and delete all PDFs keyed by document_id
        deleted = False
        objects = self.client.list_objects(
            MINIO_BUCKET_NAME,
            prefix=MINIO_PROCESSED_FOLDER + "/",
            recursive=True
        )
        for obj in objects:
            filename = obj.object_name.split('/')[-1]
            if filename.startswith(f"{document_id}_"):
                self.client.remove_object(MINIO_BUCKET_NAME, obj.object_name)
                deleted = True
        return deleted
    
    def clear_storage(self) -> bool:
        """
        Clear all objects in the Minio bucket.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, list all objects in the bucket
            objects = list(self.client.list_objects(
                MINIO_BUCKET_NAME,
                recursive=True
            ))
            
            if not objects:
                print("No objects found to clear")
                return True
            
            # Delete each object
            for obj in objects:
                try:
                    self.client.remove_object(MINIO_BUCKET_NAME, obj.object_name)
                    print(f"Removed object: {obj.object_name}")
                except Exception as e:
                    print(f"Error removing object {obj.object_name}: {str(e)}")
                    continue
            
            # Verify that all objects are removed
            remaining_objects = list(self.client.list_objects(
                MINIO_BUCKET_NAME,
                recursive=True
            ))
            
            if remaining_objects:
                print(f"Warning: {len(remaining_objects)} objects still remain in the bucket")
                return False
            
            print("All objects successfully cleared from the bucket")
            return True
            
        except Exception as e:
            print(f"Error clearing storage: {str(e)}")
            return False 