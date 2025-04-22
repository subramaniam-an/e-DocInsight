"""
Service for handling file storage using MinIO.
"""
import io
from typing import Optional
from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource
import os
import json

from doc_retriever.config.settings import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    MINIO_PROCESSING_FOLDER,
    MINIO_PROCESSED_FOLDER,
    MINIO_CHUNKS_FOLDER
)

class StorageService:
    """Service for handling file storage using MinIO."""
    
    def __init__(self):
        """
        Initialize the storage service.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Name of the bucket to use
        """
        self.client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Set to True if using HTTPS
        )
        self.bucket_name = MINIO_BUCKET_NAME
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Ensure the bucket exists, create it if it doesn't."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            print(f"Error ensuring bucket exists: {str(e)}")
            raise
    
    def upload_to_processing(self, file_bytes: bytes, file_name: str, document_id: str) -> str:
        """
        Upload file to processing folder.
        
        Args:
            file_bytes: File content as bytes
            file_name: Name of the file
            document_id: The document ID to include in the key
            
        Returns:
            str: Object path in MinIO
        """
        try:
            # Embed document_id in the object key and store raw PDF
            object_name = f"{MINIO_PROCESSING_FOLDER}/{document_id}_{file_name}"
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(file_bytes),
                length=len(file_bytes),
                content_type='application/pdf'
            )
            return object_name
        except S3Error as e:
            print(f"Error uploading file: {str(e)}")
            raise
    
    def move_to_processed(self, object_name: str) -> str:
        """
        Move a file from processing to processed folder.

        Args:
            object_name: Current object path in MinIO

        Returns:
            str: New object path in MinIO
        """
        try:
            # Move by copying and deleting, preserving PDF content
            file_name = os.path.basename(object_name)
            new_object_name = f"{MINIO_PROCESSED_FOLDER}/{file_name}"
            # Copy the object
            self.client.copy_object(
                bucket_name=self.bucket_name,
                object_name=new_object_name,
                source=CopySource(bucket_name=self.bucket_name, object_name=object_name)
            )
            # Remove the original
            self.client.remove_object(self.bucket_name, object_name)
            return new_object_name
        except S3Error as e:
            print(f"Error moving file: {str(e)}")
            raise
    
    def get_document(self, document_id: str) -> Optional[bytes]:
        """
        Get a document by its ID.
        
        Args:
            document_id: The document ID to retrieve
            
        Returns:
            Optional[bytes]: The document content if found, None otherwise
        """
        try:
            # List objects in the processed folder
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=MINIO_PROCESSED_FOLDER + "/"
            )
            
            # Find the object with matching document ID
            for obj in objects:
                try:
                    response = self.client.get_object(
                        bucket_name=self.bucket_name,
                        object_name=obj.object_name
                    )
                    content = json.loads(response.read())
                    if content.get("document_id") == document_id:
                        return content["content"]
                except S3Error:
                    continue
                    
            return None
        except S3Error as e:
            print(f"Error getting document: {str(e)}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id: The document ID to delete
            
        Returns:
            bool: True if the document was deleted, False otherwise
        """
        try:
            # List objects in the processed folder
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=MINIO_PROCESSED_FOLDER + "/"
            )
            
            # Find and delete the object with matching document ID
            for obj in objects:
                try:
                    response = self.client.get_object(
                        bucket_name=self.bucket_name,
                        object_name=obj.object_name
                    )
                    content = json.loads(response.read())
                    if content.get("document_id") == document_id:
                        self.client.remove_object(
                            bucket_name=self.bucket_name,
                            object_name=obj.object_name
                        )
                        return True
                except S3Error:
                    continue
                    
            return False
        except S3Error as e:
            print(f"Error deleting document: {str(e)}")
            raise
    
    def read_file(self, object_path: str) -> bytes:
        """
        Read file from MinIO.
        
        Args:
            object_path: Path to the object in MinIO
            
        Returns:
            bytes: File content
        """
        try:
            response = self.client.get_object(self.bucket_name, object_path)
            return response.read()
        except S3Error as e:
            print(f"Error reading file: {str(e)}")
            raise
    
    def upload_chunk(self, file_name: str, chunk_index: int, chunk_data: str, document_id: str) -> str:
        """
        Upload a chunk to MinIO.
        
        Args:
            file_name: Original file name
            chunk_index: Index of the chunk
            chunk_data: Chunk content as string
            document_id: The document ID to include in the content
            
        Returns:
            str: Object path in MinIO
        """
        try:
            folder_path = f"{MINIO_CHUNKS_FOLDER}/{file_name}"
            object_name = f"{folder_path}/{file_name}_chunk_{chunk_index:02}.json"
            
            # Parse the existing chunk data and add document_id
            chunk_content = json.loads(chunk_data)
            chunk_content["document_id"] = document_id
            
            # Convert back to string
            updated_chunk_data = json.dumps(chunk_content)
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(updated_chunk_data.encode('utf-8')),
                length=len(updated_chunk_data.encode('utf-8')),
                content_type='application/json'
            )
            return object_name
        except S3Error as e:
            print(f"Error uploading chunk: {str(e)}")
            raise 