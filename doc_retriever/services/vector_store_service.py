"""
Service for managing vector storage using Milvus.
"""
from typing import List, Dict, Optional
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    IndexType
)
from tenacity import retry, stop_after_attempt, wait_exponential

from doc_retriever.config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    DIMENSION,
    MAX_RETRIES,
    RETRY_DELAY
)

class VectorStoreService:
    """Service for managing vector storage using Milvus."""
    
    def __init__(self):
        """Initialize the Milvus connection."""
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """Connect to Milvus server."""
        try:
            # Disconnect any existing connections first
            connections.disconnect("default")
            
            # Connect with explicit alias
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
        except Exception as e:
            print(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def _disconnect(self):
        """Disconnect from Milvus server."""
        connections.disconnect("default")
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper schema and indexes."""
        try:
            # Define the schema fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
                FieldSchema(name="rewritten_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="end_context", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="entities", dtype=DataType.JSON),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields=fields, description="Document chunks with embeddings", enable_dynamic_field=True)
            
            if not utility.has_collection(COLLECTION_NAME):
                # Create new collection
                self.collection = Collection(name=COLLECTION_NAME, schema=schema)
            else:
                # Get existing collection
                self.collection = Collection(name=COLLECTION_NAME)
                
                # Verify schema
                existing_schema = self.collection.schema
                if not all(field.name in [f.name for f in existing_schema.fields] for field in fields):
                    # If schema doesn't match, recreate collection
                    print("Schema mismatch detected. Recreating collection...")
                    utility.drop_collection(COLLECTION_NAME)
                    self.collection = Collection(name=COLLECTION_NAME, schema=schema)
            
            # Check and create indexes
            existing_indexes = {idx.field_name: idx for idx in self.collection.indexes}
            
            # Check embedding index
            if "embedding" not in existing_indexes:
                print("Creating embedding index...")
                # Create IVF_PQ index for vector search
                # IVF parameters
                nlist = 1024  # Number of clusters for IVF
                
                # PQ parameters
                m = 8  # Number of sub-vectors for PQ
                nbits = 8  # Number of bits per sub-vector
                
                index_params = {
                    "metric_type": "L2",
                    "index_type": IndexType.IVF_PQ,
                    "params": {
                        "nlist": nlist,  # Number of clusters for IVF
                        "m": m,  # Number of sub-vectors for PQ
                        "nbits": nbits  # Number of bits per sub-vector
                    }
                }
                
                # Create the index
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
            
            # Check document_id index
            if "document_id" not in existing_indexes:
                print("Creating document_id index...")
                # Create a scalar index on document_id for faster filtering
                self.collection.create_index(
                    field_name="document_id",
                    index_params={
                        "index_type": "Trie"
                    }
                )
                
        except Exception as e:
            print(f"Error ensuring collection: {str(e)}")
            raise
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from the query text."""
        words = query.lower().split()
        return [word for word in words if len(word) > 3]
    
    def _calculate_entity_overlap(self, query_entities: List[str], chunk_entities: List[str]) -> float:
        """Calculate the overlap between query entities and chunk entities."""
        if not query_entities or not chunk_entities:
            return 0.0
        
        query_set = set(query_entities)
        chunk_set = set(chunk_entities)
        
        intersection = len(query_set.intersection(chunk_set))
        union = len(query_set.union(chunk_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_relevance_score(
        self,
        vector_score: float,
        entity_overlap: float,
        chunk_length: int,
        max_chunk_length: int
    ) -> float:
        """Calculate a combined relevance score."""
        # Normalize chunk length
        length_score = 1 - (chunk_length / max_chunk_length)
        
        # Weight the different components
        vector_weight = 0.6
        entity_weight = 0.3
        length_weight = 0.1
        
        return (
            vector_weight * vector_score +
            entity_weight * entity_overlap +
            length_weight * length_score
        )
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def store_chunks(
        self,
        document_id: str,
        chunks_data: List[Dict[str, str]],
        embeddings: List[List[float]],
        metadata: Dict
    ) -> None:
        """
        Store document chunks with their embeddings in Milvus.
        
        Args:
            document_id: Unique identifier for the document
            chunks_data: List of dictionaries containing chunk data
            embeddings: List of embedding vectors
            metadata: Additional metadata for the chunks
            
        Raises:
            Exception: If storage operation fails
        """
        try:
            # First ensure we have a valid connection
            self._connect()
            
            # Always ensure collection exists with proper schema
            self._ensure_collection()
            
            # Load the collection
            self.collection.load()
            
            # Prepare entities for insertion
            entities = []
            for i, (chunk_data, embedding) in enumerate(zip(chunks_data, embeddings)):
                entity = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "embedding": embedding,
                    "rewritten_text": chunk_data["rewritten_text"],
                    "summary": chunk_data["summary"],
                    "end_context": chunk_data["end_context"],
                    "entities": chunk_data.get("entities", []),
                    "metadata": metadata
                }
                entities.append(entity)
            
            # Insert entities
            self.collection.insert(entities)
            
            # Flush to ensure data is written to disk
            self.collection.flush()
            
        except Exception as e:
            error_msg = str(e)
            if "collection not loaded" in error_msg:
                error_msg = "Failed to store chunks: Collection needs to be loaded first"
            elif "collection not exist" in error_msg:
                error_msg = "Failed to store chunks: Collection needs to be created first"
            else:
                error_msg = f"Failed to store chunks: {error_msg}"
            print(error_msg)
            raise Exception(error_msg)
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        reraise=True
    )
    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = 3,
        document_id: Optional[str] = None,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity and entity matching.
        
        Args:
            query_embedding: The query embedding vector
            query_text: The original query text for entity extraction
            limit: Maximum number of results to return
            document_id: Optional document ID to filter results
            rerank: Whether to rerank results using entity matching
            
        Returns:
            List of similar chunks with their metadata
            
        Raises:
            Exception: If search operation fails
        """
        try:
            # Load collection
            self.collection.load()
            
            # Stage 1: Semantic Search using vectors
            search_params = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 64  # Number of clusters to search
                }
            }
            
            # Prepare expression for filtering
            expr = None
            if document_id:
                # Use single quotes around the document_id for Milvus expr
                expr = f"document_id == '{document_id}'"
                print(f"[DEBUG] Using filter expr: {expr}")
            
            initial_limit = limit * 5  # Get more candidates for entity filtering
            
            # Perform the search
            vector_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=initial_limit,
                expr=expr,
                output_fields=["document_id", "chunk_index", "rewritten_text", "summary", "end_context", "entities", "metadata"]
            )
            
            # Debug raw vector hits
            raw_hits = [hit for hits in vector_results for hit in hits]
            print(f"[DEBUG] Total raw vector hits: {len(raw_hits)}")
            for hit in raw_hits:
                print(f"[DEBUG] Hit doc_id={hit.entity.document_id}, score={hit.score}")
            
            # Stage 2: Entity-based filtering and reranking
            query_entities = self._extract_entities_from_query(query_text)
            
            # Format and filter results
            formatted_results = []
            max_chunk_length = 0
            
            # Collect and optionally filter
            for hit in raw_hits:
                chunk_length = len(hit.entity.rewritten_text)
                max_chunk_length = max(max_chunk_length, chunk_length)
                entity_overlap = self._calculate_entity_overlap(query_entities, hit.entity.entities)
                # If rerank is on, include all but log overlap
                if rerank:
                    print(f"[DEBUG] Entity overlap for doc_id={hit.entity.document_id}, chunk={hit.entity.chunk_index}: {entity_overlap}")
                formatted_results.append({
                    "score": hit.score,
                    "document_id": hit.entity.document_id,
                    "chunk_index": hit.entity.chunk_index,
                    "rewritten_text": hit.entity.rewritten_text,
                    "summary": hit.entity.summary,
                    "end_context": hit.entity.end_context,
                    "entities": hit.entity.entities,
                    "metadata": hit.entity.metadata,
                    "chunk_length": chunk_length,
                    "entity_overlap": entity_overlap
                })
            
            # Calculate final relevance scores
            for result in formatted_results:
                result["relevance_score"] = self._calculate_relevance_score(
                    result["score"],
                    result["entity_overlap"],
                    result["chunk_length"],
                    max_chunk_length
                )
            
            # Sort by relevance score
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Return top results
            return formatted_results[:limit]
            
        except Exception as e:
            print(f"Failed to search chunks: {str(e)}")
            raise
    
    def clear_collection(self):
        """Clear all entities from the Milvus collection."""
        try:
            # First ensure we have a valid connection
            self._connect()
            
            if utility.has_collection(COLLECTION_NAME):
                # Get a reference to the collection
                self.collection = Collection(COLLECTION_NAME)
                
                # Load the collection
                self.collection.load()
                
                # Delete all entities using a valid expression that matches everything
                self.collection.delete(expr="document_id != ''")
                self.collection.flush()
                print(f"Successfully cleared collection '{COLLECTION_NAME}'")
        except Exception as e:
            error_msg = str(e)
            if "collection not loaded" in error_msg:
                error_msg = "Failed to clear collection: Collection needs to be loaded first"
            elif "Illegal str variables" in error_msg:
                error_msg = "Failed to clear collection: Invalid deletion expression"
            else:
                error_msg = f"Failed to clear collection: {error_msg}"
            print(error_msg)
            raise Exception(error_msg)
    
    def __del__(self):
        """Cleanup when the service is destroyed."""
        self._disconnect()

    def get_collection_status(self) -> Dict:
        """
        Get the status of the collection including number of entities and indexes.
        
        Returns:
            Dictionary containing collection status information
        """
        try:
            self._connect()
            if not utility.has_collection(COLLECTION_NAME):
                return {
                    "exists": False,
                    "entity_count": 0,
                    "indexes": []
                }
            
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            
            # Get entity count
            entity_count = self.collection.num_entities
            
            # Get indexes
            indexes = []
            for index in self.collection.indexes:
                indexes.append({
                    "field_name": index.field_name,
                    "index_type": index.index_type,
                    "metric_type": index.metric_type
                })
            
            return {
                "exists": True,
                "entity_count": entity_count,
                "indexes": indexes
            }
        except Exception as e:
            print(f"Failed to get collection status: {str(e)}")
            return {
                "exists": False,
                "entity_count": 0,
                "indexes": [],
                "error": str(e)
            }

    def get_all_vectors(self, limit: int = 100) -> List[Dict]:
        """
        Get all vectors from the collection with a limit.
        
        Args:
            limit: Maximum number of vectors to retrieve
            
        Returns:
            List of vectors with their metadata
        """
        try:
            # Ensure a valid connection
            self._connect()
            
            # Get collection
            if not utility.has_collection(COLLECTION_NAME):
                print(f"Collection {COLLECTION_NAME} does not exist")
                return []
                
            self.collection = Collection(COLLECTION_NAME)
            
            # Load collection into memory
            if not self.collection.has_index():
                print("Collection has no index")
                return []
                
            self.collection.load()
            
            # Query all vectors with a limit
            results = self.collection.query(
                expr="",
                output_fields=["document_id", "chunk_index", "rewritten_text", "summary", "entities", "metadata"],
                limit=limit,
                consistency_level="Strong"
            )
            
            return results
        except Exception as e:
            print(f"Failed to get vectors: {str(e)}")
            return []
        finally:
            # Always try to disconnect
            try:
                connections.disconnect("default")
            except:
                pass