"""
Streamlit application for the e-DocInsight.
"""
import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Dict
import mimetypes
from datetime import datetime
import pandas as pd
import plotly.express as px
import os
import time

from doc_retriever.core.document_processor import DocumentProcessor
from doc_retriever.models.document import Document
from doc_retriever.config.settings import (
    MAX_FILE_SIZE,
    ALLOWED_MIME_TYPES,
    MAX_CONCURRENT_REQUESTS,
    UPLOADS_DIR
)
from doc_retriever.services.vector_store_service import VectorStoreService
from doc_retriever.services.minio_service import MinioService
from doc_retriever.services.database_service import DatabaseService

# Initialize document processor
@st.cache_resource
def get_document_processor():
    """Get or create the document processor instance."""
    return DocumentProcessor()

def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded file to the uploads directory with a unique name.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path: Path to the saved file
    """
    # Create a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{uploaded_file.name}"
    save_path = UPLOADS_DIR / unique_filename
    
    # Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return save_path

def validate_file(file) -> bool:
    """
    Validate uploaded file for security and compatibility.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file:
        return False
        
    # Check file size
    if file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
        return False
    
    # Check MIME type
    mime_type = mimetypes.guess_type(file.name)[0]
    if mime_type not in ALLOWED_MIME_TYPES:
        st.error(f"File type {mime_type} not supported. Please upload a PDF file.")
        return False
    
    # Check concurrent processing limit
    if st.session_state.processing_count >= MAX_CONCURRENT_REQUESTS:
        st.error(f"Maximum number of concurrent processing requests ({MAX_CONCURRENT_REQUESTS}) reached.")
        return False
    
    return True

def process_uploaded_file(file) -> Document:
    """Process an uploaded file and return the document object."""
    if not validate_file(file):
        raise ValueError("File validation failed")
    
    st.session_state.processing_count += 1
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        processor = get_document_processor()
        document = processor.process_document(file.getvalue(), file.name)
        return document
    finally:
        # Clean up temporary file
        Path(tmp_file_path).unlink()
        st.session_state.processing_count -= 1

def display_document_info(document: Document):
    """Display document information in a formatted way."""
    st.subheader("Document Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Filename:**", document.filename)
        st.write("**File Size:**", f"{document.file_size / 1024:.2f} KB")
        st.write("**MIME Type:**", document.mime_type)
    with col2:
        st.write("**Created At:**", document.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Updated At:**", document.updated_at.strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Page Count:**", document.page_count)
    
    st.subheader("Summary")
    st.write(document.summary)
    
    # Display document statistics
    st.subheader("Document Statistics")
    stats_data = {
        "Metric": ["Total Chunks", "Average Chunk Length", "Total Words"],
        "Value": [
            len(document.chunks),
            sum(len(chunk.split()) for chunk in document.chunks) / len(document.chunks),
            sum(len(chunk.split()) for chunk in document.chunks)
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), hide_index=True)

def display_search_results(results: List[Dict]):
    """Display search results in a formatted way."""
    st.subheader("Search Results")
    
    # Display relevance scores distribution in a chart
    if results:
        scores = [result.get('relevance_score', result['score']) for result in results]
        fig = px.bar(
            x=range(1, len(scores) + 1),
            y=scores,
            labels={'x': 'Result Rank', 'y': 'Relevance Score'},
            title='Relevance Score Distribution'
        )
        st.plotly_chart(fig)
    
    #Creates dropdown view of top matched results
    for i, result in enumerate(results, 1):
        with st.expander(f"Result {i} (Score: {result.get('relevance_score', result['score']):.4f})"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Document:**", result["metadata"]["filename"])
                st.write("**Chunk Index:**", result["chunk_index"])
                st.write("**Vector Score:**", f"{result['score']:.4f}")
                if 'entity_overlap' in result:
                    st.write("**Entity Overlap:**", f"{result['entity_overlap']:.4f}")
                st.write("**Chunk Length:**", len(result["rewritten_text"]))
            
            with col2:
                st.write("**Summary:**", result["summary"])
                st.write("**Context:**", result["end_context"])
                st.write("**Content:**", result["rewritten_text"])
                
                # Display entities if available
                if result.get("entities"):
                    st.write("**Entities:**", ", ".join(result["entities"]))

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="e-DocInsight",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("e-DocInsight")
    
    # Initialize session state attributes
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'active_processing' not in st.session_state:
        st.session_state.active_processing = 0
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Sidebar
    with st.sidebar:
        st.title("📚 e-DocInsight")
        st.write("Upload documents and search through them using natural language queries.")
        
        # Display system status
        st.subheader("System Status")
        st.write(f"Processed Documents: {len(st.session_state.processed_documents)}")
        st.write(f"Active Processing: {st.session_state.active_processing}")
        
        # Database Management
        st.sidebar.header("Database Management")
        
        # Create a container for the success message that persists across reruns
        message_container = st.sidebar.empty()
        
        if st.sidebar.button("Clear Database"):
            try:
                # Clear SQLite database
                db_service = DatabaseService()
                db_service.clear_database()
                
                # Clear Milvus collection
                vector_store = VectorStoreService()
                vector_store.clear_collection()
                
                # Clear Minio storage
                minio_service = MinioService()
                minio_service.clear_storage()
                
                # Reset session state
                st.session_state.processed_documents = []
                st.session_state.search_history = []
                
                # Show success message in the persistent container
                message_container.success("All databases and storage cleared successfully!")
                time.sleep(2)
                
                # Reload the page
                st.rerun()
            except Exception as e:
                message_container.error(f"Error clearing database: {str(e)}")
        
        # Display recent 5 searches
        if st.session_state.search_history:
            st.subheader("Recent Searches")
            for query in st.session_state.search_history[-5:]:
                st.write(f"- {query}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📥 Add Documents", "🔍 Search Documents"])
    
    # Tab 1: Add Documents
    with tab1:
        st.header("Add Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more documents to process"
        )
        
        if uploaded_files:
            # Show processing status
            if st.session_state.active_processing > 0:
                st.warning("⚠️ Processing is already in progress. Please wait...")
                st.progress(st.session_state.active_processing / MAX_CONCURRENT_REQUESTS)
            else:
                # Process files button
                if st.button("Process Documents"):
                    try:
                        # Update processing status
                        st.session_state.active_processing = len(uploaded_files)
                        
                        # Process each file
                        for file in uploaded_files:
                            try:
                                # Create temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                                    tmp_file.write(file.getvalue())
                                    tmp_file_path = tmp_file.name
                                
                                # Process the file
                                processor = get_document_processor()
                                document = processor.process_document(file.getvalue(), file.name)
                                
                                # Add to processed documents
                                st.session_state.processed_documents.append(document)
                                
                                # Display document info
                                display_document_info(document)
                                
                            except Exception as e:
                                st.error(f"Error processing {file.name}: {str(e)}")
                            finally:
                                # Clean up temporary file
                                if 'tmp_file_path' in locals():
                                    try:
                                        os.unlink(tmp_file_path)
                                    except:
                                        pass
                                
                                # Update processing status
                                st.session_state.active_processing -= 1
                        
                        # Show success message
                        st.success("All documents processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                    finally:
                        # Reset processing status
                        st.session_state.active_processing = 0
    
    # Tab 2: Search Documents
    with tab2:
        st.header("Search Documents")
        
        # Get list of available documents from MinIO
        minio_service = MinioService()
        available_docs = minio_service.list_documents()
        
        if not available_docs:
            st.info("No documents available for search. Please add documents first.")
        else:
            # Fiel Selection for querying
            selected_doc = st.selectbox(
                "Select Document to Search",
                options=available_docs,
                format_func=lambda x: x["filename"],
                help="Choose a document to search within"
            )
            
            # Show processing status
            if st.session_state.active_processing > 0:
                st.warning("⚠️ Search is disabled while documents are being processed. Please wait...")
                st.progress(st.session_state.active_processing / MAX_CONCURRENT_REQUESTS)
            else:
                # Only show search if no document processing is happening
                query = st.text_input("Enter your search query")
                
                if query:
                    # Add to search history
                    st.session_state.search_history.append(f"{selected_doc['filename']}: {query}")
                    
                    processor = get_document_processor()
                    with st.spinner("Searching..."):
                        # Search only in the selected document using its metadata
                        results = processor.search_documents(
                            query,
                            document_id=selected_doc["id"]
                        )[::-1]

                        if results:
                            chucks_data = [individual['rewritten_text'] for individual in results]
                            answer = processor.search_query(query, chucks_data)

                            st.subheader("Answer")
                            # Split the answer into main content and citations
                            parts = answer.split("\n\nCitations:")
                            if len(parts) > 1:
                                st.write(parts[0])  # Main answer
                                st.subheader("Citations")
                                st.write(parts[1])  # Citations
                            else:
                                st.write(answer)  # If no citations found, display as is
                            
                            display_search_results(results)
                        else:
                            st.info("No results found in the selected document.")

if __name__ == "__main__":
    main() 