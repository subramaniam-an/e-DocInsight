"""
Service for generating text summaries and extracting entities using LiteLLM.
"""
from typing import Dict, Optional, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from litellm import RateLimitError
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.chat_models import ChatLiteLLM
import litellm

from doc_retriever.config.settings import (
    OPENAI_API_KEY,
    MAX_RETRIES,
    RETRY_DELAY,
    TEMPERATURE,
    CHAT_MODEL
)

# Enable verbose logging for litellm
litellm.success_callback = ['langfuse']
litellm.failure_callback = ['langfuse']


class SummarizerService:
    """Service for generating text summaries using LiteLLM."""
    
    def __init__(self):
        """Initialize the SummarizerService with LiteLLM."""
        self.llm = ChatLiteLLM(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            api_base="https://api.openai.com/v1",
            temperature=TEMPERATURE
        )
        # self.langfuse = LangfuseService()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True
    )
    def process_chunk(
        self,
        chunk: str,
        previous_summary: Optional[str] = None,
        previous_end_context: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Process a text chunk to generate rewritten content, summary, end context, and entities.
        
        Args:
            chunk: The text chunk to process
            previous_summary: Summary from the previous chunk
            previous_end_context: End context from the previous chunk
            trace_id: Optional Langfuse trace ID for tracking
            metadata: Optional metadata to attach to the trace
            
        Returns:
            Dictionary containing rewritten text, summary, end context, and entities
            
        Raises:
            ValueError: If the input chunk is empty
        """
        if not chunk or not chunk.strip():
            raise ValueError("Input chunk cannot be empty")
            
        try:
            
            # Prepare context-aware input
            context = ""
            if previous_summary and previous_end_context:
                context = f"Previous summary: {previous_summary}\nPrevious ending: {previous_end_context}\n\n"
            
            full_text = context + chunk

            response = self.llm.invoke([
                SystemMessage(
                    content="You are a highly skilled assistant that can rewrite text for clarity while maintaining key information, summarize key points, extract key entities, and identify the final key points that connect to the next section."),
                HumanMessage(content=f"""
                        Given the following text, perform four tasks:
                        1. Rewrite the text to improve clarity while keeping key information intact.
                        2. Summarize the key points concisely.
                        3. Extract key entities (important people, organizations, concepts, etc.) from the text.
                        4. Identify the final key points that connect to the next section.

                        Text:
                        {full_text}

                        Return the results in the following JSON format:
                        {{
                            "rewritten_text": "<your rewritten text>",
                            "summary": "<your summary>",
                            "entities": ["<entity1>", "<entity2>", ...],
                            "end_context": "<your end context>"
                        }}
                        """)
            ])

            result = response.content.strip()
            result_dict = json.loads(result)
            
            return result_dict
            
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            raise

    def search_query(self, query: str, matched_records: List[str]) -> str:
        """
        Search using the query and matched records.
        
        Args:
            query: The search query
            matched_records: List of matched text records
            
        Returns:
            str: Search results with citations
        """
        try:
            response = self.llm.invoke([
                SystemMessage(
                    content="You are a helpful assistant that provides accurate information with proper citations."),
                HumanMessage(content=f"""
                        Given the following query and matched records, provide a comprehensive answer with citations.

                        Query: {query}

                        Matched Records:
                        {chr(10).join(f'[{i+1}] {record}' for i, record in enumerate(matched_records))}

                        Return your response in the following format:
                        <your comprehensive answer>

                        Citations:
                        [1] <citation for first record>
                        [2] <citation for second record>
                        ...
                        """)
            ])
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error searching query: {str(e)}")
            raise