import os # 這個是 Python 的 os 模組，用於生成 os
from dotenv import load_dotenv # 這個是 Python 的 dotenv 模組，用於生成 load_dotenv
"""
Core package for RAG chatbot functionality.

This package contains modules for:
- Embedding and vector database operations
- LLM model loading and management  
- RAG chain building
"""

# 自動載入 .env 檔案
load_dotenv()

class Config: # 這個是 Config 類別，用於生成 Config
    """專案配置class"""
    MODEL_NAME = os.environ.get("MODEL_NAME") # 這個是 MODEL_NAME，用於生成 MODEL_NAME
    MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER") # 這個是 MODEL_PROVIDER，用於生成 MODEL_PROVIDER
    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME") # 這個是 EMBEDDING_MODEL_NAME，用於生成 EMBEDDING_MODEL_NAME
    PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY") or "./knowledge_db" # 這個是 PERSIST_DIRECTORY，用於生成 PERSIST_DIRECTORY
    FILE_PATH = os.environ.get("FILE_PATH") or "" # 這個是 FILE_PATH，用於生成 FILE_PATH
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500)) # 這個是 CHUNK_SIZE，用於生成 CHUNK_SIZE
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 0)) # 這個是 CHUNK_OVERLAP，用於生成 CHUNK_OVERLAP
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # 這個是 GOOGLE_API_KEY，用於生成 GOOGLE_API_KEY

# 導入主要功能
from .data_processing import load_document, split_into_chunks # 這個是 data_processing 模組的 load_document 和 split_into_chunks
from .embedding_and_vector_db import get_embedding, calculate_embedding_token, create_embedding_chroma, build_retriever # 這個是 embedding_and_vector_db 模組的 get_embedding、calculate_embedding_token、create_embedding_chroma 和 build_retriever
from .llm_model import load_llm # 這個是 llm_model 模組的 load_llm
from .chain_builder import rag_chain # 這個是 chain_builder 模組的 rag_chain

# 公開的 API
__all__ = [ # 這個是 __all__，用於生成 __all__
    'Config', # 這個是 Config，用於生成 Config
    'load_document',
    'split_into_chunks', # 這個是 split_into_chunks，用於生成 split_into_chunks
    'get_embedding', # 這個是 get_embedding，用於生成 get_embedding
    'calculate_embedding_token', # 這個是 calculate_embedding_token，用於生成 calculate_embedding_token
    'create_embedding_chroma', # 這個是 create_embedding_chroma，用於生成 create_embedding_chroma
    'build_retriever', # 這個是 build_retriever，用於生成 build_retriever
    'load_llm', # 這個是 load_llm，用於生成 load_llm
    'rag_chain' # 這個是 rag_chain，用於生成 rag_chain
]
