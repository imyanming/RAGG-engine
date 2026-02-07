from langchain_community.embeddings import HuggingFaceEmbeddings 
# 這個是 LangChain 的 embeddings 模組，用於生成 HuggingFaceEmbeddings
from langchain_chroma import Chroma 
# 這個是 LangChain 的 chroma 模組，用於生成 Chroma


# Load embeddings model (Using HuggingFace Embedding model)
def get_embedding(embedding_model_name: str) -> HuggingFaceEmbeddings: 
    # 這個是 get_embedding 函數，用於生成 get_embedding
    """
    Loads the HuggingFace embedding model.
    parameters:
        embedding_model_name: The name of the embedding model to load.
    returns:
        An instance of the HuggingFaceEmbeddings class.
    """
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        model_kwargs = {'device': 'cpu'}
    )
    return embedding # 返回 HuggingFaceEmbeddings 模型物件


# Embedding into chroma locally 
def create_embedding_chroma(chunks, embedding, collection_name: str = None) -> Chroma: 
    # 這個是 create_embedding_chroma 函數，用於生成 create_embedding_chroma
    """
    建構chroma資料庫並將切片embed進去
    parameters:
        chunks: 文件切片
        embedding: 嵌入模型
        collection_name: 可選的 collection 名稱，如果不提供則使用隨機名稱
    return:
        Chroma
    """
    import uuid # 這個是 Python 的 uuid 模組，用於生成 uuid
    
    # Use a unique collection name to avoid conflicts
    if collection_name is None:
        collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}" 
    # 使用 uuid 生成唯一的 collection 名稱
    
    vector_store = Chroma.from_documents(
        chunks,
        embedding,
        collection_name=collection_name,
        # persist_directory="./knowledge_db",  # Where to save data locally, remove if not necessary
    )
    return vector_store # 返回 Chroma 模型物件


# calculate embedding cost using transformers tokenizer
def calculate_embedding_token(embedding_model_name: str, texts) -> int: 
    # 這個是 calculate_embedding_token 函數，用於生成 calculate_embedding_token
    """
    計算嵌入模型token花費
    parameters:
        embedding_model_name: str: 嵌入模型名
        texts: 文本
    return:
        total_tokens: int
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name) 
    # 使用 AutoTokenizer 初始化 tokenizer
    total_tokens = sum(len(tokenizer.encode(page.page_content)) for page in texts)
    return total_tokens # 返回編碼後的 token 數量


# build retriever
def build_retriever(vector_store, top_k: int=5): 
    # 這個是 build_retriever 函數，用於生成 build_retriever
    """
    構建Retriever，用於語意搜尋知識庫中最相關的文件片段
    parameters:
        vector_store: 已建構的Chroma向量庫
        top_k: 返回最相關片段的數量
    return:
        retriever: 可用於檢索的物件
    """
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={"k": top_k}) 
    # 使用 vector_store 建立 retriever
    return retriever
