import os # 這個是 Python 的 os 模組，用於生成 os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, JSONLoader
# 這個是 LangChain 的 document_loaders 模組，用於生成 document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter # 這個是 LangChain 的 text_splitters 模組，用於生成 text_splitters

# Load file with different file type, including PDF, DOCX, TXT, Markdown files
def load_document(file) -> list | None: # 這個是 load_document 函數，用於生成 load_document
    """
    載入文件,包含PDF, Docx, Txt, Markdown檔案類型
    parameters:
        file: 欲傳入文件
    return: 
        list | None
    """
    
    # 將文件路徑依照'名稱'和'檔案名'切分，只需要檔案名
    name, extension = os.path.splitext(file)
    extension = extension.lower()  # 轉換為小寫以支持大小寫不同的擴展名
    
    if extension == '.pdf': # 如果檔案是 PDF 格式
        print(f'loading {file}')
        loader = PyPDFLoader(file, extract_images=True) # 使用 PyPDFLoader 載入 PDF 文件
    elif extension == '.docx':
        print(f'loading {file}')
        loader = Docx2txtLoader(file) # 使用 Docx2txtLoader 載入 DOCX 文件
    elif extension == '.txt':
        print(f'loading {file}')
        loader = TextLoader(file, encoding='utf-8') # 使用 TextLoader 載入 TXT 文件
    elif extension == '.md' or extension == '.markdown':
        print(f'loading {file}')
        loader = TextLoader(file, encoding='utf-8') # 使用 TextLoader 載入 Markdown 文件
    elif extension == '.json':
        print(f'loading {file}')
        loader = JSONLoader(file) # 使用 JSONLoader 載入 JSON 文件
    else:
        print(f'This document format ({extension}) is not supported') # 如果檔案格式不支持，則返回 None
        return None
    
    data = loader.load()
    return data


# Split file into chunks
def split_into_chunks(data, chunk_size: int = 500, chunk_overlap: int = 0) -> list[any]: 
    # 這個是 split_into_chunks 函數，用於生成 split_into_chunks
    """
    將載入文件先合併再依照大小和重疊部分切分成區塊(切片)
    parameters:
        data: 傳入文件
        chunk_size: 切片大小,預設500
        chunk_overlap: 切片之間重疊部分大小,預設0
    returns:
        list
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function=len) 
    # 使用 RecursiveCharacterTextSplitter 切分文件
    chunks = splitter.split_documents(data) # 使用 RecursiveCharacterTextSplitter 切分文件
    return chunks # 返回切分後的文件
