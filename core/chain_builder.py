from operator import itemgetter 
# 這個是 Python 的 operator 模組，用於生成 itemgetter
from langchain_core.output_parsers import StrOutputParser 
# 這個是 LangChain 的 StrOutputParser，用於生成 StrOutputParser
from core.prompt_template import prompt_template 
# 這個是 core 模組的 prompt_template，用於生成 prompt_template


# RAG chain building
def rag_chain(retriever, llm, memory=None): # 這個是 RAG chain building 函數，用於生成 RAG chain
    """
    使用 LCEL 建構 RAG Chain，包含檢索、Prompt 組裝與對話歷史記憶
    parameter:
        retriever: 由 build_retriever() 建立的檢索器
        llm: 由 load_llm() 已初始化模型
    return:
        callable chain，對話時呼叫 .invoke(query) 取得回答
        
    LCEL組合：
        1. 傳入使用者query > retriever檢索文件 > docs_to_context組成context
        2. 讓 prompt_template 按結構format，放入 context、query，並用 MessagesPlaceholder('history') 放入對話記憶
        3. 用 llm 生成結果
        4. StrOutputParser取字符串回答
    """
    
    # 取得帶有保留歷史訊息的 prompt_template
    chat_prompt = prompt_template() # 這個是 core 模組的 prompt_template，用於生成 prompt_template

    # 將檢索結果Documents變成一長段context文字
    def docs_to_context(docs): # 這個是 docs_to_context 函數，用於生成 docs_to_context
        return "\n\n".join(doc.page_content for doc in docs)

    chain = ( # 這個是 chain，用於生成 chain
        {
            "context": itemgetter('query') | retriever | docs_to_context,         # 把搜尋到的文件內容放context
            "query": itemgetter("query"),                  # 傳遞原始query
            "chat_history": itemgetter("chat_history")     # 多輪歷史
        }
        | chat_prompt                                      # 用定義好的 prompt_template 組訊息串
        | llm                                              # LLM 生成回答
        # | StrOutputParser()                                # 擷取回答純文字，如果想要結構化或彈性輸出可以註解掉
    )

    return chain # 這個是 chain，用於生成 chain