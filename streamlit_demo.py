import os # 這個是 Python 的 os 模組，用於生成 os
import streamlit as st
import shutil # 這個是 Python 的 shutil 模組，用於生成 shutil
from transformers import AutoTokenizer
from langchain_core.output_parsers import StrOutputParser # 這個是 LangChain 的 StrOutputParser，用於生成 StrOutputParser
from langchain.memory import ConversationBufferMemory # 這個是 LangChain 的 ConversationBufferMemory，用於生成 ConversationBufferMemory
from core import ( # 這個是 core 模組的 Config、load_document、split_into_chunks、get_embedding、create_embedding_chroma、build_retriever、load_llm、rag_chain
    Config, 
    load_document, 
    split_into_chunks, 
    get_embedding,
    create_embedding_chroma,
    build_retriever,
    load_llm,
    rag_chain
)

# intialize session state
if 'messages' not in st.session_state: # 如果 'messages' 不在 st.session_state 中，則初始化 st.session_state.messages
    st.session_state.messages = []
if 'chat_memory' not in st.session_state: # 如果 'chat_memory' 不在 st.session_state 中，則初始化 st.session_state.chat_memory
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # 使用 ConversationBufferMemory 初始化 st.session_state.chat_memory
if 'vs' not in st.session_state: # 如果 'vs' 不在 st.session_state 中，則初始化 st.session_state.vs
    st.session_state.vs = None # 初始化 st.session_state.vs

# calculate embedding cost using tikoken
def calculate_embedding_token(embedding_model_name: str, texts): # 這個是 calculate_embedding_token 函數，用於生成 calculate_embedding_token
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name) # 使用 AutoTokenizer 初始化 tokenizer
    total_tokens = sum(len(tokenizer.encode(page.page_content)) for page in texts) # 使用 tokenizer 編碼 texts
    return total_tokens # 返回編碼後的 token 數量

# remove history（memory, messages, vector store）
def clear_history(): # 這個是 clear_history 函數，用於生成 clear_history
    st.session_state.messages = [] # 清空 st.session_state.messages
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # 使用 ConversationBufferMemory 初始化 st.session_state.chat_memory
    # Properly delete vector store
    if st.session_state.vs is not None: # 如果 st.session_state.vs 不是 None，則清空 st.session_state.vs
        try:
            del st.session_state.vs # 清空 st.session_state.vs
        except:
            pass # 如果清空 st.session_state.vs 失敗，則不進行任何操作
    st.session_state.vs = None
    
# clear_chroma_db
def clear_chroma_db(path='./knowledge_db'): # 這個是 clear_chroma_db 函數，用於生成 clear_chroma_db
    """清空 Chroma 本地嵌入資料資料庫目錄"""
    if os.path.isdir(path): # 如果 path 是目錄，則清空 path
        shutil.rmtree(path) # 清空 path
        print(f"Chroma 向量資料庫位於 {path} 已被清空。") # 如果 path 是目錄，則清空 path
    else: # 如果 path 不是目錄，則不進行任何操作
        print(f"找不到 {path} 目錄，不需清空。") # 如果 path 不是目錄，則不進行任何操作


# ask and get answer method (normal output)
def ask_then_get_answer(vector_store, query, top_k): # 這個是 ask_then_get_answer 函數，用於生成 ask_then_get_answer
    import time
    from google.api_core import exceptions as google_exceptions # 這個是 Google API 的 exceptions 模組，用於生成 google_exceptions
    
    try: # 如果清空 st.session_state.vs 失敗，則不進行任何操作
        llm = load_llm(model_name=Config.MODEL_NAME, model_provider=Config.MODEL_PROVIDER, temperature=1) # 使用 load_llm 初始化 llm
        retriever = build_retriever(vector_store=vector_store, top_k=top_k) # 使用 build_retriever 初始化 retriever
        memory = st.session_state.chat_memory # 使用 st.session_state.chat_memory 初始化 memory
        chain = rag_chain(retriever=retriever, llm=llm, memory=None)
        chat_input = {
            'query': query,
            'chat_history': memory.load_memory_variables({})['chat_history'] # 使用 memory.load_memory_variables 初始化 chat_input
        }
        answer = chain.invoke(chat_input) # 使用 chain.invoke 初始化 answer
        return answer # 返回 answer
    except google_exceptions.ResourceExhausted as e: # 如果 google_exceptions.ResourceExhausted 發生，則不進行任何操作
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            wait_time = 20  # Default wait 20 seconds # 默認等待 20 秒
            # Try to extract wait time from error message
            if "retry in" in error_msg.lower():
                import re
                match = re.search(r'retry in ([\d.]+)s', error_msg.lower()) # 使用 re.search 初始化 match
                if match:
                    wait_time = int(float(match.group(1))) + 5  # Add 5 seconds buffer
            
            st.error(f"""
            ⚠️ **API Quota Exceeded**
            
            Google Gemini API free tier has a daily limit of **20 requests**, and you have reached this limit.
            
            **Solutions:**
            1. ⏰ Wait approximately {wait_time} seconds and try again
            2. 💳 Go to [Google Cloud Console](https://console.cloud.google.com/) to set up billing for higher quota
            3. 🔄 Try again tomorrow (quota resets daily)
            
            **More Information:** [Gemini API Quota Guide](https://ai.google.dev/gemini-api/docs/rate-limits)
            """)
            return None
        else:
            raise
    except Exception as e: # 如果 Exception 發生，則不進行任何操作
        st.error(f"❌ An error occurred: {str(e)}") # 使用 st.error 初始化 error
        return None # 返回 None

# Streaming answer method
def ask_then_get_answer_streaming(vector_store, query, top_k):
    from google.api_core import exceptions as google_exceptions
    
    try: # 如果清空 st.session_state.vs 失敗，則不進行任何操作
        llm = load_llm(model_name=Config.MODEL_NAME, model_provider=Config.MODEL_PROVIDER, temperature=1) # 使用 load_llm 初始化 llm
        retriever = build_retriever(vector_store=vector_store, top_k=top_k) # 使用 build_retriever 初始化 retriever
        memory = st.session_state.chat_memory # 使用 st.session_state.chat_memory 初始化 memory
        chain = rag_chain(retriever=retriever, llm=llm, memory=None) # 使用 rag_chain 初始化 chain
        chat_input = {
            'query': query,
            'chat_history': memory.load_memory_variables({})['chat_history'] # 使用 memory.load_memory_variables 初始化 chat_input
        }
        
        # Stream the response
        full_response = "" # 初始化 full_response
        for chunk in chain.stream(chat_input): # 使用 chain.stream 初始化 chunk
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                yield chunk.content # 返回 chunk.content
            else:
                content = str(chunk) # 使用 str 初始化 content
                full_response += content # 使用 full_response 初始化 full_response
                yield content # 返回 content
        
        return full_response # 返回 full_response
        
    except google_exceptions.ResourceExhausted as e: # 如果 google_exceptions.ResourceExhausted 發生，則不進行任何操作
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            wait_time = 20 # 默認等待 20 秒
            if "retry in" in error_msg.lower():
                import re
                match = re.search(r'retry in ([\d.]+)s', error_msg.lower()) # 使用 re.search 初始化 match
                if match:
                    wait_time = int(float(match.group(1))) + 5 # 使用 int 初始化 wait_time
            
            st.error(f"""
            ⚠️ **API Quota Exceeded**
            
            Google Gemini API free tier has a daily limit of **20 requests**, and you have reached this limit.
            
            **Solutions:**
            1. ⏰ Wait approximately {wait_time} seconds and try again
            2. 💳 Go to [Google Cloud Console](https://console.cloud.google.com/) to set up billing for higher quota
            3. 🔄 Try again tomorrow (quota resets daily)
            
            **More Information:** [Gemini API Quota Guide](https://ai.google.dev/gemini-api/docs/rate-limits)
            """)
            return None # 返回 None
        else:
            raise # 如果 Exception 發生，則不進行任何操作
    except Exception as e: # 如果 Exception 發生，則不進行任何操作
        st.error(f"❌ An error occurred: {str(e)}") # 使用 st.error 初始化 error
        return None # 返回 None

if __name__ == "__main__": # 如果 __name__ 是 "__main__"，則執行以下程式
    
    # Page configuration
    st.set_page_config( # 使用 st.set_page_config 初始化頁面配置
        page_title="RAG Question Answering System", # 使用 page_title 初始化頁面標題
        page_icon="🤖", # 使用 page_icon 初始化頁面圖示
        layout="wide", # 使用 layout 初始化頁面布局
        initial_sidebar_state="expanded" # 使用 initial_sidebar_state 初始化頁面側邊欄狀態
    )
    
    # Main title area
    col1, col2 = st.columns([3, 1]) # 使用 st.columns 初始化 col1 和 col2
    with col1:
        st.title('🤖 RAG Question Answering System') # 使用 st.title 初始化頁面標題
        st.markdown('**Knowledge Base Q&A System based on Retrieval-Augmented Generation (RAG)**') # 使用 st.markdown 初始化頁面內容
    with col2:
        if st.session_state.vs is not None: # 如果 st.session_state.vs 不是 None，則使用 st.success 初始化頁面狀態
            st.success('✅ Vector Store Ready', icon="📚") # 使用 st.success 初始化頁面狀態
        else: # 如果 st.session_state.vs 是 None，則使用 st.info 初始化頁面狀態
            st.info('⏳ Waiting for File Upload', icon="📦") # 使用 st.info 初始化頁面狀態
    
    st.divider() # 使用 st.divider 初始化頁面分割線
    
    with st.sidebar:
        st.markdown(
            "<h3 style='text-align: center; margin-bottom: 0;'>RAG FILE SYSTEM CONSOLE</h3>", # 使用 st.markdown 初始化頁面內容
            unsafe_allow_html=True, # 使用 unsafe_allow_html 初始化頁面內容
        )
        
        st.divider() # 使用 st.divider 初始化頁面分割線
        
        # API Configuration
        with st.expander("🔑 API Configuration", expanded=True): # 使用 st.expander 初始化頁面內容  
            api_key = st.text_input(
                label='Google API Key', 
                type='password',
                help='Enter your Google Gemini API Key',
                placeholder='Enter your API Key...'
            )
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key
                st.success('✅ API Key Set')
        
        # Model Configuration
        with st.expander("⚙️ Model Settings", expanded=False):
            llm = st.selectbox(
                label='Chat Model',
                options=['gemini-2.5-flash'],
                help='Select the language model to use'
            )
            
            vector_store_option = st.selectbox(
                label='Vector Database',
                options=['Chroma'],
                help='Select vector database type'
            )
            
            output_type = st.selectbox(
                label='Output Mode', 
                options=['Normal Output', 'Streaming Output'],
                help='Select response output mode'
            )
        
        # File Upload
        with st.expander("📄 Document Management", expanded=True):
            upload_file = st.file_uploader(
                'Upload Knowledge Base Document', 
                type=['pdf', 'docx', 'txt', 'markdown'],
                help='Supports PDF, DOCX, TXT, Markdown formats, max 200MB'
            )
            
            if upload_file:
                file_size = len(upload_file.getvalue()) / 1024  # KB
                st.caption(f"📎 {upload_file.name} ({file_size:.1f} KB)")
        
        # Advanced Parameters
        with st.expander("🔧 Advanced Parameters", expanded=False):
            chunk_size = st.number_input(
                'Chunk Size', 
                min_value=100, 
                max_value=2048, 
                value=500,
                step=50,
                help='Size of text chunks for splitting',
                on_change=clear_history
            )
            
            chunk_overlap = st.number_input(
                'Chunk Overlap', 
                min_value=0, 
                max_value=512, 
                value=2,
                step=1,
                help='Number of overlapping characters between chunks',
                on_change=clear_history
            )
            
            k = st.number_input(
                'Top K', 
                min_value=1, 
                max_value=20, 
                value=3,
                step=1,
                help='Number of most relevant document chunks to retrieve',
                on_change=clear_history
            )
        
        st.divider()
        
        # Action Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            add_data = st.button('Add Data', use_container_width=True, type='primary')
        with col_btn2:
            if st.button('Clear All', use_container_width=True, on_click=clear_history):
                st.rerun()
        
        # Status Display
        if st.session_state.vs is not None:
            st.success('✅ Knowledge Base Loaded', icon="📚")
        else:
            st.info('Please upload and add a file first', icon="📝")

        # File Processing
        if upload_file and add_data:
            # Clear old vector store before processing new file
            if st.session_state.vs is not None:
                try:
                    # Try to delete the collection if it exists
                    from langchain_chroma import Chroma
                    if hasattr(st.session_state.vs, '_collection') and st.session_state.vs._collection is not None:
                        try:
                            # Delete the collection
                            st.session_state.vs._client.delete_collection(st.session_state.vs._collection.name)
                        except:
                            pass
                    # Delete the old vector store object
                    del st.session_state.vs
                except:
                    pass
                st.session_state.vs = None
            
            # Clear chat history when uploading new file
            st.session_state.messages = []
            st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load document
                status_text.info('📖 Loading document...')
                progress_bar.progress(10)
                bytes_data = upload_file.read()
                file_name = os.path.join('./', upload_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                
                # Step 2: Split text
                status_text.info('✂️ Splitting text...')
                progress_bar.progress(30)
                chunks = split_into_chunks(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                # Step 3: Calculate tokens
                status_text.info('🔢 Calculating tokens...')
                progress_bar.progress(50)
                total_tokens = calculate_embedding_token(Config.EMBEDDING_MODEL_NAME, chunks)
                
                # Step 4: Generate embeddings
                status_text.info('🧮 Generating vector embeddings...')
                progress_bar.progress(70)
                embedding = get_embedding(Config.EMBEDDING_MODEL_NAME)
                
                # Step 5: Create NEW vector store with unique collection name
                status_text.info('💾 Creating vector database...')
                progress_bar.progress(90)
                if vector_store_option == 'Chroma':
                    # Create a completely new vector store with unique collection name
                    import uuid
                    collection_name = f"doc_{uuid.uuid4().hex[:8]}"
                    vector_store_obj = create_embedding_chroma(chunks, embedding, collection_name=collection_name)
                    st.session_state.vs = vector_store_obj
                
                # Complete
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Chunks", len(chunks))
                with col2:
                    st.metric("Chunk Size", chunk_size)
                with col3:
                    st.metric("Tokens", f"{total_tokens:,}")
                
                st.success(f'✅ Document "{upload_file.name}" has been successfully processed and loaded into the knowledge base!')
                
                # Clean up temporary file
                os.remove(file_name)
                
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f'❌ Error processing file: {str(e)}')
    
    # Main chat area
    if st.session_state.vs is None:
        st.info('💡 **Tip**: Please upload a file in the sidebar and click "Add Data" to initialize the knowledge base', icon='📦')
        st.markdown("""
        ### 📋 Usage Steps:
        1. **Upload Document**: Select a PDF, DOCX, TXT, or Markdown file in the sidebar
        2. **Set Parameters**: Adjust Chunk Size, Overlap, and Top K (optional)
        3. **Add Data**: Click the "Add Data" button and wait for processing to complete
        4. **Start Chatting**: Enter your question in the input box below to start chatting with the knowledge base
        """)    
    
    # Chat history display
    if st.session_state.messages:
        st.markdown("### 💬 Chat History")
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
    else:
        # Welcome message
        st.markdown("### 👋 Welcome to RAG Question Answering System")
        st.markdown("""
        This is an intelligent Q&A system based on **Retrieval-Augmented Generation (RAG)** technology that can:
        
        - 📚 **Knowledge Retrieval**: Accurately retrieve relevant information from your uploaded documents
        - 🤖 **Intelligent Answers**: Generate accurate, well-grounded answers using large language models
        - 💭 **Context Memory**: Support multi-turn conversations with context understanding
        
        **Get Started**: Enter your question in the input box below 👇
        """)
            
    # User input
    prompt = st.chat_input(placeholder='Enter your question...')
    if prompt:
        vector_store = st.session_state.vs   # <--- Must be here
        if vector_store is None:
            st.warning('Vector store not initialized. Please upload and embed a file first!')
            st.stop()
        # User input
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Assistant response
        response = None
        response_content = ""
        
        if output_type == 'Normal Output':
            if llm == 'gemini-2.5-flash':
                response = ask_then_get_answer(vector_store, prompt, k)
            else:
                response = "Please select a valid model."
            
            if response is not None:
                with st.chat_message('assistant'):
                    if hasattr(response, 'content'):
                        st.markdown(response.content)
                        response_content = response.content
                    else:
                        st.markdown(str(response))
                        response_content = str(response)
                st.session_state.messages.append({'role': 'assistant', 'content': response_content})

                # Sync conversation to LangChain memory
                st.session_state.chat_memory.save_context(
                    {"input": prompt},
                    {"output": response_content}
                )
        
        elif output_type == 'Streaming Output':
            if llm == 'gemini-2.5-flash':
                with st.chat_message('assistant'):
                    # Create a placeholder for streaming text
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    try:
                        for chunk in ask_then_get_answer_streaming(vector_store, prompt, k):
                            if chunk:
                                full_response += chunk
                                # Update the placeholder with the current response
                                message_placeholder.markdown(full_response + "▌")
                        
                        # Final update without cursor
                        message_placeholder.markdown(full_response)
                        response_content = full_response
                        
                    except Exception as e:
                        st.error(f"❌ Streaming error: {str(e)}")
                        response_content = ""
                
                if response_content:
                    st.session_state.messages.append({'role': 'assistant', 'content': response_content})
                    
                    # Sync conversation to LangChain memory
                    st.session_state.chat_memory.save_context(
                        {"input": prompt},
                        {"output": response_content}
                    )
            else:
                with st.chat_message('assistant'):
                    st.markdown("Please select a valid model.")
                st.session_state.messages.append({'role': 'assistant', 'content': "Please select a valid model."})