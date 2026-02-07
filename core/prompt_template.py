from langchain.prompts.chat import ( 
    # 這個是 LangChain 的 ChatPromptTemplate，用於生成 Chat Prompt template
    
    ChatPromptTemplate, # 這個是 LangChain 的 ChatPromptTemplate，用於生成 Chat Prompt template
    SystemMessagePromptTemplate, # 這個是 LangChain 的 SystemMessagePromptTemplate，用於生成 System Prompt template
    HumanMessagePromptTemplate, # 這個是 LangChain 的 HumanMessagePromptTemplate，用於生成 Human Prompt template
    MessagesPlaceholder, # 這個是 LangChain 的 MessagesPlaceholder，用於生成 Messages Placeholder
)

# Prompt template
def prompt_template() -> ChatPromptTemplate: 
    # 這個是 LangChain 的 ChatPromptTemplate，用於生成 Chat Prompt template
    system_prompt = SystemMessagePromptTemplate.from_template( 
    # 這個是 LangChain 的 SystemMessagePromptTemplate，用於生成 System Prompt template
        """
        You are a powerful and helpful AI assistant and analyst. Your task is to provide comprehensive and accurate answers based on the provided "Document Content" and "Chat History".

        **CRITICAL LANGUAGE RULE**: 
        - You MUST reply in the EXACT SAME LANGUAGE as the user's question.
        - If the user asks in Chinese, reply in Chinese.
        - If the user asks in English, reply in English.
        - If the user asks in Japanese, reply in Japanese.
        - Always match the user's language automatically.

        **Behavior Guidelines**:
        1. **Comprehensive Analysis**: Your answers must reference both "Document Content" and "Chat History" to provide the most complete and coherent responses.
        2. **Complete Answers**: Generate complete, logical, and natural-sounding responses. Avoid fragmented short sentences.
        3. **Stay True to Source**: All answers must be based on the provided "Document Content". Do not make up information.
        4. **Admit Unknown**: If you cannot find relevant information in "Document Content" and "Chat History" to answer the question, clearly state that you cannot answer based on the available information. Never fabricate answers.
        5. **Language Consistency**: Always use the same language as the user's question for your reply.
        """
    )
    human_prompt = HumanMessagePromptTemplate.from_template( 
    # 這個是 LangChain 的 HumanMessagePromptTemplate，用於生成 Human Prompt template
        '''
        ---
        **Document Content:**
        {context}
        ---
        **Chat History:**
        (Previous conversation between user and you)
        ---
        **User's Current Question:**
        {query}
        ---
        '''
    )# 這個是 LangChain 的 HumanMessagePromptTemplate，用於生成 Human Prompt template
    
    chat_prompt = ChatPromptTemplate.from_messages([ 
    # 這個是 LangChain 的 ChatPromptTemplate，用於生成 Chat Prompt template
    
        system_prompt, 
        # 這個是 LangChain 的 SystemMessagePromptTemplate，用於生成 System Prompt template
        MessagesPlaceholder(variable_name='chat_history'), 
        # 這個是 LangChain 的 MessagesPlaceholder，用於生成 Messages Placeholder
        human_prompt, 
        # 這個是 LangChain 的 HumanMessagePromptTemplate，用於生成 Human Prompt template
    ])
    return chat_prompt # 這個是 LangChain 的 ChatPromptTemplate，用於生成 Chat Prompt template