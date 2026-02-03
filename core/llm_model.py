from dotenv import load_dotenv # 這個是 Python 的 dotenv 模組，用於生成 load_dotenv

# Loading environment variable in .env file 
load_dotenv(verbose=True, override=True)

from langchain.chat_models import init_chat_model # 這個是 LangChain 的 chat_models 模組，用於生成 init_chat_model

# Load chat model
def load_llm(model_name: str ="gemini-2.5-flash", model_provider: str|None ="google_genai", temperature: int =0): # 這個是 load_llm 函數，用於生成 load_llm
    """
    載入LLM模型，可根據需求替換不同Provider與模型名稱
    parameters:
        model_name: 模型名稱或API KEY (如 "gpt-3.5-turbo", "gemini-2.5-flash"等)
        provider: 廠商 (如 "openai", "google_genai"等)
        temperature: 溫度，控制回答多樣性
    return:
        model: LLM模型物件，可呼叫 .invoke(prompt)
    """
    model = init_chat_model(model_name, model_provider=model_provider, temperature=temperature)
    # 使用 init_chat_model 初始化 LLM 模型
    return model # 返回 LLM 模型物件