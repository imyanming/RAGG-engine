#  RAG Question Answering System

基於檢索增強生成（Retrieval-Augmented Generation, RAG）技術的智能問答系統，能夠從上傳的文檔中提取知識並回答問題。

##  功能特點

- **文檔處理**：支援 PDF、DOCX、TXT、Markdown 等多種格式
- **智能檢索**：使用向量資料庫進行語義相似度搜尋
- **AI 問答**：整合 Google Gemini API，提供準確的回答
- **對話記憶**：支援多輪對話，保持上下文理解
- **流式輸出**：支援即時流式回應，提升使用者體驗
- **友善介面**：基於 Streamlit 的現代化網頁介面

##  技術棧

- **語言模型**：Google Gemini 2.5 Flash
- **向量資料庫**：ChromaDB
- **嵌入模型**：Sentence Transformers
- **框架**：LangChain、Streamlit
- **文檔處理**：PyPDF、LangChain Document Loaders
- **Python 版本**：>= 3.13

##  安裝步驟

### 1. 克隆專案

```bash
git clone https://github.com/imyanming/RAGG-engine.git
cd RAGG-engine
```

### 2. 安裝依賴

本專案使用 `uv` 作為套件管理工具：

```bash
# 如果尚未安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安裝專案依賴
uv sync
```

或使用傳統的 pip：

```bash
pip install -r requirements.txt
```

### 3. 環境配置

複製 `.env.example` 為 `.env`，並填入以下配置：

```env
# Google Gemini API 配置
GOOGLE_API_KEY=your_google_api_key_here

# 模型配置
MODEL_NAME=gemini-2.5-flash
MODEL_PROVIDER=google

# 嵌入模型配置
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# 向量資料庫路徑
PERSIST_DIRECTORY=./knowledge_db

# 文檔處理參數
CHUNK_SIZE=500
CHUNK_OVERLAP=0
```

### 4. 取得 Google Gemini API Key

1. 前往 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 登入您的 Google 帳號
3. 創建新的 API Key
4. 將 API Key 填入 `.env` 檔案中的 `GOOGLE_API_KEY`

## 使用方法

### 快速啟動

使用提供的啟動腳本：

```bash
chmod +x start.sh
./start.sh
```

或直接使用 Streamlit：

```bash
streamlit run streamlit_demo.py
```

啟動後，瀏覽器會自動開啟 `http://localhost:8501`

### 使用流程

1. **上傳文檔**
   - 在側邊欄的「Document Management」區域上傳 PDF、DOCX、TXT 或 Markdown 檔案
   - 支援最大 200MB 的檔案

2. **設定參數**（可選）
   - **Chunk Size**：文字分塊大小（預設 500）
   - **Chunk Overlap**：分塊重疊字元數（預設 0）
   - **Top K**：檢索最相關的文檔片段數量（預設 3）

3. **處理文檔**
   - 點擊「Add Data」按鈕
   - 系統會自動進行文檔載入、分塊、向量化等處理
   - 處理完成後會顯示文檔塊數量、分塊大小和 Token 數量

4. **開始問答**
   - 在對話框中輸入問題
   - 系統會從知識庫中檢索相關資訊並生成回答
   - 支援多輪對話，系統會記住之前的對話內容

### 輸出模式

- **Normal Output**：一次性顯示完整回答
- **Streaming Output**：即時流式顯示回答內容

## 專案結構

```
RAGG-engine/
├── core/                          # 核心功能模組
│   ├── __init__.py               # 模組初始化與配置
│   ├── data_processing.py        # 文檔載入與分塊處理
│   ├── embedding_and_vector_db.py # 嵌入與向量資料庫操作
│   ├── llm_model.py              # 語言模型載入
│   ├── chain_builder.py          # RAG 鏈構建
│   └── prompt_template.py        # 提示詞模板
├── docs/                          # 文檔與資源
│   ├── assets/                   # 圖片資源
│   └── llm_test.pdf             # 測試文檔
├── knowledge_db/                  # 向量資料庫儲存目錄
├── streamlit_demo.py             # Streamlit 主程式
├── start.sh                      # 快速啟動腳本
├── pyproject.toml                # 專案配置與依賴
├── .gitignore                    # Git 忽略檔案
└── README.md                     # 專案說明文件
```

## 核心模組說明

### `core/data_processing.py`
- `load_document()`：載入各種格式的文檔
- `split_into_chunks()`：將文檔分割成固定大小的塊

### `core/embedding_and_vector_db.py`
- `get_embedding()`：獲取嵌入模型
- `create_embedding_chroma()`：創建 ChromaDB 向量資料庫
- `build_retriever()`：構建檢索器

### `core/llm_model.py`
- `load_llm()`：載入語言模型（支援 Google Gemini）

### `core/chain_builder.py`
- `rag_chain()`：構建 RAG 問答鏈

## 注意事項

### API 配額限制

Google Gemini API 免費版有每日請求限制（約 20 次請求）。如果遇到配額限制：

1. 等待一段時間後重試
2. 前往 [Google Cloud Console](https://console.cloud.google.com/) 設定付費方案以獲得更高配額
3. 等待配額重置（每日重置）

### 環境變數

- 確保 `.env` 檔案已正確配置
- 不要將 `.env` 檔案提交到 Git 倉庫（已加入 `.gitignore`）

### 向量資料庫

- 向量資料庫儲存在 `knowledge_db/` 目錄
- 上傳新文檔時會自動清理舊的向量資料庫
- 如需手動清理，可刪除 `knowledge_db/` 目錄

## 進階配置

### 自訂嵌入模型

在 `.env` 中修改 `EMBEDDING_MODEL_NAME`：

```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

### 調整分塊參數

根據文檔類型調整分塊大小和重疊：

- **長文檔**：建議使用較大的 `CHUNK_SIZE`（如 1000）
- **技術文檔**：建議使用 `CHUNK_OVERLAP`（如 50-100）以保持上下文連貫性

## 授權

本專案採用 MIT 授權條款。

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 聯絡方式

如有問題或建議，請透過 GitHub Issues 聯繫。

---

**享受使用 RAG 問答系統！**
