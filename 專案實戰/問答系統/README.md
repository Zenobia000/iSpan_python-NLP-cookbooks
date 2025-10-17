# 智能問答系統 (RAG)

**專案類型**: 檢索增強生成 (Retrieval-Augmented Generation)
**難度**: ⭐⭐⭐⭐⭐ 專家級
**預計時間**: 5-6 小時
**技術棧**: RAG, FAISS, LangChain, Sentence Transformers

---

## 📋 專案概述

本專案實作**生產級智能問答系統**,基於 RAG (檢索增強生成) 架構,展示:

1. 向量數據庫構建 (FAISS)
2. 語義檢索技術
3. 檢索與生成整合
4. 混合檢索優化
5. FastAPI 服務部署

**商業應用**:
- 企業知識管理系統
- 智能客服助手
- 文檔搜尋引擎
- 法律/醫療問答系統

---

## 🎯 核心技術

### RAG 架構優勢

| 優勢 | 說明 | 商業價值 |
|------|------|----------|
| **動態知識** | 無需重新訓練模型 | 降低維護成本 |
| **可追溯性** | 引用具體來源 | 提升可信度 |
| **可擴展性** | 輕鬆添加新知識 | 快速適應業務變化 |
| **準確性** | 基於實際文檔回答 | 減少幻覺 (Hallucination) |

### 技術棧說明

| 組件 | 技術 | 作用 |
|------|------|------|
| **Embedding** | Sentence-BERT | 文本向量化 |
| **Vector DB** | FAISS | 高效相似度檢索 |
| **QA Model** | DistilBERT-SQuAD | 答案抽取 |
| **Orchestration** | LangChain | 流程編排 |
| **Deployment** | FastAPI | API 服務 |

---

## 🚀 快速開始

### 安裝依賴

```bash
# 核心套件
poetry add transformers sentence-transformers faiss-cpu

# LangChain 生態
poetry add langchain langchain-community

# API 服務 (選用)
poetry add fastapi uvicorn

# 網頁爬蟲 (選用)
poetry add beautifulsoup4 requests
```

### 運行 Notebook

```bash
poetry run jupyter notebook
# 開啟: 專案_問答系統_RAG檢索增強生成.ipynb
```

### 運行 API 服務

```bash
# 啟動 FastAPI
poetry run uvicorn qa_api:app --reload

# 訪問 API 文檔: http://localhost:8000/docs
```

---

## 🧮 核心流程

### 階段 1: 知識庫構建

```python
# 1. 準備文檔
documents = [
    "Document 1 content...",
    "Document 2 content...",
    # ...
]

# 2. 向量化
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# 3. 建立索引
import faiss

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

### 階段 2: 檢索

```python
# 查詢向量化
query = "What is BERT?"
query_embedding = model.encode([query])

# 檢索 Top-K
distances, indices = index.search(query_embedding, k=3)

# 獲取相關文檔
relevant_docs = [documents[i] for i in indices[0]]
```

### 階段 3: 答案生成

```python
from transformers import pipeline

qa = pipeline("question-answering")

# 合併檢索文檔作為 context
context = " ".join(relevant_docs)

# 抽取答案
answer = qa(question=query, context=context)
```

---

## 📁 知識庫來源

### 支援的文檔格式

- **純文本**: .txt
- **Markdown**: .md
- **PDF**: 使用 PyPDF2 解析
- **網頁**: BeautifulSoup 爬取
- **JSON/CSV**: 結構化數據

### 載入範例

```python
# 從目錄載入所有 .md 文檔
from pathlib import Path

docs_dir = Path("../../docs")
md_files = docs_dir.glob("*.md")

documents = []
for file in md_files:
    with open(file, 'r', encoding='utf-8') as f:
        documents.append(f.read())
```

---

## 🎨 預期效果

### 問答範例

```
Q: What is Natural Language Processing?
A: Natural Language Processing (NLP) is a field of AI that focuses on
   the interaction between computers and human language.
   Confidence: 95.7%
   Source: [Document 1]

Q: Who developed BERT?
A: Google
   Confidence: 92.3%
   Source: [Document 3]

Q: Explain the attention mechanism.
A: The attention mechanism allows models to focus on relevant parts of
   the input when generating output.
   Confidence: 88.9%
   Source: [Document 9]
```

### 性能指標

- **檢索速度**: < 50ms (FAISS)
- **問答延遲**: < 200ms (DistilBERT)
- **準確率**: 85-95% (取決於知識庫品質)
- **覆蓋率**: 取決於知識庫完整性

---

## 🔧 常見問題

### Q1: FAISS 安裝失敗

```bash
# CPU 版本
pip install faiss-cpu

# GPU 版本 (需要 CUDA)
pip install faiss-gpu
```

### Q2: 檢索結果不相關

```python
# 調整檢索參數
results = qa_system.retrieve(question, top_k=5)  # 增加 top_k

# 使用混合檢索
hybrid_retriever = HybridRetriever(
    documents,
    semantic_weight=0.7  # 調整權重
)
```

### Q3: 答案信心度低

```python
# 增加檢索文檔數
top_k = 5  # 從 3 增加到 5

# 使用重排序
reranked = rerank_documents(query, retrieved_docs, top_k=3)

# 調整信心度閾值
min_confidence = 0.2  # 降低閾值
```

---

## 📈 進階優化

### 1. 使用 Chroma 向量數據庫

```python
from langchain.vectorstores import Chroma

# 建立持久化向量數據庫
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory=\"./chroma_db\"
)

# 持久化
vectorstore.persist()

# 載入
vectorstore = Chroma(
    persist_directory=\"./chroma_db\",
    embedding_function=embeddings
)
```

### 2. 整合 LLaMA/GPT 生成

```python
# 使用 OpenAI GPT (需要 API key)
from langchain.llms import OpenAI

llm = OpenAI(temperature=0, openai_api_key=\"your-key\")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
```

### 3. 對話式問答

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 帶記憶的對話式問答
memory = ConversationBufferMemory(\n",
    "    memory_key=\"chat_history\",\n",
    "    return_messages=True\n",
    ")\n",
    "\n",
    "conversational_qa = ConversationalRetrievalChain.from_llm(\n",
    "    llm=llm,\n",
    "    retriever=vectorstore.as_retriever(),\n",
    "    memory=memory\n",
    ")\n",
    "\n",
    "# 多輪對話\n",
    "conversational_qa({\"question\": \"What is BERT?\"})\n",
    "conversational_qa({\"question\": \"Who developed it?\"})  # 理解 \"it\" 指 BERT\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "## 🏆 作品集展示\n",
    "\n",
    "### 技術亮點\n",
    "\n",
    "1. **前沿技術**: \"實作 RAG 架構,結合檢索與生成\"\n",
    "2. **高效檢索**: \"使用 FAISS 實現毫秒級語義檢索\"\n",
    "3. **可擴展**: \"支援動態知識庫更新,無需重新訓練\"\n",
    "4. **生產就緒**: \"FastAPI 服務,完整 API 文檔\"\n",
    "\n",
    "### Demo 建議\n",
    "\n",
    "- 展示不同領域的問答 (技術、常識、特定領域)\n",
    "- 對比有/無 RAG 的回答品質\n",
    "- 展示即時添加新知識的能力\n",
    "- 性能基準測試報告\n",
    "\n",
    "---\n",
    "\n",
    "**專案版本**: v1.0\n",
    "**Notebooks**: 1\n",
    "**最後更新**: 2025-10-17\n",
    "**維護者**: iSpan NLP Team\n",
    "**授權**: MIT License\n",
    "