# 聊天機器人系統專案

**專案類型**: 對話系統 - Transformer 生成式聊天機器人
**難度**: ⭐⭐⭐⭐ 進階
**預計時間**: 4-5 小時
**技術棧**: Hugging Face, DialoGPT, Blenderbot, Streamlit

---

## 📋 專案概述

本專案實作**生產級聊天機器人系統**,基於 Microsoft DialoGPT 和 Meta Blenderbot,展示:

1. 多輪對話管理
2. 對話歷史追蹤
3. 回覆品質控制
4. 情緒感知對話
5. Streamlit 互動介面部署

**商業應用**:
- 客戶服務機器人
- 智能FAQ助手
- 教學陪伴機器人
- 心理諮詢助手

---

## 🎯 學習目標

- ✅ 理解生成式對話系統原理
- ✅ 掌握 DialoGPT/Blenderbot 使用
- ✅ 實作多輪對話管理
- ✅ 構建互動式聊天介面
- ✅ 部署生產級聊天機器人

---

## 📊 專案特色

### 核心功能

1. **多模型支持**
   - DialoGPT (Microsoft) - 快速響應
   - Blenderbot (Meta) - 知識豐富

2. **對話管理**
   - 多輪對話追蹤
   - 歷史長度限制
   - Context 管理

3. **品質控制**
   - 回覆過濾 (長度、重複、不當內容)
   - 重試機制
   - 備用回覆

4. **進階功能**
   - 情緒識別
   - 意圖分類
   - 用戶資訊記憶
   - 個性化回覆

5. **生產部署**
   - Streamlit Web 介面
   - 模型快取優化
   - 對話統計儀表板

---

## 🚀 快速開始

### 安裝依賴

```bash
# 核心套件
poetry add transformers torch accelerate

# Web 介面 (選用)
poetry add streamlit

# 情緒識別 (選用)
poetry add emoji
```

### 運行 Notebook

```bash
# 啟動 Jupyter
poetry run jupyter notebook

# 開啟 notebook
專案_聊天機器人_Hugging_Face對話模型.ipynb
```

### 運行 Streamlit App

```bash
# 運行 chatbot_app.py
poetry run streamlit run chatbot_app.py

# 瀏覽器開啟: http://localhost:8501
```

---

## 🧮 技術架構

### 模型選擇對比

| 模型 | 參數量 | 速度 | 知識 | 適用場景 |
|------|--------|------|------|----------|
| DialoGPT-small | 117M | ⚡⚡⚡ | ⭐⭐ | 快速原型 |
| DialoGPT-medium | 345M | ⚡⚡ | ⭐⭐⭐ | **推薦** |
| DialoGPT-large | 762M | ⚡ | ⭐⭐⭐⭐ | 高品質對話 |
| Blenderbot-400M | 400M | ⚡⚡ | ⭐⭐⭐⭐⭐ | 知識問答 |

### 生成參數調優

```python
# 控制回覆品質的關鍵參數
model.generate(
    input_ids,
    max_length=1000,          # 最大長度
    temperature=0.7,          # 創造性 (0.1-1.5)
    top_k=50,                 # Top-K 採樣
    top_p=0.95,               # Nucleus 採樣
    do_sample=True,           # 啟用採樣
    no_repeat_ngram_size=3,   # 避免重複
    repetition_penalty=1.2    # 重複懲罰
)
```

**參數說明**:
- `temperature`: 越高越創意,越低越保守
- `top_k`: 只從最可能的 k 個詞中選
- `top_p`: 累積機率達 p 的詞集合
- `no_repeat_ngram_size`: 避免 n-gram 重複

---

## 🎨 預期效果

### 對話範例 1: 一般閒聊

```
User: Hi! How are you today?
Bot: I'm doing great, thanks for asking! How about you?

User: I'm good. What's your favorite movie?
Bot: I really enjoy science fiction films. Have you seen Interstellar?

User: Yes! It's amazing. What did you like about it?
Bot: The way it explores time and space is fascinating. The soundtrack is incredible too!
```

### 對話範例 2: 知識問答

```
User: What is natural language processing?
Bot: NLP is a field of AI that focuses on the interaction between computers
     and human language. It enables machines to understand, interpret, and
     generate human language.

User: What are some applications?
Bot: Common applications include chatbots like me, machine translation,
     sentiment analysis, and text summarization.
```

### 對話範例 3: 情緒感知

```
User: I just lost my job...
Bot: I'm sorry to hear that. Losing a job can be really tough. Remember
     that this is just a temporary setback. What skills do you have that
     you're proud of?

User: I'm good at programming.
Bot: That's wonderful! Programming skills are highly valued. Have you
     considered freelancing or applying to tech companies?
```

---

## 🔧 常見問題

### Q1: 模型下載失敗

```python
# 使用鏡像站 (中國大陸)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 或手動下載後本地加載
model = AutoModelForCausalLM.from_pretrained("./local_model")
```

### Q2: 記憶體不足

```python
# 使用更小的模型
model_name = \"microsoft/DialoGPT-small\"  # 117M 參數

# 或使用量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True  # 需要 bitsandbytes
)
```

### Q3: 回覆太慢

```python
# 使用 GPU
device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
model = model.to(device)

# 減少生成長度
model.generate(..., max_length=100)  # 縮短

# 使用 ONNX 優化
from optimum.onnxruntime import ORTModelForCausalLM
onnx_model = ORTModelForCausalLM.from_pretrained(model_name, export=True)
```

### Q4: 回覆不夠相關

```python
# 調整生成參數
model.generate(
    input_ids,
    temperature=0.5,          # 降低,更保守
    top_p=0.9,                # 降低
    repetition_penalty=1.5    # 增加,避免重複
)

# 或使用 Beam Search
model.generate(
    input_ids,
    num_beams=5,              # Beam Search
    early_stopping=True,
    do_sample=False           # 關閉採樣
)
```

---

## 📈 進階優化

### 1. 整合檢索增強生成 (RAG)

```python
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 建立知識庫
knowledge_base = [
    \"Python is a programming language.\",
    \"NLP stands for Natural Language Processing.\",
    # ... 更多知識
]

# 向量化知識庫
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(knowledge_base, embeddings)

def rag_enhanced_chat(user_input):
    # 1. 檢索相關知識
    relevant_docs = vectorstore.similarity_search(user_input, k=3)

    # 2. 構建增強 prompt
    context = \"\\n\".join([doc.page_content for doc in relevant_docs])
    enhanced_prompt = f\"Context: {context}\\n\\nUser: {user_input}\\nBot:\"\n",
    "\n",
    "    # 3. 生成回覆
    return bot.chat(enhanced_prompt)
```

### 2. 對話評估指標

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge\n",
    "\n",
    "def evaluate_dialogue(references, hypotheses):\n",
    "    \"\"\"\n",
    "    評估對話品質\n",
    "    \"\"\"\n",
    "    # BLEU Score\n",
    "    bleu = sentence_bleu([ref.split() for ref in references], hypotheses.split())\n",
    "\n",
    "    # ROUGE Score\n",
    "    rouge = Rouge()\n",
    "    scores = rouge.get_scores(hypotheses, references[0])\n",
    "\n",
    "    return {\n",
    "        'bleu': bleu,\n",
    "        'rouge-1': scores[0]['rouge-1']['f'],\n",
    "        'rouge-l': scores[0]['rouge-l']['f']\n",
    "    }\n",
    "```

---

## 🏆 作品集展示建議

### 展示要點

1. **技術深度**\n",
    "   - \"實作基於 Transformer 的對話系統\"\n",
    "   - \"使用 DialoGPT 處理多輪對話\"\n",
    "   - \"整合情緒識別提升用戶體驗\"\n",
    "\n",
    "2. **商業價值**\n",
    "   - \"可應用於客服自動化,降低 80% 人力成本\"\n",
    "   - \"支援 24/7 不間斷服務\"\n",
    "\n",
    "3. **技術亮點**\n",
    "   - 多輪對話管理\n",
    "   - 回覆品質控制\n",
    "   - 生產級部署 (Streamlit)\n",
    "\n",
    "### GitHub README 建議\n",
    "\n",
    "```markdown\n",
    "# AI 聊天機器人\n",
    "\n",
    "## Demo\n",
    "[Live Demo](link) | [Video](link)\n",
    "\n",
    "## 技術棧\n",
    "- DialoGPT (Microsoft)\n",
    "- Transformers\n",
    "- Streamlit\n",
    "\n",
    "## 主要功能\n",
    "- ✅ 多輪對話\n",
    "- ✅ 情緒感知\n",
    "- ✅ Web 介面\n",
    "\n",
    "## 運行方式\n",
    "```bash\n",
    "pip install -r requirements.txt\n",
    "streamlit run chatbot_app.py\n",
    "```\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "**專案版本**: v1.0\n",
    "**Notebooks**: 1\n",
    "**最後更新**: 2025-10-17\n",
    "**維護者**: iSpan NLP Team\n",
    "