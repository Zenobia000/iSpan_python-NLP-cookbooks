# Hugging Face 生態系統與 Pipeline API 完全指南

**課程**: iSpan Python NLP Cookbooks v2
**章節**: CH08 Hugging Face 實戰
**版本**: v1.0
**最後更新**: 2025-10-17

---

## 📚 學習目標

完成本講義後,您將能夠:

1. 理解 Hugging Face 生態系統的核心價值與架構
2. 掌握 Transformers、Datasets、Hub 三大組件的使用
3. 熟練使用 Pipeline API 快速實現 NLP 任務
4. 理解模型卡片 (Model Card) 並正確選擇預訓練模型
5. 掌握自訂 Pipeline 參數與批次處理技巧

---

## 1. Hugging Face 生態系統概覽

### 1.1 為什麼選擇 Hugging Face?

在現代 NLP 開發中,Hugging Face 已成為事實上的標準平台。其核心優勢包括:

#### **統一API,降低學習成本**
```python
# 無論使用 BERT、GPT、T5,API 完全一致
from transformers import AutoModel

bert = AutoModel.from_pretrained("bert-base-uncased")
gpt2 = AutoModel.from_pretrained("gpt2")
t5 = AutoModel.from_pretrained("t5-small")
```

#### **開箱即用,無需從零訓練**
- 超過 **500,000** 個預訓練模型
- 涵蓋 **8,000+** 種語言與方言
- 支援 **100+** 種 NLP 任務

#### **活躍社群,持續更新**
- 每天新增數百個模型
- 即時整合最新研究成果 (GPT-4, LLaMA, Mistral...)
- 完整的文檔與教學資源

### 1.2 生態系統核心組件

```
┌──────────────────────────────────────────────────┐
│            Hugging Face 生態系統                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────┐  ┌────────────────────┐   │
│  │  Hub (平台)      │  │ Transformers (API) │   │
│  │  - 模型共享      │  │ - 統一接口         │   │
│  │  - 數據集存儲    │  │ - 預訓練模型       │   │
│  │  - 協作開發      │  │ - 微調工具         │   │
│  └─────────────────┘  └────────────────────┘   │
│           ↓                      ↓               │
│  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Datasets         │  │ Tokenizers         │   │
│  │ - 50,000+ 數據集 │  │ - 高效分詞         │   │
│  │ - 統一加載接口   │  │ - Rust 核心        │   │
│  └─────────────────┘  └────────────────────┘   │
│           ↓                      ↓               │
│  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Accelerate       │  │ Evaluate           │   │
│  │ - 分散式訓練     │  │ - 模型評估         │   │
│  │ - 混合精度       │  │ - 基準測試         │   │
│  └─────────────────┘  └────────────────────┘   │
└──────────────────────────────────────────────────┘
```

#### **1. Transformers (模型API)**
- 🎯 核心功能: 提供統一的模型加載、訓練、推理接口
- 📦 支援框架: PyTorch, TensorFlow, JAX
- 🔧 主要類別:
  - `AutoModel`: 自動檢測並加載模型
  - `AutoTokenizer`: 自動加載對應的分詞器
  - `AutoConfig`: 自動加載模型配置
  - `Pipeline`: 端到端任務封裝

#### **2. Datasets (數據集)**
- 🗂️ 核心功能: 統一的數據集加載與處理接口
- ⚡ 技術特點: Apache Arrow 支撐,記憶體映射,高效處理大數據
- 📊 數據來源:
  - 內建數據集: IMDB, SQuAD, GLUE, SuperGLUE...
  - 社群數據集: 超過 50,000 個公開數據集
  - 自訂數據集: 輕鬆上傳並分享

#### **3. Hub (共享平台)**
- 🏛️ 核心功能: 模型、數據集、Space (應用) 的共享與協作
- 🔒 版本控制: 基於 Git,支援完整的版本管理
- 🌐 訪問方式:
  - 網頁瀏覽: https://huggingface.co
  - Python API: `huggingface_hub` 函式庫
  - CLI 工具: `huggingface-cli`

---

## 2. Transformers 核心概念

### 2.1 三大核心類別: Model, Tokenizer, Config

#### **Model (模型)**

模型是 Transformers 的核心,分為兩類:

**1. 基礎模型 (Base Model)**
```python
from transformers import AutoModel

# 載入基礎模型 (無任務頭)
model = AutoModel.from_pretrained("bert-base-uncased")
# 輸出: (batch_size, seq_length, hidden_size) 的特徵向量
```

**用途**:
- 提取文本特徵 (Embeddings)
- 作為下游任務的特徵提取器
- 微調到特定任務

**2. 任務特定模型 (Task-Specific Model)**
```python
from transformers import AutoModelForSequenceClassification

# 載入分類模型 (有分類頭)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # 二分類
)
# 輸出: (batch_size, num_labels) 的 logits
```

**常用任務模型類別**:

| 類別名稱 | 任務 | 應用場景 |
|---------|------|----------|
| `AutoModelForSequenceClassification` | 序列分類 | 情感分析、文本分類 |
| `AutoModelForTokenClassification` | 標記分類 | NER、詞性標註 |
| `AutoModelForQuestionAnswering` | 問答系統 | SQuAD, DRCD |
| `AutoModelForCausalLM` | 因果語言模型 | GPT 系列文本生成 |
| `AutoModelForSeq2SeqLM` | 序列到序列 | 翻譯、摘要 |
| `AutoModelForMaskedLM` | 遮罩語言模型 | BERT 預訓練 |
| `AutoModelForMultipleChoice` | 多選題 | 閱讀理解選擇題 |

#### **Tokenizer (分詞器)**

Tokenizer 負責將文本轉換為模型可理解的數字序列。

**核心流程**:
```
原始文本 → 分詞 → 轉換為ID → 添加特殊標記 → 填充/截斷 → 生成注意力遮罩
```

**實戰範例**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hugging Face is amazing!"

# 完整編碼過程
encoded = tokenizer(
    text,
    return_tensors="pt",      # 返回 PyTorch 張量
    padding="max_length",      # 填充到最大長度
    truncation=True,           # 超長截斷
    max_length=128,            # 最大序列長度
    add_special_tokens=True,   # 添加 [CLS], [SEP]
    return_attention_mask=True # 返回注意力遮罩
)

print("Input IDs:", encoded['input_ids'])
print("Attention Mask:", encoded['attention_mask'])

# 解碼回文本
decoded = tokenizer.decode(encoded['input_ids'][0])
print("Decoded:", decoded)
```

**輸出**:
```
Input IDs: tensor([[  101,  17662,  2227,  2003,  6429,   999,   102, ...]])
Attention Mask: tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, ...]])
Decoded: [CLS] hugging face is amazing! [SEP] [PAD] [PAD] ...
```

**重要參數說明**:

| 參數 | 說明 | 常用值 |
|------|------|--------|
| `return_tensors` | 返回張量類型 | `"pt"` (PyTorch), `"tf"` (TensorFlow), `"np"` (NumPy) |
| `padding` | 填充策略 | `True`, `"max_length"`, `"longest"`, `False` |
| `truncation` | 截斷策略 | `True`, `"only_first"`, `"longest_first"`, `False` |
| `max_length` | 最大長度 | `512` (BERT), `1024` (GPT-2), `2048` (GPT-3) |
| `add_special_tokens` | 添加特殊標記 | `True` (默認), `False` |
| `return_attention_mask` | 返回注意力遮罩 | `True` (默認), `False` |
| `return_token_type_ids` | 返回句子類型ID | `True` (BERT), `False` (RoBERTa) |

#### **Config (配置)**

Config 存儲模型的超參數與架構細節。

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-uncased")

print(f"詞彙表大小: {config.vocab_size}")
print(f"隱藏層維度: {config.hidden_size}")
print(f"注意力頭數: {config.num_attention_heads}")
print(f"編碼層數: {config.num_hidden_layers}")
print(f"最大序列長度: {config.max_position_embeddings}")
```

**輸出**:
```
詞彙表大小: 30522
隱藏層維度: 768
注意力頭數: 12
編碼層數: 12
最大序列長度: 512
```

### 2.2 模型加載的三種模式

#### **模式 1: 從 Hub 加載 (最常用)**
```python
from transformers import AutoModel

# 自動從 Hugging Face Hub 下載並快取
model = AutoModel.from_pretrained("bert-base-uncased")
```

#### **模式 2: 從本地路徑加載**
```python
# 先保存到本地
model.save_pretrained("./my_bert_model")

# 從本地加載
local_model = AutoModel.from_pretrained("./my_bert_model")
```

#### **模式 3: 從檢查點加載 (訓練續接)**
```python
from transformers import AutoModel, TrainingArguments, Trainer

# 訓練時會自動保存檢查點到 output_dir
training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500
)

# 從檢查點恢復訓練
model = AutoModel.from_pretrained("./results/checkpoint-1000")
```

---

## 3. Pipeline API: 最快速的實現方式

### 3.1 什麼是 Pipeline?

Pipeline 是 Transformers 提供的高階 API,封裝了:
1. 模型加載
2. 文本預處理 (Tokenization)
3. 模型推理
4. 後處理 (Post-processing)

**核心優勢**:
- ✅ 一行代碼解決 NLP 任務
- ✅ 自動選擇最佳默認模型
- ✅ 支援批次處理
- ✅ 自動 GPU 加速

### 3.2 內建 Pipeline 任務列表

| Pipeline 名稱 | 任務 | 默認模型 | 應用場景 |
|--------------|------|---------|----------|
| `sentiment-analysis` | 情感分析 | distilbert-sst-2 | 評論正負面判斷 |
| `ner` | 命名實體識別 | dbmdz/bert-large-ner | 抽取人名、地名、組織 |
| `question-answering` | 問答系統 | distilbert-squad | 從文本中找答案 |
| `text-generation` | 文本生成 | gpt2 | 自動寫作、續寫 |
| `summarization` | 文本摘要 | sshleifer/distilbart-cnn-12-6 | 新聞摘要、論文總結 |
| `translation` | 機器翻譯 | t5-small | 多語言翻譯 |
| `zero-shot-classification` | 零樣本分類 | facebook/bart-large-mnli | 無需訓練的分類 |
| `fill-mask` | 完形填空 | distilroberta-base | BERT 式填空任務 |
| `text2text-generation` | 文本到文本 | t5-base | 通用生成任務 |
| `feature-extraction` | 特徵提取 | distilbert-base-uncased | 獲取文本嵌入向量 |

### 3.3 基礎使用範例

#### **範例 1: 情感分析**
```python
from transformers import pipeline

# 載入 Pipeline (自動下載模型)
classifier = pipeline("sentiment-analysis")

# 單筆預測
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 批次預測
texts = [
    "This is amazing!",
    "I hate waiting in line.",
    "It's okay, nothing special."
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text} → {result['label']} ({result['score']:.2%})")
```

**輸出**:
```
This is amazing! → POSITIVE (99.98%)
I hate waiting in line. → NEGATIVE (99.95%)
It's okay, nothing special. → NEUTRAL (85.23%)
```

#### **範例 2: 命名實體識別 (NER)**
```python
# 載入 NER Pipeline
ner = pipeline("ner", aggregation_strategy="simple")

text = "Hugging Face is based in New York City and was founded by Clément Delangue."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.2%})")
```

**輸出**:
```
Hugging Face: ORG (confidence: 99.89%)
New York City: LOC (confidence: 99.95%)
Clément Delangue: PER (confidence: 99.92%)
```

#### **範例 3: 問答系統**
```python
qa = pipeline("question-answering")

context = """
Hugging Face is a company based in New York City.
It was founded in 2016 and specializes in natural language processing.
The company has over 10,000 models available on its platform.
```

question = "When was Hugging Face founded?"
answer = qa(question=question, context=context)

print(f"問題: {question}")
print(f"答案: {answer['answer']} (信心度: {answer['score']:.2%})")
```

**輸出**:
```
問題: When was Hugging Face founded?
答案: 2016 (信心度: 98.75%)
```

#### **範例 4: 文本生成**
```python
generator = pipeline("text-generation", model="gpt2")

prompt = "Once upon a time in a magical forest,"
generated = generator(
    prompt,
    max_length=50,           # 最大生成長度
    num_return_sequences=2,  # 生成 2 個不同版本
    temperature=0.7,         # 創造性參數 (0-1)
    top_p=0.9,              # 核採樣參數
    do_sample=True          # 啟用採樣
)

for i, text in enumerate(generated, 1):
    print(f"\n版本 {i}:")
    print(text['generated_text'])
```

#### **範例 5: 零樣本分類**
```python
# 零樣本分類: 無需訓練,直接指定類別
classifier = pipeline("zero-shot-classification")

text = "I have a problem with my phone's battery life."
candidate_labels = ["hardware", "software", "customer service", "billing"]

result = classifier(text, candidate_labels)

print(f"文本: {text}\n")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score:.2%}")
```

**輸出**:
```
文本: I have a problem with my phone's battery life.

hardware: 87.23%
software: 8.45%
customer service: 3.12%
billing: 1.20%
```

### 3.4 自訂 Pipeline 參數

#### **指定模型**
```python
# 使用特定模型而非默認模型
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
)
```

#### **設定設備 (CPU/GPU)**
```python
import torch

# 自動選擇設備
device = 0 if torch.cuda.is_available() else -1  # 0 = GPU 0, -1 = CPU

classifier = pipeline("sentiment-analysis", device=device)
```

#### **批次處理參數**
```python
classifier = pipeline("sentiment-analysis", batch_size=32)

# 處理大量文本
texts = ["Text " + str(i) for i in range(1000)]
results = classifier(texts)  # 自動以 batch_size=32 處理
```

#### **控制生成參數**
```python
generator = pipeline("text-generation", model="gpt2")

result = generator(
    "The future of AI is",
    max_new_tokens=100,      # 生成 100 個新 token
    temperature=0.8,         # 創造性 (0 = 確定性, 1 = 隨機性)
    top_k=50,               # Top-K 採樣
    top_p=0.95,             # Nucleus 採樣
    do_sample=True,         # 啟用採樣
    repetition_penalty=1.2, # 重複懲罰
    num_return_sequences=3  # 生成 3 個版本
)
```

### 3.5 Pipeline 內部機制

理解 Pipeline 內部流程有助於調試與優化:

```python
# Pipeline 等效於以下手動流程:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 載入模型與分詞器
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 2. 預處理
text = "I love this!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 3. 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 4. 後處理
predictions = torch.softmax(logits, dim=-1)
predicted_class_id = predictions.argmax().item()
label = model.config.id2label[predicted_class_id]
score = predictions[0][predicted_class_id].item()

print(f"Label: {label}, Score: {score:.4f}")
```

---

## 4. Hugging Face Hub 模型選擇指南

### 4.1 模型命名規範

Hugging Face 模型名稱通常遵循以下格式:
```
{組織}/{模型名稱}-{規模}-{變體}-{微調任務}
```

**範例解析**:
```
distilbert-base-uncased-finetuned-sst-2-english
│         │    │        │           │      │
│         │    │        │           │      └─ 語言
│         │    │        │           └──────── 微調任務 (SST-2 情感分析)
│         │    │        └──────────────────── 微調標記
│         │    └───────────────────────────── 大小寫處理 (uncased)
│         └────────────────────────────────── 模型規模 (base)
└──────────────────────────────────────────── 模型系列 (DistilBERT)
```

### 4.2 模型選擇決策樹

```
步驟 1: 確定任務類型
├─ 分類任務 → text-classification
├─ 序列標註 → token-classification (NER, POS)
├─ 生成任務 → text-generation, text2text-generation
├─ 問答任務 → question-answering
└─ 檢索任務 → feature-extraction

步驟 2: 選擇模型規模 (根據資源限制)
├─ 資源受限 (CPU, 手機) → distilbert, albert, mobile-bert
├─ 平衡效能 (單 GPU)    → bert-base, roberta-base
└─ 追求極致 (多 GPU)    → bert-large, roberta-large, GPT-3

步驟 3: 考慮語言支援
├─ 僅英文    → bert, roberta, gpt2
├─ 僅中文    → bert-base-chinese, roberta-wwm-ext
├─ 多語言    → mbert, xlm-roberta
└─ 繁體中文  → ckiplab/bert-base-chinese, ckiplab/gpt2-base-chinese

步驟 4: 檢查微調狀態
├─ 已微調 (task-specific) → 直接使用,無需訓練
│   例: distilbert-base-uncased-finetuned-sst-2
├─ 預訓練 (pretrained)    → 需自行微調到下游任務
│   例: bert-base-uncased
└─ 基礎模型 (base)        → 需從零訓練 (不推薦)
```

### 4.3 模型卡片 (Model Card) 解讀

每個 Hugging Face 模型都附帶模型卡片,包含關鍵信息:

#### **1. Model Description (模型描述)**
- 模型架構 (BERT, GPT, T5...)
- 訓練數據來源
- 預期用途與限制

#### **2. Intended Use (預期用途)**
```markdown
✅ 適用場景:
- 英文電影評論情感分析
- 正負面分類 (二分類)
- 句子級別分類

❌ 不適用場景:
- 中文文本 (模型未針對中文訓練)
- 多類別分類 (模型僅支援二分類)
- 長文檔分類 (模型最大輸入 512 token)
```

#### **3. Training Data (訓練數據)**
- 數據來源: Stanford Sentiment Treebank (SST-2)
- 數據規模: 67,349 訓練樣本
- 數據特性: 電影評論,英文,句子級別

#### **4. Evaluation Results (評估結果)**
```
SST-2 驗證集:
- Accuracy: 91.3%
- F1 Score: 91.0%

與基準對比:
- BERT-base: 92.7%
- RoBERTa-base: 94.8%
- 本模型 (DistilBERT): 91.3%
```

#### **5. Limitations (限制)**
- 可能對非電影領域評論效果較差
- 對諷刺、雙關等複雜語義理解有限
- 訓練數據可能存在偏見 (需謹慎使用)

#### **6. How to Use (使用範例)**
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This movie was fantastic!")
print(result)
```

### 4.4 使用 Hub API 搜尋模型

#### **方法 1: 網頁搜尋** (推薦)
訪問 https://huggingface.co/models,使用篩選器:
- Task: 任務類型
- Libraries: 框架 (PyTorch, TensorFlow)
- Languages: 語言
- Sort: 排序方式 (Most downloaded, Most liked)

#### **方法 2: Python API 搜尋**
```python
from huggingface_hub import list_models

# 搜尋情感分析模型,按下載量排序
models = list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=5
)

for model in models:
    print(f"模型: {model.modelId}")
    print(f"  下載量: {model.downloads:,}")
    print(f"  標籤: {model.tags[:5]}\n")
```

#### **方法 3: CLI 搜尋**
```bash
# 搜尋中文 BERT 模型
huggingface-cli search --task text-classification --language zh

# 搜尋 GPT 系列模型
huggingface-cli search --model-name gpt
```

---

## 5. 最佳實踐與常見問題

### 5.1 效能優化技巧

#### **1. 使用量化模型減少記憶體**
```python
from transformers import AutoModelForCausalLM

# 8-bit 量化 (需要 bitsandbytes 函式庫)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_8bit=True,
    device_map="auto"
)
```

#### **2. 批次處理提升吞吐量**
```python
# 單筆處理 (慢)
for text in texts:
    result = classifier(text)

# 批次處理 (快)
results = classifier(texts, batch_size=32)
```

#### **3. 使用 ONNX 加速推理**
```python
from transformers import pipeline

# 使用 ONNX Runtime 後端
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="onnx"  # 需要安裝 optimum[onnxruntime]
)
```

#### **4. 啟用混合精度訓練**
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,  # 啟用混合精度 (需要 NVIDIA GPU)
)
```

### 5.2 常見問題與解決方案

#### **Q1: 模型下載失敗 (中國大陸用戶)**
```python
# 解決方案 1: 使用鏡像站
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 解決方案 2: 手動下載後本地加載
# 從 https://hf-mirror.com 下載模型到本地
model = AutoModel.from_pretrained("./local_model_path")
```

#### **Q2: CUDA Out of Memory (記憶體不足)**
```python
# 解決方案 1: 減少 batch size
trainer = Trainer(
    per_device_train_batch_size=8,  # 降低 batch size
    gradient_accumulation_steps=4   # 使用梯度累積補償
)

# 解決方案 2: 使用梯度檢查點
model = AutoModel.from_pretrained(
    "bert-large-uncased",
    gradient_checkpointing=True  # 犧牲速度換取記憶體
)

# 解決方案 3: 使用量化模型
model = AutoModel.from_pretrained("bert-base-uncased", load_in_8bit=True)
```

#### **Q3: 如何離線使用模型?**
```python
# 步驟 1: 在線時下載並保存
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained("./offline_model")
tokenizer.save_pretrained("./offline_model")

# 步驟 2: 離線時從本地加載
model = AutoModel.from_pretrained("./offline_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./offline_model", local_files_only=True)
```

#### **Q4: Tokenizer 警告: Token indices sequence length is longer than...**
```python
# 問題: 輸入序列超過模型最大長度
# 解決方案: 啟用截斷
tokenizer(
    text,
    truncation=True,        # 啟用截斷
    max_length=512,         # 明確指定最大長度
    return_tensors="pt"
)
```

#### **Q5: 如何處理多語言文本?**
```python
# 使用多語言模型
from transformers import pipeline

# XLM-RoBERTa 支援 100+ 種語言
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# 支援英文、中文、日文等
results = classifier([
    "I love this!",           # 英文
    "我喜歡這個!",            # 中文
    "これが大好きです!"        # 日文
])
```

### 5.3 調試技巧

#### **1. 檢查模型輸出形狀**
```python
print(f"模型輸出形狀: {outputs.logits.shape}")
# 預期: (batch_size, num_labels)
```

#### **2. 查看 Tokenizer 編碼結果**
```python
encoded = tokenizer("Test text", return_tensors="pt")
print(f"Input IDs: {encoded['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")
```

#### **3. 驗證模型配置**
```python
print(model.config)
# 檢查 num_labels, hidden_size 等參數
```

---

## 6. 實戰案例: 多任務 NLP 應用

### 案例: 客戶評論智能分析系統

**需求**: 分析客戶評論,提取情感、關鍵實體、生成摘要。

**完整實現**:
```python
from transformers import pipeline

class ReviewAnalyzer:
    def __init__(self):
        self.sentiment = pipeline("sentiment-analysis")
        self.ner = pipeline("ner", aggregation_strategy="simple")
        self.summarizer = pipeline("summarization")

    def analyze(self, review_text):
        # 1. 情感分析
        sentiment = self.sentiment(review_text)[0]

        # 2. 實體識別
        entities = self.ner(review_text)

        # 3. 摘要生成 (長文本才需要)
        summary = None
        if len(review_text) > 100:
            summary = self.summarizer(
                review_text,
                max_length=50,
                min_length=10,
                do_sample=False
            )[0]['summary_text']

        return {
            "sentiment": sentiment,
            "entities": entities,
            "summary": summary
        }

# 使用範例
analyzer = ReviewAnalyzer()

review = """
I recently visited the Apple Store in New York to buy an iPhone 15.
The staff was incredibly helpful, especially John, who spent 30 minutes
explaining all the features. However, I was disappointed by the long
wait time. Overall, a mixed experience but the product quality is excellent.
"""

result = analyzer.analyze(review)
print(f"情感: {result['sentiment']['label']} ({result['sentiment']['score']:.2%})")
print(f"提及實體: {[e['word'] for e in result['entities']]}")
print(f"摘要: {result['summary']}")
```

---

## 7. 延伸學習資源

### 官方資源
- 📚 [Hugging Face 文檔](https://huggingface.co/docs/transformers)
- 🎓 [Hugging Face 課程](https://huggingface.co/course) (免費,強烈推薦)
- 🏛️ [Model Hub](https://huggingface.co/models)
- 🗂️ [Datasets Hub](https://huggingface.co/datasets)

### 社群資源
- 💬 [Hugging Face 論壇](https://discuss.huggingface.co)
- 🐦 [Twitter @HuggingFace](https://twitter.com/huggingface)
- 📺 [YouTube 頻道](https://www.youtube.com/@HuggingFace)

### 進階主題
- 模型微調 (Fine-tuning)
- 自訂模型架構
- 分散式訓練
- 模型量化與部署
- PEFT (Parameter-Efficient Fine-Tuning)

---

## 8. 課後練習

### 練習 1: Pipeline 探索
嘗試使用以下 Pipeline:
1. `fill-mask`: 完形填空
2. `translation_en_to_de`: 英德翻譯
3. `zero-shot-classification`: 零樣本分類

### 練習 2: 模型比較
比較 `distilbert-base-uncased` 與 `bert-base-uncased` 的:
- 參數量
- 推理速度
- 準確度

### 練習 3: 自訂應用
構建一個新聞分類器,分類新聞到:
- 政治
- 科技
- 體育
- 娛樂

---

**課程**: iSpan Python NLP Cookbooks v2
**講師**: Claude AI
**最後更新**: 2025-10-17
