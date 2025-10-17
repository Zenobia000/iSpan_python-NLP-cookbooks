# iSpan Python NLP Cookbooks - 最佳實踐指南

**版本**: v1.0
**最後更新**: 2025-10-17
**目標讀者**: 學生、開發者、講師

---

## 📋 目錄

1. [代碼品質最佳實踐](#1-代碼品質最佳實踐)
2. [Notebook 撰寫規範](#2-notebook-撰寫規範)
3. [NLP 專案開發流程](#3-nlp-專案開發流程)
4. [性能優化技巧](#4-性能優化技巧)
5. [模型訓練最佳實踐](#5-模型訓練最佳實踐)
6. [數據處理規範](#6-數據處理規範)
7. [部署與維護](#7-部署與維護)
8. [安全性考量](#8-安全性考量)

---

## 1. 代碼品質最佳實踐

### 1.1 Python 編碼規範

#### **PEP 8 核心原則**

```python
# ✅ 好的代碼: 清晰、可讀、符合規範
def calculate_tf_idf(documents, vocabulary):
    """
    Calculate TF-IDF scores for documents.

    Args:
        documents (list): List of tokenized documents
        vocabulary (set): Vocabulary set

    Returns:
        np.ndarray: TF-IDF matrix
    """
    n_docs = len(documents)
    n_vocab = len(vocabulary)

    # Initialize TF-IDF matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab))

    # Calculate TF-IDF
    for doc_idx, doc in enumerate(documents):
        for word_idx, word in enumerate(vocabulary):
            tf = doc.count(word) / len(doc)
            idf = np.log(n_docs / (1 + sum(word in d for d in documents)))
            tfidf_matrix[doc_idx, word_idx] = tf * idf

    return tfidf_matrix


# ❌ 壞的代碼: 不清晰、沒註解、命名差
def calc(d,v):
    n=len(d)
    m=len(v)
    r=np.zeros((n,m))
    for i,x in enumerate(d):
        for j,y in enumerate(v):
            r[i,j]=(x.count(y)/len(x))*np.log(n/(1+sum(y in z for z in d)))
    return r
```

#### **命名規範**

| 類型 | 規範 | 範例 |
|------|------|------|
| 變數 | `snake_case` | `word_count`, `document_list` |
| 函數 | `snake_case` | `tokenize_text()`, `train_model()` |
| 類別 | `PascalCase` | `TextPreprocessor`, `SentimentAnalyzer` |
| 常數 | `UPPER_SNAKE_CASE` | `MAX_LENGTH`, `DEFAULT_BATCH_SIZE` |
| 私有方法 | `_snake_case` | `_validate_input()` |

#### **Type Hints (類型提示)**

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

# ✅ 使用 Type Hints
def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True
) -> str:
    """
    Preprocess input text.

    Args:
        text: Input text string
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation

    Returns:
        Preprocessed text
    """
    if lowercase:
        text = text.lower()

    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)

    return text


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32
) -> Tuple[object, Dict[str, float]]:
    """Train classification model."""
    # Training code...
    return model, metrics
```

### 1.2 錯誤處理

#### **防禦性編程**

```python
# ✅ 完善的錯誤處理
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset with comprehensive error handling.
    """
    # 檢查檔案存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # 檢查檔案格式
    if not file_path.endswith('.csv'):
        raise ValueError(f"Expected CSV file, got: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 嘗試其他編碼
        try:
            df = pd.read_csv(file_path, encoding='big5')
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    # 驗證必要欄位
    required_columns = ['text', 'label']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ❌ 沒有錯誤處理
def load_dataset(file_path):
    return pd.read_csv(file_path)  # 可能崩潰!
```

#### **使用 try-except-finally**

```python
# 完整的資源管理
def process_large_file(file_path: str):
    file_handle = None
    try:
        file_handle = open(file_path, 'r', encoding='utf-8')

        # 處理數據
        for line in file_handle:
            process_line(line)

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        raise
    except UnicodeDecodeError:
        print(f"Error: Encoding issue in {file_path}")
        raise
    finally:
        # 確保檔案關閉
        if file_handle:
            file_handle.close()


# 更好的方式: 使用 context manager
def process_large_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                process_line(line)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        raise
```

### 1.3 日誌記錄

```python
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 在代碼中使用
def train_model(data, epochs=10):
    logger.info(f"Starting training with {len(data)} samples")

    for epoch in range(epochs):
        logger.debug(f"Epoch {epoch+1}/{epochs}")

        try:
            loss = training_step(data)
            logger.info(f"Epoch {epoch+1} - Loss: {loss:.4f}")
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch+1}: {e}")
            raise

    logger.info("Training completed successfully")
```

---

## 2. Notebook 撰寫規範

### 2.1 Notebook 結構模板

```markdown
# CH0X-0Y: [章節標題]

**課程**: iSpan Python NLP Cookbooks v2
**章節**: CH0X [章節名稱]
**版本**: v1.0
**更新日期**: YYYY-MM-DD

---

## 📚 本節學習目標

1. 學習目標 1
2. 學習目標 2
3. 學習目標 3

---

## 1. 理論基礎

### 1.1 核心概念
[理論說明]

### 1.2 數學原理
[公式推導]

---

## 2. 實作示範

### 2.1 環境準備
[導入套件、設置參數]

### 2.2 數據準備
[加載數據、EDA]

### 2.3 核心實作
[主要代碼]

---

## 3. 結果分析

### 3.1 結果展示
[輸出結果]

### 3.2 視覺化
[圖表]

---

## 4. 延伸練習

### 練習 1: [練習名稱]
[練習說明]

---

## 5. 本節總結

- ✅ 重點 1
- ✅ 重點 2
- ✅ 重點 3

---

## 6. 延伸閱讀

- [資源 1]
- [資源 2]
```

### 2.2 Cell 組織原則

#### **Markdown Cell 使用**
```markdown
# ✅ 清晰的章節標題
## 1. 數據載入與探索
### 1.1 載入數據

# ✅ 使用列表說明步驟
**接下來我們將:**
1. 載入數據集
2. 檢查數據維度
3. 查看前 5 筆資料

# ✅ 使用表格整理資訊
| 參數 | 說明 | 默認值 |
|------|------|--------|
| max_length | 最大序列長度 | 512 |
| batch_size | 批次大小 | 32 |

# ✅ 使用警告框
> ⚠️ **注意**: 此步驟需要 8GB+ RAM
> 💡 **提示**: 可以減少 batch_size 來降低記憶體需求
```

#### **Code Cell 原則**

```python
# ✅ 原則 1: 一個 Cell 做一件事
# Cell 1: 導入套件
import numpy as np
import pandas as pd
from transformers import pipeline

# Cell 2: 載入數據
df = pd.read_csv('data.csv')

# Cell 3: 數據探索
print(df.shape)
df.head()

# ❌ 避免: 一個 Cell 做太多事
import numpy as np
df = pd.read_csv('data.csv')
df.dropna(inplace=True)
model = train_model(df)
results = evaluate(model)
plot_results(results)  # 太複雜!
```

```python
# ✅ 原則 2: 重要輸出要顯示
print(f"✅ 數據載入完成: {len(df)} 筆資料")
print(f"✅ 訓練完成: Accuracy = {accuracy:.2%}")

# ✅ 原則 3: 使用進度條 (長時間操作)
from tqdm.notebook import tqdm

for epoch in tqdm(range(100), desc="Training"):
    train_epoch()

# ✅ 原則 4: 結果視覺化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses)
plt.title('Training Loss', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.3 註解與文檔字串

```python
# ✅ 完整的函數文檔
def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical).

    Formula:
        similarity = (A · B) / (||A|| × ||B||)

    Args:
        vec1 (np.ndarray): First vector (shape: [n,])
        vec2 (np.ndarray): Second vector (shape: [n,])

    Returns:
        float: Cosine similarity score in range [-1, 1]

    Raises:
        ValueError: If vectors have different dimensions

    Examples:
        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([4, 5, 6])
        >>> similarity = calculate_cosine_similarity(vec1, vec2)
        >>> print(f"Similarity: {similarity:.4f}")
        Similarity: 0.9746
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector dimensions mismatch: {vec1.shape} vs {vec2.shape}")

    # 計算點積
    dot_product = np.dot(vec1, vec2)

    # 計算範數
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 避免除以零
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
```

---

## 3. NLP 專案開發流程

### 3.1 標準開發流程

```
階段 1: 問題定義 (1-2 天)
├── 明確業務目標
├── 定義評估指標
└── 確認數據可得性

階段 2: 數據準備 (3-5 天)
├── 數據收集
├── 數據清理
├── EDA (探索性數據分析)
└── 數據標註 (如需要)

階段 3: 基準模型 (2-3 天)
├── 建立簡單基準 (Baseline)
├── 評估基準性能
└── 設定目標改進幅度

階段 4: 模型開發 (5-10 天)
├── 特徵工程
├── 模型選擇與訓練
├── 超參數調優
└── 交叉驗證

階段 5: 評估與優化 (3-5 天)
├── 錯誤分析
├── 模型改進
├── A/B 測試
└── 最終評估

階段 6: 部署與監控 (2-4 天)
├── 模型打包
├── API 開發
├── 部署到生產
└── 建立監控系統
```

### 3.2 實戰案例: 垃圾郵件分類器

#### **階段 1: 問題定義**

```markdown
## 業務目標
過濾垃圾郵件,降低用戶干擾

## 成功指標
- Precision > 95% (避免誤判正常郵件)
- Recall > 90% (抓住大部分垃圾郵件)
- 推理時間 < 100ms

## 數據需求
- 5000+ 標註郵件
- 垃圾/正常 比例 1:1
```

#### **階段 2: 數據準備**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 載入數據
df = pd.read_csv('spam_dataset.csv')

# 2. EDA
print(f"數據總數: {len(df)}")
print(f"垃圾郵件: {(df['label'] == 1).sum()}")
print(f"正常郵件: {(df['label'] == 0).sum()}")

# 3. 檢查缺失值
print(f"\n缺失值:\n{df.isnull().sum()}")

# 4. 數據清理
df = df.dropna()
df = df.drop_duplicates(subset=['text'])

# 5. 劃分數據集
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']  # 保持類別比例
)

print(f"\n訓練集: {len(X_train)} | 測試集: {len(X_test)}")
```

#### **階段 3: 建立基準**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 簡單基準: TF-IDF + Naive Bayes
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

baseline_model = MultinomialNB()
baseline_model.fit(X_train_tfidf, y_train)

# 評估
y_pred = baseline_model.predict(X_test_tfidf)
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Spam']))
```

#### **階段 4: 模型開發**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# 1. 載入預訓練模型
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 2. Tokenization
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

# 3. 訓練配置
training_args = TrainingArguments(
    output_dir='./spam_classifier',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100
)

# 4. 訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

#### **階段 5: 錯誤分析**

```python
# 找出預測錯誤的案例
y_pred = trainer.predict(eval_dataset).predictions.argmax(-1)
errors = X_test[y_pred != y_test]

print(f"錯誤預測數: {len(errors)}")
print("\n前 5 個錯誤案例:")
for i, (text, true_label, pred_label) in enumerate(
    zip(errors[:5], y_test[y_pred != y_test][:5], y_pred[y_pred != y_test][:5])
):
    print(f"\n案例 {i+1}:")
    print(f"  文本: {text[:100]}...")
    print(f"  真實標籤: {true_label}")
    print(f"  預測標籤: {pred_label}")
```

---

## 4. 性能優化技巧

### 4.1 數據處理優化

#### **使用向量化操作**

```python
# ❌ 慢: 使用循環
def count_words_slow(texts):
    word_counts = []
    for text in texts:
        count = len(text.split())
        word_counts.append(count)
    return word_counts

# ✅ 快: 使用 Pandas 向量化
def count_words_fast(texts):
    return texts.str.split().str.len()

# 速度對比
import time
texts = pd.Series(["sample text"] * 10000)

start = time.time()
result1 = count_words_slow(texts)
print(f"循環方式: {time.time() - start:.3f}s")

start = time.time()
result2 = count_words_fast(texts)
print(f"向量化: {time.time() - start:.3f}s")
# 向量化快 10-100 倍!
```

#### **批次處理大型數據**

```python
# ✅ 使用 chunk 處理大檔案
def process_large_csv(file_path, chunk_size=10000):
    """逐塊處理大型 CSV,避免記憶體溢出"""
    results = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # 處理每個 chunk
        processed_chunk = preprocess_chunk(chunk)
        results.append(processed_chunk)

    # 合併結果
    return pd.concat(results, ignore_index=True)
```

### 4.2 模型推理優化

#### **使用 @torch.no_grad() 節省記憶體**

```python
import torch

# ✅ 推理時關閉梯度計算
@torch.no_grad()
def predict(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits.argmax(-1)

# 或使用 context manager
def predict_batch(model, texts):
    with torch.no_grad():
        # 不會計算梯度,節省記憶體
        predictions = []
        for text in texts:
            pred = model(text)
            predictions.append(pred)
    return predictions
```

#### **模型量化**

```python
# 動態量化 (INT8)
import torch

model_fp32 = load_model()  # FP32 模型
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# 比較大小
print(f"FP32 size: {get_model_size(model_fp32):.2f} MB")
print(f"INT8 size: {get_model_size(model_int8):.2f} MB")
# 通常減少 75%
```

### 4.3 GPU 使用優化

```python
# 檢查並使用 GPU
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 將模型移至 GPU
model = model.to(device)

# 推理時也要移到 GPU
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs)

# 清理 GPU 記憶體
torch.cuda.empty_cache()
```

---

## 5. 模型訓練最佳實踐

### 5.1 訓練前檢查清單

- [ ] **數據驗證**
  - [ ] 無缺失值或已處理
  - [ ] 類別平衡或已加權
  - [ ] 訓練/驗證/測試集劃分正確

- [ ] **模型配置**
  - [ ] 超參數設置合理
  - [ ] 學習率適當 (2e-5 ~ 5e-5 for BERT)
  - [ ] Batch size 適配記憶體

- [ ] **評估指標**
  - [ ] 選擇合適的指標 (Accuracy, F1, AUC...)
  - [ ] 定義成功標準

- [ ] **實驗追蹤**
  - [ ] 使用 TensorBoard 或 W&B
  - [ ] 記錄超參數
  - [ ] 可重現 (設置 random seed)

### 5.2 防止過擬合

```python
from transformers import TrainingArguments, EarlyStoppingCallback

# 1. Early Stopping
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # 載入最佳模型
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 2. Dropout (模型層面)
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-uncased")
config.hidden_dropout_prob = 0.2  # 增加 dropout
config.attention_probs_dropout_prob = 0.2

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config
)

# 3. Weight Decay (L2 正則化)
training_args = TrainingArguments(
    weight_decay=0.01  # L2 正則化
)

# 4. 數據增強
def augment_text(text):
    """簡單數據增強: 同義詞替換"""
    # 使用 nlpaug 或手動實作
    return augmented_text
```

### 5.3 超參數調優

```python
# 使用 Optuna 自動搜索超參數
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
    }

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 執行搜索
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=20
)

print(f"Best hyperparameters: {best_run.hyperparameters}")
```

---

## 6. 數據處理規範

### 6.1 文本清理標準流程

```python
import re
import string

class TextCleaner:
    """標準文本清理類"""

    def __init__(self):
        self.punct = set(string.punctuation)

    def clean(self, text: str) -> str:
        """
        完整清理流程
        """
        # 1. 轉小寫
        text = text.lower()

        # 2. 移除 URL
        text = re.sub(r'http\S+|www\S+', '', text)

        # 3. 移除 email
        text = re.sub(r'\S+@\S+', '', text)

        # 4. 移除數字 (視需求)
        text = re.sub(r'\d+', '', text)

        # 5. 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()

        # 6. 移除標點 (視需求)
        text = ''.join(char for char in text if char not in self.punct)

        return text

# 使用
cleaner = TextCleaner()
clean_text = cleaner.clean("Check out http://example.com! Email: test@test.com")
print(clean_text)
# "check out email"
```

### 6.2 中文文本處理

```python
import jieba
import re
from opencc import OpenCC

class ChineseTextProcessor:
    """中文文本處理器"""

    def __init__(self, stopwords_path='shared_resources/stopwords/stopwords_zh_tw.txt'):
        # 載入停用詞
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            self.stopwords = set(f.read().splitlines())

        # 簡繁轉換器
        self.cc = OpenCC('s2tw')  # 簡體轉繁體

    def clean(self, text: str) -> str:
        """清理中文文本"""
        # 移除英文和數字
        text = re.sub(r'[a-zA-Z0-9]', '', text)

        # 移除標點符號
        text = re.sub(r'[^\w\s]', '', text)

        # 移除空白
        text = re.sub(r'\s+', '', text)

        return text

    def segment(self, text: str, remove_stopwords: bool = True) -> list:
        """中文斷詞"""
        # 斷詞
        words = jieba.lcut(text)

        # 移除停用詞
        if remove_stopwords:
            words = [w for w in words if w not in self.stopwords]

        # 移除單字
        words = [w for w in words if len(w) > 1]

        return words

    def convert_to_traditional(self, text: str) -> str:
        """簡體轉繁體"""
        return self.cc.convert(text)

# 使用範例
processor = ChineseTextProcessor()

text = "我爱自然语言处理!"
text_tw = processor.convert_to_traditional(text)  # 繁體
words = processor.segment(text_tw)                 # 斷詞

print(f"原文: {text}")
print(f"繁體: {text_tw}")
print(f"斷詞: {words}")
```

### 6.3 數據增強技術

```python
# 1. 回譯 (Back Translation)
from transformers import pipeline

translator_en2de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
translator_de2en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

def back_translate(text):
    """英文回譯增強"""
    # 英 → 德
    de_text = translator_en2de(text)[0]['translation_text']
    # 德 → 英
    augmented = translator_de2en(de_text)[0]['translation_text']
    return augmented

# 2. 同義詞替換
import nltk
from nltk.corpus import wordnet

def synonym_replacement(text, n=2):
    """隨機替換 n 個詞為同義詞"""
    words = text.split()

    for _ in range(n):
        idx = random.randint(0, len(words)-1)
        word = words[idx]

        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            words[idx] = synonym

    return ' '.join(words)

# 3. 隨機插入/刪除/交換
def random_insertion(text):
    """隨機插入詞"""
    # 實作邏輯...
    pass

def random_deletion(text, p=0.1):
    """隨機刪除詞 (機率 p)"""
    words = text.split()
    if len(words) == 1:
        return text

    new_words = [w for w in words if random.random() > p]
    return ' '.join(new_words) if new_words else random.choice(words)
```

---

## 7. 部署與維護

### 7.1 模型版本管理

```python
# 使用 MLflow 追蹤模型
import mlflow
import mlflow.pytorch

# 開始實驗
mlflow.start_run()

# 記錄參數
mlflow.log_param("learning_rate", 2e-5)
mlflow.log_param("batch_size", 16)
mlflow.log_param("epochs", 3)

# 訓練模型
trainer.train()

# 記錄指標
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_score", f1)

# 保存模型
mlflow.pytorch.log_model(model, "model")

# 結束實驗
mlflow.end_run()
```

### 7.2 A/B 測試

```python
# 線上 A/B 測試框架
class ABTestRouter:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a  # 基準模型
        self.model_b = model_b  # 新模型
        self.split_ratio = split_ratio
        self.metrics_a = []
        self.metrics_b = []

    def predict(self, text, user_id):
        """根據 user_id 路由到不同模型"""
        # 使用 user_id hash 決定使用哪個模型
        use_model_b = hash(user_id) % 100 < (self.split_ratio * 100)

        if use_model_b:
            result = self.model_b(text)
            self.metrics_b.append(result)
            return result, "model_b"
        else:
            result = self.model_a(text)
            self.metrics_a.append(result)
            return result, "model_a"

    def get_stats(self):
        """獲取 A/B 測試統計"""
        return {
            "model_a_requests": len(self.metrics_a),
            "model_b_requests": len(self.metrics_b),
            # 添加更多指標...
        }
```

---

## 8. 安全性考量

### 8.1 輸入驗證

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @validator('text')
    def validate_text(cls, v):
        # 檢查是否為空白
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")

        # 檢查惡意輸入 (簡單範例)
        if '<script>' in v.lower():
            raise ValueError("Potential XSS attack detected")

        return v

@app.post("/predict")
def predict(input_data: TextInput):
    # 輸入已經過驗證
    result = classifier(input_data.text)
    return result
```

### 8.2 API 速率限制

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# 限制: 每分鐘 60 次請求
@app.post("/predict")
@limiter.limit("60/minute")
def predict(request: Request, input_data: TextInput):
    result = classifier(input_data.text)
    return result
```

### 8.3 模型安全

```python
# 避免模型被盜用: 添加水印
def add_watermark(model, watermark_text="iSpan NLP Project"):
    """在模型中嵌入水印"""
    # 進階技巧: 在特定層添加隱藏特徵
    pass

# 避免模型投毒攻擊: 驗證訓練數據
def validate_training_data(dataset):
    """檢查訓練數據異常"""
    # 檢查標籤分布
    label_dist = dataset['label'].value_counts()
    if label_dist.min() / label_dist.max() < 0.1:
        print("⚠️ Warning: Severe class imbalance detected")

    # 檢查異常文本
    for text in dataset['text']:
        if len(text) > 10000:  # 異常長文本
            print(f"⚠️ Warning: Abnormally long text found")
```

---

## 9. 課後總結

### ✅ 核心最佳實踐

1. **代碼品質**
   - 遵循 PEP 8
   - 使用 Type Hints
   - 完整錯誤處理
   - 詳細註解

2. **Notebook 規範**
   - 清晰結構
   - Markdown 說明
   - 視覺化結果
   - 練習與總結

3. **性能優化**
   - 向量化操作
   - 批次處理
   - GPU 加速
   - 模型量化

4. **模型訓練**
   - 基準先行
   - 防止過擬合
   - 實驗追蹤
   - 錯誤分析

5. **部署維護**
   - 版本管理
   - A/B 測試
   - 監控告警
   - 安全防護

### 🎯 持續改進

- 定期 Code Review
- 關注最新技術
- 參與開源社群
- 建立個人知識庫

---

**文件版本**: v1.0
**維護者**: iSpan NLP Team
**授權**: CC BY-NC-SA 4.0
**最後更新**: 2025-10-17
