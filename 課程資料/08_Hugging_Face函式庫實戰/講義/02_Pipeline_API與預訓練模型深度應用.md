# CH08 講義 02: Pipeline API 與預訓練模型深度應用

---

**課程**: iSpan Python NLP 速成教案
**章節**: CH08 Hugging Face 函式庫實戰
**講義編號**: 02/04
**預計時間**: 120 分鐘
**對應 Notebooks**: `03-07` (情感分析、NER、零樣本、摘要、生成)

---

## 📚 學習目標

完成本講義後,你將能夠:

1. ✅ 深度掌握 Pipeline API 的 5 大核心任務
2. ✅ 理解不同預訓練模型的適用場景
3. ✅ 實作情感分析、NER、零樣本分類等實際應用
4. ✅ 掌握文本摘要與生成技術
5. ✅ 解決實際業務問題

---

## 1. 情感分析 (Sentiment Analysis)

### 1.1 什麼是情感分析?

**情感分析**是判斷文本表達的情緒或態度 (正面、負面、中性)。

#### 應用場景

| 領域 | 應用 | 價值 |
|------|------|------|
| **電商** | 產品評論分析 | 快速發現問題產品 |
| **社交媒體** | 品牌聲譽監控 | 即時危機預警 |
| **客服** | 客戶滿意度追蹤 | 改善服務品質 |
| **金融** | 市場情緒分析 | 輔助投資決策 |

---

### 1.2 使用 Pipeline 實作

#### 基礎使用

```python
from transformers import pipeline

# 創建情感分析器 (使用預設模型)
classifier = pipeline("sentiment-analysis")

# 單個文本
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 批量處理
texts = [
    "This is amazing!",
    "Terrible experience.",
    "It's okay."
]
results = classifier(texts)

for text, result in zip(texts, results):
    sentiment = result['label']
    confidence = result['score']
    print(f"{text:30} → {sentiment:8} ({confidence:.2%})")
```

**輸出**:
```
This is amazing!               → POSITIVE (99.98%)
Terrible experience.           → NEGATIVE (99.95%)
It's okay.                     → POSITIVE (52.34%)
```

---

### 1.3 使用不同的模型

#### 模型選擇

```python
# 英文情感分析 (預設)
classifier_en = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 中文情感分析
classifier_zh = pipeline(
    "sentiment-analysis",
    model="uer/roberta-base-finetuned-chinanews-chinese"
)

# 多語言情感分析
classifier_multi = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
```

#### 細粒度情感分析 (1-5 星)

```python
# 5 分類情感分析
sentiment_5class = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

text = "The product is pretty good but delivery was slow."
result = sentiment_5class(text)

print(f"Rating: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.2%}")

# Rating: 3 stars
# Confidence: 65.43%
```

---

### 1.4 實戰案例: 產品評論分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 模擬產品評論
reviews = [
    "Best purchase ever! Highly recommend.",
    "Product broke after 2 days. Very disappointed.",
    "Good quality for the price.",
    "Shipping was terrible, product is okay.",
    "Amazing! Exceeded my expectations.",
    "Do not buy. Complete waste of money.",
    "It's fine, nothing special.",
    "Love it! Will buy again."
]

# 分析情感
classifier = pipeline("sentiment-analysis")
results = classifier(reviews)

# 統計
positive = sum(1 for r in results if r['label'] == 'POSITIVE')
negative = len(results) - positive

print(f"正面評論: {positive} ({positive/len(reviews)*100:.1f}%)")
print(f"負面評論: {negative} ({negative/len(reviews)*100:.1f}%)")

# 可視化
labels = ['Positive', 'Negative']
sizes = [positive, negative]
colors = ['#2ecc71', '#e74c3c']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Product Review Sentiment Distribution')
plt.show()
```

---

## 2. 命名實體識別 (Named Entity Recognition, NER)

### 2.1 什麼是 NER?

**NER** 是從文本中提取並分類實體 (人名、地名、組織、日期等)。

#### 常見實體類型

| 類型 | 英文 | 範例 |
|------|------|------|
| 人名 | PER (Person) | Steve Jobs, 馬雲 |
| 地名 | LOC (Location) | California, 台北 |
| 組織 | ORG (Organization) | Apple, 阿里巴巴 |
| 日期 | DATE | 2024-01-01 |
| 金額 | MONEY | $1,000 |

---

### 2.2 使用 Pipeline 實作

#### 基礎 NER

```python
# 創建 NER pipeline
ner = pipeline("ner", grouped_entities=True)

text = """
Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
in April 1976 in Cupertino, California.
"""

entities = ner(text)

# 輸出結果
for entity in entities:
    print(f"{entity['word']:20} → {entity['entity_group']:5} ({entity['score']:.2%})")
```

**輸出**:
```
Apple                → ORG   (99.87%)
Steve Jobs           → PER   (99.92%)
Steve Wozniak        → PER   (99.88%)
Ronald Wayne         → PER   (99.85%)
April 1976           → DATE  (98.76%)
Cupertino            → LOC   (99.65%)
California           → LOC   (99.78%)
```

---

### 2.3 進階配置

#### 不合併實體

```python
# 不合併 entities (token-level)
ner_token = pipeline("ner")

text = "Steve Jobs founded Apple in California."
entities = ner_token(text)

for entity in entities:
    print(f"Token: {entity['word']:15} | Entity: {entity['entity']:10} | Score: {entity['score']:.2%}")
```

**輸出**:
```
Token: Steve           | Entity: B-PER      | Score: 99.92%
Token: Jobs            | Entity: I-PER      | Score: 99.89%
Token: Apple           | Entity: B-ORG      | Score: 99.87%
Token: California      | Entity: B-LOC      | Score: 99.78%
```

**標籤說明**:
- `B-PER`: Beginning of Person
- `I-PER`: Inside Person
- `B-ORG`: Beginning of Organization

---

### 2.4 實戰案例: 新聞文章實體提取

```python
# 新聞文章範例
article = """
OpenAI CEO Sam Altman announced the release of GPT-4 in San Francisco
on March 14, 2023. The new model, developed in collaboration with
Microsoft, represents a major breakthrough in artificial intelligence.
The company plans to invest $10 billion in AI research over the next
five years.
"""

# NER 分析
ner = pipeline("ner", grouped_entities=True)
entities = ner(article)

# 按類型分組
from collections import defaultdict
entities_by_type = defaultdict(list)

for entity in entities:
    entities_by_type[entity['entity_group']].append(entity['word'])

# 輸出
print("📰 新聞文章實體提取結果\n")
for entity_type, words in entities_by_type.items():
    print(f"{entity_type}:")
    for word in set(words):  # 去重
        print(f"  - {word}")
    print()
```

**輸出**:
```
📰 新聞文章實體提取結果

PER:
  - Sam Altman

ORG:
  - OpenAI
  - Microsoft

LOC:
  - San Francisco

DATE:
  - March 14, 2023
```

---

## 3. 零樣本分類 (Zero-Shot Classification)

### 3.1 什麼是零樣本分類?

**零樣本分類**允許你**不需要訓練數據**就能對文本進行分類。

#### 傳統分類 vs 零樣本分類

| 方法 | 需要訓練數據? | 靈活性 | 適用場景 |
|------|-------------|--------|----------|
| **傳統分類** | ✅ 需要 (1000+ 樣本) | 低 (固定類別) | 長期穩定任務 |
| **零樣本分類** | ❌ 不需要 | 高 (動態類別) | 快速原型、靈活任務 |

---

### 3.2 使用 Pipeline 實作

#### 基礎使用

```python
# 創建零樣本分類器
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# 文本
text = "I'm having trouble logging into my account."

# 候選標籤 (無需訓練!)
candidate_labels = [
    "Technical Support",
    "Billing Question",
    "Product Inquiry",
    "Account Issue"
]

# 分類
result = classifier(text, candidate_labels)

# 輸出
print(f"Text: {text}\n")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label:20} → {score:.2%}")
```

**輸出**:
```
Text: I'm having trouble logging into my account.

Account Issue        → 89.76%
Technical Support    → 6.45%
Billing Question     → 2.34%
Product Inquiry      → 1.45%
```

---

### 3.3 多標籤分類

```python
# 多標籤分類 (一個文本可有多個標籤)
text = "This smartphone has excellent camera quality but poor battery life."

labels = ["Camera Quality", "Battery Life", "Screen Quality", "Price"]

result = classifier(
    text,
    labels,
    multi_label=True  # 啟用多標籤
)

print("Multi-Label Results:\n")
for label, score in zip(result['labels'], result['scores']):
    if score > 0.5:  # 只顯示高信心標籤
        print(f"✓ {label:20} ({score:.2%})")
```

**輸出**:
```
Multi-Label Results:

✓ Camera Quality      (95.34%)
✓ Battery Life        (92.18%)
```

---

### 3.4 實戰案例: 客服郵件自動路由

```python
# 客服郵件分類系統
classifier = pipeline("zero-shot-classification")

# 部門標籤
departments = [
    "Technical Support",
    "Billing & Payments",
    "Shipping & Delivery",
    "Product Questions",
    "Returns & Refunds"
]

# 郵件範例
emails = [
    "My order hasn't arrived yet. Where is it?",
    "I was charged twice for the same order.",
    "How do I reset my password?",
    "Can I return this product if I don't like it?",
    "What are the specifications of this laptop?"
]

# 批量分類
print("📧 客服郵件自動路由\n")
for i, email in enumerate(emails, 1):
    result = classifier(email, departments)
    top_label = result['labels'][0]
    confidence = result['scores'][0]

    print(f"Email {i}: {email[:50]}...")
    print(f"  → Route to: {top_label} ({confidence:.1%})\n")
```

---

## 4. 文本摘要 (Summarization)

### 4.1 什麼是文本摘要?

**文本摘要**是將長文本濃縮成簡短摘要,保留核心資訊。

#### 摘要類型

| 類型 | 說明 | 範例 |
|------|------|------|
| **抽取式** | 直接選取原文句子 | 傳統方法 (TextRank) |
| **生成式** | 生成新的摘要文本 | Transformer 模型 (T5, BART) |

---

### 4.2 使用 Pipeline 實作

#### 基礎摘要

```python
# 創建摘要器
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 長文本
article = """
Artificial intelligence (AI) is intelligence demonstrated by machines,
in contrast to the natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of "intelligent agents":
any device that perceives its environment and takes actions that maximize
its chance of successfully achieving its goals. Colloquially, the term
"artificial intelligence" is often used to describe machines (or computers)
that mimic "cognitive" functions that humans associate with the human mind,
such as "learning" and "problem solving".

As machines become increasingly capable, tasks considered to require
"intelligence" are often removed from the definition of AI, a phenomenon
known as the AI effect. A quip in Tesler's Theorem says "AI is whatever
hasn't been done yet." For instance, optical character recognition is
frequently excluded from things considered to be AI, having become a
routine technology. Modern machine capabilities generally classified as
AI include successfully understanding human speech, competing at the
highest level in strategic game systems, autonomously operating cars,
intelligent routing in content delivery networks, and military simulations.
"""

# 生成摘要
summary = summarizer(
    article,
    max_length=100,
    min_length=30,
    do_sample=False
)

print("📄 原文長度:", len(article.split()))
print("\n摘要:")
print(summary[0]['summary_text'])
```

---

### 4.3 進階參數調整

```python
# 不同長度的摘要
summary_short = summarizer(
    article,
    max_length=50,
    min_length=20,
    do_sample=False
)

summary_medium = summarizer(
    article,
    max_length=100,
    min_length=50,
    do_sample=False
)

summary_long = summarizer(
    article,
    max_length=150,
    min_length=80,
    do_sample=False
)

print("簡短摘要 (50 words):")
print(summary_short[0]['summary_text'])

print("\n中等摘要 (100 words):")
print(summary_medium[0]['summary_text'])

print("\n詳細摘要 (150 words):")
print(summary_long[0]['summary_text'])
```

---

### 4.4 實戰案例: 新聞摘要系統

```python
# 新聞文章範例
news_article = """
(長篇新聞內容...)
"""

# 摘要器
summarizer = pipeline("summarization")

# 生成摘要
summary = summarizer(
    news_article,
    max_length=130,
    min_length=30,
    do_sample=False
)[0]['summary_text']

# 輸出格式化
print("=" * 70)
print("📰 新聞快訊")
print("=" * 70)
print(f"\n{summary}\n")
print("=" * 70)
```

---

## 5. 文本生成 (Text Generation)

### 5.1 什麼是文本生成?

**文本生成**是給定開頭 (prompt),模型自動續寫後續文本。

#### 應用場景

- ✍️ AI 寫作助手
- 💬 聊天機器人
- 📧 郵件自動回覆
- 📝 內容創作輔助

---

### 5.2 使用 Pipeline 實作

#### 基礎生成

```python
# 創建文本生成器
generator = pipeline("text-generation", model="gpt2")

# Prompt
prompt = "Artificial intelligence will"

# 生成文本
output = generator(
    prompt,
    max_length=50,
    num_return_sequences=3,
    temperature=0.7
)

print(f"Prompt: {prompt}\n")
for i, result in enumerate(output, 1):
    print(f"Version {i}:")
    print(result['generated_text'])
    print()
```

---

### 5.3 參數調整

#### 溫度 (Temperature)

```python
# 低溫度 (0.3) - 保守、確定性高
output_low_temp = generator(
    "Once upon a time",
    max_length=50,
    temperature=0.3,
    num_return_sequences=1
)

# 高溫度 (1.5) - 創意、多樣性高
output_high_temp = generator(
    "Once upon a time",
    max_length=50,
    temperature=1.5,
    num_return_sequences=1
)

print("低溫度 (0.3) - 更保守:")
print(output_low_temp[0]['generated_text'])

print("\n高溫度 (1.5) - 更創意:")
print(output_high_temp[0]['generated_text'])
```

#### Top-k 和 Top-p 採樣

```python
# Top-k sampling (只從最可能的 k 個詞中選)
output_topk = generator(
    "The future of technology is",
    max_length=50,
    top_k=50,
    num_return_sequences=1
)

# Top-p (nucleus) sampling
output_topp = generator(
    "The future of technology is",
    max_length=50,
    top_p=0.9,
    num_return_sequences=1
)
```

---

### 5.4 實戰案例: AI 寫作助手

```python
# 不同風格的文本生成
prompts = {
    "新聞報導": "Breaking news: Scientists have discovered",
    "故事創作": "In a distant galaxy, a young hero",
    "技術文章": "Machine learning is a subset of AI that",
    "產品描述": "Introducing our latest smartphone with"
}

generator = pipeline("text-generation", model="gpt2")

for style, prompt in prompts.items():
    print(f"\n📝 {style}:")
    print("-" * 60)

    output = generator(
        prompt,
        max_length=80,
        num_return_sequences=1,
        temperature=0.8
    )

    print(output[0]['generated_text'])
```

---

## 6. 模型選擇指南

### 6.1 任務 vs 模型對照表

| 任務 | 推薦模型 | 大小 | 速度 | 精度 |
|------|---------|------|------|------|
| **情感分析** | `distilbert-base-uncased-finetuned-sst-2` | 小 | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **NER** | `dslim/bert-base-NER` | 中 | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **零樣本** | `facebook/bart-large-mnli` | 大 | ⚡ | ⭐⭐⭐⭐⭐ |
| **摘要** | `facebook/bart-large-cnn` | 大 | ⚡ | ⭐⭐⭐⭐⭐ |
| **生成** | `gpt2` / `gpt2-medium` | 中-大 | ⚡⚡ | ⭐⭐⭐⭐ |

---

### 6.2 中文模型推薦

```python
# 中文情感分析
sentiment_zh = pipeline(
    "sentiment-analysis",
    model="uer/roberta-base-finetuned-chinanews-chinese"
)

# 中文 NER
ner_zh = pipeline(
    "ner",
    model="ckiplab/bert-base-chinese-ner",
    grouped_entities=True
)

# 中文文本生成
generator_zh = pipeline(
    "text-generation",
    model="uer/gpt2-chinese-cluecorpussmall"
)
```

---

## 7. 性能優化技巧

### 7.1 批量處理

```python
# ❌ 慢: 逐個處理
texts = ["text1", "text2", ..., "text1000"]

results = []
for text in texts:
    result = classifier(text)  # 1000 次調用
    results.append(result)

# ✅ 快: 批量處理
results = classifier(texts, batch_size=32)  # ~30 次調用
```

**加速比**: ~30x

---

### 7.2 使用 GPU

```python
import torch

# 檢查 GPU
device = 0 if torch.cuda.is_available() else -1

# 在 GPU 上運行
classifier = pipeline(
    "sentiment-analysis",
    device=device  # 0 = GPU, -1 = CPU
)
```

---

### 7.3 模型量化

```python
# 使用量化模型 (更小更快)
classifier_quantized = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # CPU
)

# 動態量化 (進階)
import torch
model = classifier_quantized.model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

---

## 8. 課後練習

### 練習 1: 情感分析儀表板

建立一個產品評論分析系統:
1. 讀取 100+ 條評論
2. 分析情感
3. 生成統計報表
4. 可視化結果

### 練習 2: 智能客服路由

實作客服郵件自動分類:
1. 定義 5 個部門類別
2. 使用零樣本分類
3. 計算準確率

### 練習 3: 新聞摘要器

建立新聞摘要工具:
1. 爬取 5 篇新聞
2. 生成摘要
3. 比較不同模型效果

---

## 9. 總結

### ✅ 核心要點

1. **5 大核心任務**
   - 情感分析: 判斷情緒
   - NER: 提取實體
   - 零樣本分類: 無需訓練數據
   - 摘要: 濃縮長文本
   - 生成: 續寫文本

2. **Pipeline 優勢**
   - 簡單易用 (3 行代碼)
   - 自動處理預處理
   - 支援批量操作

3. **模型選擇**
   - 根據任務選模型
   - 平衡速度與精度
   - 考慮語言需求

### 🎯 下一步

講義 03: 模型微調與部署實戰

---

**講義版本**: v1.0
**最後更新**: 2025-10-17
**作者**: iSpan NLP Team / Claude AI
