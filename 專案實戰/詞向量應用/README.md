# 詞向量應用專案

**專案類型**: 詞向量 - Word2Vec 與深度學習
**難度**: ⭐⭐⭐⭐ 進階
**預計時間**: 6-8 小時 (3 notebooks)
**技術棧**: Word2Vec, Gensim, Keras, LSTM

---

## 📋 專案概述

本專案深入探索**詞向量 (Word Embeddings)** 技術,從概念到實戰應用:

1. **Word2Vec 概念介紹**: 理解詞向量原理
2. **嵌入層應用**: Keras Embedding 層實戰
3. **情緒分析**: 使用詞向量+LSTM 構建情感分類器

**核心價值**:
- 將文字轉換為數值向量
- 捕捉語義相似性
- 提升深度學習模型性能

---

## 🎯 學習目標

- ✅ 理解詞向量的核心概念
- ✅ 掌握 Word2Vec (CBOW & Skip-gram) 原理
- ✅ 使用 Gensim 訓練 Word2Vec 模型
- ✅ 應用 Keras Embedding 層
- ✅ 構建詞向量+LSTM 情感分類器
- ✅ 視覺化詞向量 (t-SNE)

---

## 📊 Notebooks 詳解

### Notebook 1: Word2Vec 概念介紹

**檔名**: `專案_詞向量_Word2Vec概念介紹.ipynb`

**核心內容**:
1. 詞向量的動機 (為什麼需要?)
2. Word2Vec 兩種架構:
   - CBOW (Continuous Bag of Words)
   - Skip-gram
3. 負採樣 (Negative Sampling)
4. 使用 Gensim 訓練 Word2Vec
5. 詞向量應用:
   - 相似詞查詢
   - 詞類比 (king - man + woman = queen)
   - 詞向量可視化

**關鍵代碼**:
```python
from gensim.models import Word2Vec

# 訓練 Word2Vec
sentences = [['I', 'love', 'NLP'], ['Word2Vec', 'is', 'amazing']]
model = Word2Vec(
    sentences,
    vector_size=100,      # 向量維度
    window=5,             # 上下文窗口
    min_count=1,          # 最小詞頻
    sg=0,                 # 0=CBOW, 1=Skip-gram
    epochs=10
)

# 相似詞查詢
similar_words = model.wv.most_similar('love', topn=5)

# 詞類比
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
```

---

### Notebook 2: 嵌入層應用

**檔名**: `專案_詞向量_嵌入層應用_Keras.ipynb`

**核心內容**:
1. Keras Embedding 層原理
2. 隨機初始化 vs 預訓練詞向量
3. 載入 GloVe/Word2Vec 預訓練向量
4. 凍結/微調 Embedding 層
5. 文本分類應用

**關鍵代碼**:
```python
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 方法 1: 隨機初始化 Embedding
model = keras.Sequential([
    Embedding(
        input_dim=vocab_size,     # 詞彙表大小
        output_dim=100,           # 詞向量維度
        input_length=max_length   # 序列長度
    ),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# 方法 2: 使用預訓練詞向量
embedding_matrix = load_glove_embeddings()  # 載入 GloVe

model = keras.Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=100,
        weights=[embedding_matrix],  # 預訓練權重
        trainable=False              # 凍結 Embedding
    ),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

**對比實驗**:
| 方法 | 訓練時間 | 準確率 | 說明 |
|------|---------|-------|------|
| 隨機初始化 | 快 | 85% | 從零學習 |
| GloVe 凍結 | 快 | 91% | 利用預訓練知識 |
| GloVe 微調 | 慢 | 93% | 最佳性能 |

---

### Notebook 3: 情緒分析 (詞向量+神經網路)

**檔名**: `專案_詞向量_情緒分析_神經網路.ipynb`

**核心內容**:
1. 數據準備 (IMDB 或 Twitter)
2. 文本預處理與 Tokenization
3. 構建詞向量+LSTM 模型
4. 訓練與評估
5. 錯誤分析
6. 模型優化技巧

**完整模型架構**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
    GlobalMaxPooling1D
)

model = Sequential([
    # Embedding 層
    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=True
    ),

    # 雙向 LSTM
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),

    # 全局池化
    GlobalMaxPooling1D(),

    # 分類層
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 訓練
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

**性能指標**:
```
訓練準確率: 95.2%
驗證準確率: 91.8%
測試準確率: 90.5%
F1-Score: 0.89
```

---

## 📁 數據說明

### Notebook 1-2 數據
- 可使用任何文本語料
- 推薦: Text8, Wikipedia dump, 或自訂語料

### Notebook 3 數據
- **IMDB 電影評論**: Keras 內建
  ```python
  from tensorflow.keras.datasets import imdb
  (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
  ```
- **Twitter 情感**: Kaggle 下載
- **中文評論**: 使用 `datasets/google_reviews/`

---

## 🔧 技術難點與解決

### 難點 1: 詞彙表外詞彙 (OOV)

```python
# 問題: 測試集有訓練時未見過的詞

# 解決方案 1: 使用 <UNK> token
word_index = {'<PAD>': 0, '<UNK>': 1}
for word in vocab:
    word_index[word] = len(word_index)

# 解決方案 2: 使用字符級模型
# 解決方案 3: 使用 subword tokenization (BPE)
```

### 難點 2: 序列長度不一

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 填充/截斷到統一長度
padded_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding='post',      # 後填充
    truncating='post'    # 後截斷
)
```

### 難點 3: 類別不平衡

```python
# 解決方案 1: 類別權重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

model.fit(
    X_train, y_train,
    class_weight=dict(enumerate(class_weights))
)

# 解決方案 2: 數據重採樣
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
```

---

## 📈 進階優化

### 使用預訓練詞向量

```python
# GloVe 詞向量下載
# wget http://nlp.stanford.edu/data/glove.6B.zip

def load_glove_embeddings(glove_file, word_index, embedding_dim=100):
    """載入 GloVe 詞向量"""
    embeddings_index = {}

    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"✅ 載入 {len(embeddings_index)} 個詞向量")

    # 創建 Embedding Matrix
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# 使用
glove_matrix = load_glove_embeddings(
    'glove.6B.100d.txt',
    word_index,
    embedding_dim=100
)
```

### 詞向量可視化 (t-SNE)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 提取詞向量
words_to_plot = ['love', 'hate', 'good', 'bad', 'happy', 'sad']
vectors = [model.wv[word] for word in words_to_plot]

# t-SNE 降維到 2D
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# 繪圖
plt.figure(figsize=(10, 8))
for i, word in enumerate(words_to_plot):
    x, y = vectors_2d[i]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), fontsize=12)

plt.title('Word Embeddings Visualization (t-SNE)')
plt.show()
```

---

## ✅ 檢查清單

- [ ] 理解 Word2Vec CBOW 與 Skip-gram 差異
- [ ] 成功訓練 Word2Vec 模型
- [ ] 完成相似詞查詢與詞類比
- [ ] 掌握 Keras Embedding 層使用
- [ ] 對比隨機初始化與預訓練詞向量
- [ ] 構建完整的情感分類模型
- [ ] 達到 90%+ 準確率
- [ ] 完成詞向量 t-SNE 視覺化

---

**專案版本**: v1.0
**Notebooks 數量**: 3
**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team
