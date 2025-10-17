# 主題建模應用 - LDA 實戰專案

**專案類型**: 無監督學習 - 主題發現
**難度**: ⭐⭐⭐⭐ 進階
**預計時間**: 6-8 小時 (2 個子專案)
**技術棧**: LDA, Gensim, pyLDAvis

---

## 📋 專案概述

本專案展示**潛在狄利克雷分配 (Latent Dirichlet Allocation, LDA)** 主題建模技術在兩個不同領域的應用:

1. **唐詩主題分析**: 探索古典文學主題分布
2. **系統日誌主題分析**: IT 運維日誌模式識別

**核心價值**:
- 無需標註數據即可發現隱藏主題
- 自動分類大規模文檔
- 提供可解釋的主題詞彙

---

## 🎯 學習目標

- ✅ 理解 LDA 主題建模原理
- ✅ 掌握 Gensim LDA 訓練流程
- ✅ 使用 pyLDAvis 進行互動式視覺化
- ✅ 解釋主題建模結果
- ✅ 調優主題數量與參數

---

## 📊 子專案說明

### 專案 1: 唐詩主題分析

**目錄**: `主題建模應用/唐詩主題分析/`

**檔案**:
- `專案_主題建模_唐詩分析_LDA.ipynb` - 主要 notebook
- `唐詩三百首.json` - 唐詩數據集
- `poetry-lda-visualization.html` - 互動式視覺化結果

**專案亮點**:
- 使用經典文學語料 (唐詩三百首)
- 發現唐詩主題分布 (山水、邊塞、離別、愛情...)
- 生成互動式 HTML 視覺化

**核心流程**:
```python
# 1. 載入唐詩數據
import json
with open('唐詩三百首.json', 'r', encoding='utf-8') as f:
    poems = json.load(f)

# 2. 文本預處理
import jieba
processed_poems = [jieba.lcut(poem['content']) for poem in poems]

# 3. 建立詞典與語料庫
from gensim import corpora
dictionary = corpora.Dictionary(processed_poems)
corpus = [dictionary.doc2bow(poem) for poem in processed_poems]

# 4. 訓練 LDA 模型
from gensim.models import LdaModel
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,         # 主題數量
    random_state=42,
    passes=10,            # 訓練迭代次數
    alpha='auto',
    per_word_topics=True
)

# 5. 查看主題
for idx, topic in lda_model.print_topics(-1):
    print(f"主題 {idx}: {topic}")

# 6. 互動式視覺化
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'poetry-lda-visualization.html')
```

**預期主題發現**:
```
主題 0 (山水田園): 山、水、雲、風、花
主題 1 (邊塞征戰): 將、軍、戰、馬、關
主題 2 (離別相思): 別、思、歸、遠、憶
主題 3 (宮廷宴飲): 宮、酒、宴、歌、舞
主題 4 (愛情懷古): 情、愛、夢、淚、心
```

---

### 專案 2: 系統日誌主題分析

**目錄**: `主題建模應用/系統反應主題分析/`

**檔案**:
- `專案_主題建模_系統日誌_LDA.ipynb` - 主要 notebook
- `poetry-lda-visualization.html` - 視覺化結果

**應用場景**:
- IT 運維日誌分析
- 異常模式識別
- 故障預警
- 日誌分類

**核心流程**:
```python
# 系統日誌範例
logs = [
    "[ERROR] Database connection timeout at 10.0.0.5:3306",
    "[WARNING] High CPU usage detected: 95%",
    "[INFO] User authentication successful",
    "[ERROR] Null pointer exception in module payment.py",
    ...
]

# 預處理: 提取關鍵詞
def extract_log_keywords(log):
    """從日誌提取關鍵詞"""
    # 移除時間戳
    log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', log)

    # 移除 IP 地址
    log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', log)

    # 提取重要詞彙
    words = log.lower().split()
    return words

# LDA 訓練
processed_logs = [extract_log_keywords(log) for log in logs]
# ... (同唐詩專案流程)
```

**預期主題發現**:
```
主題 0 (資料庫問題): database, connection, timeout, mysql
主題 1 (性能問題): cpu, memory, high, usage, slow
主題 2 (認證問題): authentication, login, user, failed
主題 3 (應用錯誤): error, exception, null, pointer
主題 4 (網路問題): network, timeout, unreachable, connection
```

---

## 🔧 技術細節

### LDA 參數說明

| 參數 | 說明 | 推薦值 | 影響 |
|------|------|--------|------|
| `num_topics` | 主題數量 | 5-20 | 太少: 主題粗糙<br>太多: 主題碎片化 |
| `passes` | 訓練輪數 | 10-50 | 影響收斂性 |
| `iterations` | 每輪迭代次數 | 50-100 | 影響精度 |
| `alpha` | 文檔-主題分布 | 'auto' 或 0.1 | 控制主題稀疏性 |
| `eta` | 主題-詞彙分布 | 'auto' 或 0.01 | 控制詞彙稀疏性 |
| `random_state` | 隨機種子 | 42 | 確保可重現 |

### 主題數量選擇

```python
from gensim.models import CoherenceModel

# 方法 1: Coherence Score (推薦)
coherence_scores = []
topic_range = range(2, 15)

for num_topics in topic_range:
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    coherence_model = CoherenceModel(
        model=lda,
        texts=processed_texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_scores.append(coherence_model.get_coherence())

# 找出最佳主題數
best_num_topics = topic_range[np.argmax(coherence_scores)]
print(f"最佳主題數: {best_num_topics}")

# 方法 2: Perplexity (困惑度)
perplexity = lda_model.log_perplexity(corpus)
print(f"Perplexity: {perplexity}")  # 越低越好
```

---

## 🎨 視覺化技巧

### pyLDAvis 互動式視覺化

```python
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# 生成視覺化
vis = gensimvis.prepare(
    lda_model,
    corpus,
    dictionary,
    sort_topics=False
)

# 在 Notebook 中顯示
pyLDAvis.display(vis)

# 保存為 HTML (可分享)
pyLDAvis.save_html(vis, 'lda_visualization.html')
```

**視覺化功能**:
- 左側: 主題分布圖 (Inter-Topic Distance Map)
- 右側: 主題詞彙列表
- 互動: 點擊主題查看詞彙,調整 λ 參數

### 主題演化分析

```python
import matplotlib.pyplot as plt

# 追蹤主題隨時間變化
def plot_topic_evolution(documents_by_time, topic_id):
    """繪製主題隨時間的演化"""
    topic_proportions = []

    for time_period, docs in documents_by_time.items():
        corpus_period = [dictionary.doc2bow(doc) for doc in docs]

        # 計算該時間段的主題比例
        topic_dist = [lda_model.get_document_topics(doc) for doc in corpus_period]
        topic_prop = np.mean([
            dict(dist).get(topic_id, 0) for dist in topic_dist
        ])
        topic_proportions.append(topic_prop)

    plt.plot(documents_by_time.keys(), topic_proportions)
    plt.title(f'Topic {topic_id} Evolution Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('Topic Proportion')
    plt.show()
```

---

## 📈 擴展建議

### 初級擴展
- [ ] 嘗試不同的主題數量 (5, 10, 15)
- [ ] 調整預處理流程 (停用詞、min_df)
- [ ] 分析不同作者的主題偏好

### 中級擴展
- [ ] 使用 NMF (非負矩陣分解) 對比
- [ ] 整合情感分析
- [ ] 建立主題搜尋引擎

### 進階擴展
- [ ] 使用 BERTopic (基於 BERT 的主題建模)
- [ ] 動態主題追蹤
- [ ] 跨語言主題對齊
- [ ] 階層式主題建模 (hLDA)

---

## 🏆 作品集展示建議

### 展示要點

1. **唐詩專案**:
   - "使用 LDA 分析唐詩三百首,發現 5 大主題類型"
   - "互動式視覺化展示主題詞彙分布"
   - "中文 NLP 與文學分析結合"

2. **日誌專案**:
   - "自動分析 10,000+ 條系統日誌"
   - "識別 5 類常見故障模式"
   - "實際應用於 IT 運維場景"

### GitHub README 建議結構

```markdown
# 唐詩主題建模專案

## 專案背景
探索唐詩三百首的主題分布...

## 技術架構
- LDA 主題建模
- Gensim 訓練
- pyLDAvis 視覺化

## 主要發現
1. 發現 5 大主題...
2. 山水詩占比 35%...

## 視覺化結果
![LDA Visualization](screenshots/lda_vis.png)

## 如何運行
...
```

---

## 📚 延伸閱讀

- [LDA 原論文](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [Gensim LDA 教學](https://radimrehurek.com/gensim/models/ldamodel.html)
- [pyLDAvis 文檔](https://github.com/bmabey/pyLDAvis)
- [BERTopic: 現代主題建模](https://github.com/MaartenGr/BERTopic)

---

**專案版本**: v1.0
**最後更新**: 2025-10-17
**包含子專案**: 2 個
**總 Notebooks**: 2
**視覺化檔案**: 2 個 HTML
