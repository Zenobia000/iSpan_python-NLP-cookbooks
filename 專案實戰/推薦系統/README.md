# 推薦系統 - 內容過濾實戰

**專案類型**: 推薦演算法 - 基於內容的過濾
**難度**: ⭐⭐⭐ 中級
**預計時間**: 2-3 小時
**技術棧**: TF-IDF, Cosine Similarity, Scikit-learn

---

## 📋 專案概述

本專案實作**基於內容的推薦系統** (Content-Based Filtering),使用 TF-IDF 向量化技術和餘弦相似度計算,實現文章/商品的智能推薦。

**核心技術**:
- TF-IDF 文本向量化
- 餘弦相似度計算
- 相似項目推薦
- 推薦結果排序與過濾

---

## 🎯 學習目標

- ✅ 理解基於內容推薦的核心原理
- ✅ 掌握 TF-IDF 向量化技術
- ✅ 實作餘弦相似度計算
- ✅ 構建完整的推薦系統
- ✅ 評估推薦系統性能

---

## 📊 Notebook 內容

**檔名**: `專案_推薦系統_內容過濾_TFIDF.ipynb`

### 核心流程

```
Step 1: 數據準備
    ├── 載入文章/商品數據
    ├── 文本預處理
    └── 特徵提取

Step 2: TF-IDF 向量化
    ├── 建立詞彙表
    ├── 計算 TF-IDF 權重
    └── 生成文檔向量矩陣

Step 3: 相似度計算
    ├── 計算餘弦相似度
    ├── 建立相似度矩陣
    └── 找出最相似項目

Step 4: 推薦生成
    ├── 根據用戶歷史推薦
    ├── 排序與過濾
    └── Top-N 推薦結果

Step 5: 評估與優化
    ├── 推薦多樣性
    ├── 覆蓋率分析
    └── 準確率評估
```

---

## 🧮 核心算法

### TF-IDF 公式

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

其中:
TF(t, d) = 詞 t 在文檔 d 中的頻率
IDF(t) = log(總文檔數 / 包含詞 t 的文檔數)
```

### 餘弦相似度公式

```
similarity(A, B) = (A · B) / (||A|| × ||B||)

範圍: [-1, 1]
- 1: 完全相同
- 0: 無關聯
- -1: 完全相反
```

### 實作範例

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. TF-IDF 向量化
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(documents)

# 2. 計算相似度矩陣
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. 推薦函數
def get_recommendations(item_id, top_n=5):
    """
    根據項目 ID 推薦最相似的 N 個項目
    """
    # 獲取相似度分數
    sim_scores = list(enumerate(similarity_matrix[item_id]))

    # 排序 (降序)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 排除自己,取前 N 個
    sim_scores = sim_scores[1:top_n+1]

    # 提取項目索引
    item_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    return item_indices, scores

# 使用
recommendations, scores = get_recommendations(item_id=0, top_n=5)
print(f"推薦項目: {recommendations}")
print(f"相似度: {scores}")
```

---

## 📁 數據說明

### 數據來源
- **位置**: `datasets/lyrics/情歌歌詞/` 或自訂數據
- **格式**: 純文本 .txt 檔案
- **數量**: 290+ 首情歌
- **語言**: 繁體中文

### 數據格式範例
```
每個 .txt 檔案包含一首歌的完整歌詞:

檔名: 100_原來這才是真的你.txt
內容:
---
原來這才是真的你
原來愛會讓人失去自己
原來陪伴才是最深情的告白
...
---
```

### 載入數據
```python
from pathlib import Path

lyrics_dir = Path("../../datasets/lyrics/情歌歌詞")
lyrics_files = sorted(lyrics_dir.glob("*.txt"))

lyrics_data = []
for file in lyrics_files:
    with open(file, 'r', encoding='utf-8') as f:
        lyrics_data.append({
            'title': file.stem,  # 檔名作為歌名
            'lyrics': f.read()
        })

print(f"✅ 載入 {len(lyrics_data)} 首歌詞")
```

---

## 🎨 預期結果

### 推薦結果範例

```
輸入歌曲: "愛久見人心"

推薦歌曲:
1. 心心相印 (相似度: 0.87)
2. 愛的天靈靈 (相似度: 0.82)
3. 單戀 (相似度: 0.79)
4. 我喜歡 (相似度: 0.76)
5. 最想環遊的世界 (相似度: 0.74)
```

### 視覺化

1. **詞頻長條圖**
   - 展示 TOP 20 高頻詞
   - 橫軸: 詞彙
   - 縱軸: 出現次數

2. **文字雲**
   - 愛情相關詞彙: "愛"、"心"、"你"、"我"
   - 字體大小對應頻率

3. **相似度熱圖**
   - 展示前 20 首歌的相似度矩陣
   - 顏色深淺代表相似程度

---

## 🔧 技術細節

### TF-IDF 參數調優

```python
# 基礎配置
vectorizer = TfidfVectorizer()

# 進階配置
vectorizer = TfidfVectorizer(
    max_features=1000,        # 最多保留 1000 個詞
    min_df=2,                 # 至少出現在 2 個文檔
    max_df=0.8,               # 最多出現在 80% 文檔
    ngram_range=(1, 2),       # 1-gram 和 2-gram
    sublinear_tf=True,        # 使用 log TF
    use_idf=True,             # 使用 IDF
    smooth_idf=True           # 平滑 IDF
)
```

### 提升推薦質量

```python
# 1. 多樣性過濾 (避免推薦太相似的項目)
def diverse_recommendations(item_id, top_n=10, diversity_threshold=0.95):
    """推薦多樣化的項目"""
    candidates, scores = get_recommendations(item_id, top_n=50)

    selected = []
    for candidate, score in zip(candidates, scores):
        # 檢查與已選項目的相似度
        is_diverse = True
        for selected_item in selected:
            if similarity_matrix[candidate][selected_item] > diversity_threshold:
                is_diverse = False
                break

        if is_diverse:
            selected.append(candidate)

        if len(selected) == top_n:
            break

    return selected

# 2. 時間衰減 (newer items get boost)
def time_aware_recommendations(item_id, top_n=5, time_decay=0.1):
    """考慮時間因素的推薦"""
    base_scores = similarity_matrix[item_id]

    # 時間權重 (假設 items 按時間排序)
    time_weights = np.exp(-time_decay * np.arange(len(base_scores)))

    # 加權分數
    weighted_scores = base_scores * time_weights

    # 排序
    top_indices = np.argsort(weighted_scores)[::-1][1:top_n+1]
    return top_indices
```

---

## 📈 評估指標

### 推薦系統評估

```python
# 1. 覆蓋率 (Coverage)
def calculate_coverage(all_recommendations, total_items):
    """計算推薦系統覆蓋率"""
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)

    coverage = len(unique_recommended) / total_items
    return coverage

# 2. 多樣性 (Diversity)
def calculate_diversity(recommendations, similarity_matrix):
    """計算推薦列表內部多樣性"""
    diversity_scores = []

    for recs in recommendations:
        # 計算列表內所有配對的相似度
        n = len(recs)
        if n < 2:
            continue

        avg_similarity = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                avg_similarity += similarity_matrix[recs[i]][recs[j]]
                count += 1

        diversity = 1 - (avg_similarity / count) if count > 0 else 0
        diversity_scores.append(diversity)

    return np.mean(diversity_scores)

# 3. 新穎度 (Novelty)
def calculate_novelty(recommendations, popularity):
    """計算推薦的新穎度 (推薦冷門項目能力)"""
    novelty_scores = []

    for recs in recommendations:
        # 推薦項目的平均流行度
        avg_popularity = np.mean([popularity[r] for r in recs])
        # 新穎度 = 1 - 流行度
        novelty = 1 - avg_popularity
        novelty_scores.append(novelty)

    return np.mean(novelty_scores)
```

---

## 🚀 進階主題

### 混合推薦系統

結合內容過濾與協同過濾:

```python
class HybridRecommender:
    def __init__(self, content_weight=0.6, collaborative_weight=0.4):
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

    def recommend(self, user_id, item_id, top_n=5):
        """
        混合推薦
        """
        # 內容過濾分數
        content_scores = content_based_score(item_id)

        # 協同過濾分數
        collaborative_scores = collaborative_filtering_score(user_id)

        # 加權合併
        hybrid_scores = (
            self.content_weight * content_scores +
            self.collaborative_weight * collaborative_scores
        )

        # 排序
        top_items = np.argsort(hybrid_scores)[::-1][:top_n]
        return top_items
```

---

## ✅ 檢查清單

完成專案後,確認:

- [ ] 理解 TF-IDF 原理與實作
- [ ] 掌握餘弦相似度計算
- [ ] 能夠建立基礎推薦系統
- [ ] 理解推薦系統評估指標
- [ ] 嘗試調整參數提升推薦質量
- [ ] (選) 實作多樣性優化
- [ ] (選) 擴展到混合推薦

---

**專案版本**: v1.0
**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team
**授權**: MIT License
