# 客戶意見分析儀 - 完整商業應用專案

**專案類型**: Hugging Face 綜合應用 - 端到端商業系統
**難度**: ⭐⭐⭐⭐ 進階
**預計時間**: 3-4 小時
**技術棧**: Transformers, Pipeline API, Pandas, Matplotlib

---

## 📋 專案概述

本專案是 **CH08 Hugging Face 實戰** 的綜合應用專案,整合前面學習的所有技術,構建一個完整的商業級客戶意見分析系統。

### 核心功能

1. **情感分析**: 自動判斷評論正負面
2. **主題識別**: 分類問題類型 (產品、服務、物流等)
3. **關鍵字提取**: 找出高頻議題
4. **趨勢分析**: 追蹤滿意度變化
5. **可視化儀表板**: 管理層決策支持

### 商業價值

- 💰 **成本節省**: 減少 90% 人工審閱時間
- ⚡ **即時反饋**: 快速發現產品/服務問題
- 📊 **數據驅動**: 客觀量化客戶滿意度
- 🎯 **精準改進**: 定位改善重點領域

---

## 🎯 學習目標

完成本專案後,您將能夠:

- ✅ 整合多個 Hugging Face Pipeline
- ✅ 構建端到端的 NLP 分析系統
- ✅ 處理真實商業數據
- ✅ 生成可視化分析報告
- ✅ 部署生產級應用

---

## 📊 系統架構

```
原始評論數據 (CSV)
    ↓
數據預處理
    ├── 去除噪音 (HTML, emoji)
    ├── 文本清理
    └── 語言識別
    ↓
NLP 分析引擎
    ├── 情感分析 (Sentiment Analysis)
    ├── 主題分類 (Zero-Shot Classification)
    ├── 關鍵字提取 (NER + TF-IDF)
    └── 情緒檢測 (Emotion Detection)
    ↓
數據聚合與統計
    ├── 情感分布計算
    ├── 主題熱點分析
    ├── 趨勢變化追蹤
    └── 異常評論識別
    ↓
可視化輸出
    ├── 情感分布餅圖
    ├── 主題熱點圖
    ├── 詞雲 (Word Cloud)
    ├── 趨勢線圖
    └── 綜合儀表板
```

---

## 🚀 快速開始

### 前置需求

```bash
# 安裝必要套件
poetry add transformers datasets torch
poetry add pandas matplotlib seaborn wordcloud
```

### 運行專案

```bash
# 進入目錄
cd 課程資料/08_Hugging_Face函式庫實戰/專案實作_客戶意見分析儀

# 啟動 Jupyter
poetry run jupyter notebook

# 開啟 notebook
09_專案實戰_客戶意見分析儀.ipynb
```

---

## 📁 專案檔案

### Notebook
**檔名**: `09_專案實戰_客戶意見分析儀.ipynb`

**章節結構**:
1. Part 1: 專案需求與架構設計
2. Part 2: 數據載入與探索性分析 (EDA)
3. Part 3: 情感分析模組
4. Part 4: 主題分類模組
5. Part 5: 關鍵字提取模組
6. Part 6: 數據聚合與統計
7. Part 7: 可視化儀表板
8. Part 8: 完整系統整合
9. Part 9: 部署與優化
10. Part 10: 總結與擴展

---

## 📊 數據說明

### 數據來源
- **位置**: `datasets/google_reviews/` 或自訂數據
- **格式**: CSV 或 Pandas DataFrame
- **建議大小**: 500+ 條評論

### 數據格式範例

```python
{
    'review_id': 'R001',
    'text': '產品質量很好，但是物流太慢了',
    'rating': 3,  # 1-5 星
    'date': '2024-01-15',
    'product_id': 'P123'
}
```

### 載入數據

```python
import pandas as pd

# 方法 1: 從 CSV 載入
df = pd.read_csv('../../datasets/google_reviews/reviews.csv')

# 方法 2: 從 Pickle 載入
df = pd.read_pickle('../../datasets/google_reviews/BigCity_GoogleComments')

# 方法 3: 使用 Hugging Face datasets
from datasets import load_dataset
dataset = load_dataset("csv", data_files="reviews.csv")
```

---

## 🎨 預期產出

### 1. 情感分析報告

```
📊 情感分析總結
====================
總評論數: 1,000
正面評論: 650 (65.0%)
負面評論: 350 (35.0%)

平均信心度: 91.3%
```

### 2. 主題分析

```
🎯 主題分布
====================
產品質量:  380 (38.0%)
客戶服務:  280 (28.0%)
物流配送:  210 (21.0%)
價格相關:  130 (13.0%)
```

### 3. 關鍵字分析

**正面關鍵詞 TOP 10**:
```
good, excellent, fast, quality, recommend,
satisfied, great, amazing, helpful, perfect
```

**負面關鍵詞 TOP 10**:
```
bad, slow, poor, terrible, disappointed,
waste, wrong, broken, defective, refund
```

### 4. 視覺化

- **情感分布餅圖**: 正負面比例
- **主題熱點圖**: 各主題評論數量
- **詞雲**: 高頻詞彙
- **趨勢圖**: 每日/每週滿意度變化

---

## 💡 核心代碼框架

### 完整系統類別

```python
from transformers import pipeline
import pandas as pd

class CustomerFeedbackAnalyzer:
    """
    完整的客戶意見分析系統
    """
    def __init__(self):
        # 載入 Pipelines
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.topic_classifier = pipeline("zero-shot-classification")
        self.emotion_detector = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )

    def analyze_reviews(self, reviews_df):
        """
        完整分析流程
        """
        # 1. 情感分析
        sentiments = self.sentiment_analyzer(reviews_df['text'].tolist())

        # 2. 主題分類
        topics = self._classify_topics(reviews_df['text'].tolist())

        # 3. 關鍵字提取
        keywords = self._extract_keywords(reviews_df['text'].tolist())

        # 4. 生成報告
        report = self._generate_report(sentiments, topics, keywords)

        return report
```

---

## 🔧 常見問題

### Q1: Google 評論數據集在哪裡?

```python
# 檢查數據位置
import os
from pathlib import Path

data_path = Path("../../datasets/google_reviews/BigCity_GoogleComments")
print(f"數據存在: {data_path.exists()}")

# 如果不存在,使用替代數據或自行創建
```

### Q2: Pipeline 太慢怎麼辦?

```python
# 使用批次處理
results = sentiment_analyzer(
    texts,
    batch_size=32  # 批次處理加速
)

# 使用 GPU
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    device=0  # 使用 GPU 0
)
```

### Q3: 記憶體不足

```python
# 分批處理大數據集
def analyze_in_chunks(df, chunk_size=100):
    results = []

    for i in range(0, len(df), chunk_size):
        chunk = df[i:i+chunk_size]
        chunk_results = analyzer.analyze(chunk)
        results.append(chunk_results)

    return pd.concat(results)
```

---

## 📈 擴展建議

### 初級擴展
- [ ] 增加多語言支持 (中文評論)
- [ ] 細粒度情感 (1-5 星預測)
- [ ] 導出 Excel/PDF 報告

### 中級擴展
- [ ] 實時監控告警 (負面評論激增)
- [ ] 競品對比分析
- [ ] 情感變化趨勢預測

### 進階擴展
- [ ] 建立 Streamlit 儀表板
- [ ] FastAPI 服務化部署
- [ ] 整合數據庫 (PostgreSQL)
- [ ] 自動生成改進建議

---

## 🏆 作品集展示

### 展示重點

1. **系統化思維**: "設計完整的商業級分析系統"
2. **技術整合**: "整合情感分析、主題分類、NER 等多項技術"
3. **實際應用**: "可直接應用於電商、客服等場景"
4. **視覺化**: "生成高質量分析報告與儀表板"

### Demo 建議

- 準備 100+ 條真實評論數據
- 展示完整分析流程 (3-5 分鐘)
- 重點展示視覺化結果
- 說明商業價值與應用場景

---

**專案版本**: v1.0
**對應章節**: CH08 Hugging Face 實戰
**Notebook**: 1
**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team
