# 歌詞分析系統

**專案類型**: 中文 NLP - 詞頻分析與視覺化
**難度**: ⭐⭐ 初級-中級
**預計時間**: 2-3 小時
**技術棧**: Jieba, WordCloud, Matplotlib

---

## 📋 專案概述

本專案使用 **290+ 首情歌歌詞**進行中文文本分析,展示:
- Jieba 中文斷詞技術
- 詞頻統計分析
- 文字雲視覺化
- 中文 NLP 完整流程

**適合**: NLP 初學者、中文文本處理入門

---

## 🎯 學習目標

- ✅ 掌握 Jieba 中文斷詞技巧
- ✅ 理解詞頻統計原理
- ✅ 學會使用 WordCloud 生成文字雲
- ✅ 處理繁體中文文本
- ✅ 實作完整的中文文本分析流程

---

## 📊 Notebooks 說明

### 1. 斷詞分析
**檔名**: `專案_歌詞分析_斷詞分析_Jieba.ipynb`

**核心內容**:
1. 載入 290+ 首情歌歌詞
2. 使用 Jieba 進行中文斷詞
3. 詞性標註 (POS Tagging)
4. 停用詞過濾
5. 斷詞結果分析

**關鍵技術**:
```python
import jieba
import jieba.posseg as pseg

# 精確模式斷詞
words = jieba.cut("我愛自然語言處理", cut_all=False)

# 詞性標註
words_with_pos = pseg.cut("我愛自然語言處理")
for word, flag in words_with_pos:
    print(f"{word}/{flag}")
```

---

### 2. 詞頻統計
**檔名**: `專案_歌詞分析_詞頻統計_Jieba.ipynb`

**核心內容**:
1. 批量讀取所有歌詞檔案
2. 統計詞頻分布
3. 找出高頻詞彙
4. 生成文字雲
5. 視覺化結果

**關鍵技術**:
```python
from collections import Counter
from wordcloud import WordCloud

# 詞頻統計
word_freq = Counter(all_words)
top_words = word_freq.most_common(50)

# 文字雲生成
wordcloud = WordCloud(
    font_path='../../shared_resources/fonts/jf-openhuninn-1.0.ttf',
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(dict(top_words))
```

---

## 📁 數據說明

### 歌詞數據集
- **位置**: `datasets/lyrics/情歌歌詞/`
- **數量**: 290+ 個 .txt 檔案
- **格式**: 純文本,UTF-8 編碼
- **語言**: 繁體中文

### 數據結構
```
datasets/lyrics/情歌歌詞/
├── 0_愛久見人心.txt
├── 0_戀.txt
├── 0_沈睡的森林.txt
├── ...
└── 139_要我的命.txt

每個檔案包含一首歌的完整歌詞
```

### 數據載入範例
```python
from pathlib import Path

# 讀取所有歌詞
lyrics_path = Path("../../datasets/lyrics/情歌歌詞")
lyrics_files = list(lyrics_path.glob("*.txt"))

all_lyrics = []
for file in lyrics_files:
    with open(file, 'r', encoding='utf-8') as f:
        lyrics = f.read()
        all_lyrics.append(lyrics)

print(f"✅ 載入 {len(all_lyrics)} 首歌詞")
```

---

## 🎨 預期結果

### 詞頻統計 TOP 20
```
愛     - 1,234 次
心     - 987 次
你     - 856 次
我     - 745 次
不     - 623 次
...
```

### 文字雲範例
生成繁體中文文字雲,高頻詞以大字體顯示,展現情歌主題詞彙。

### 詞性分布
```
名詞 (n):    35%
動詞 (v):    28%
形容詞 (a):  18%
副詞 (d):    12%
其他:        7%
```

---

## 🔧 常見問題

### Q1: 繁體中文字型問題

```python
# 問題: 文字雲顯示為方框

# 解決方案: 指定繁體中文字型
wordcloud = WordCloud(
    font_path='../../shared_resources/fonts/jf-openhuninn-1.0.ttf',  # 繁中字型
    width=800,
    height=400,
    background_color='white'
).generate(text)

# 確認字型檔案存在
from pathlib import Path
font_path = Path("../../shared_resources/fonts/jf-openhuninn-1.0.ttf")
print(f"字型存在: {font_path.exists()}")
```

### Q2: Jieba 斷詞不準確

```python
# 解決方案 1: 添加自訂詞典
jieba.load_userdict("custom_dict.txt")

# 解決方案 2: 強制分詞
jieba.suggest_freq('自然語言', True)
jieba.suggest_freq('語言處理', True)

# 解決方案 3: 使用繁體詞典
jieba.set_dictionary('../../shared_resources/dictionaries/dict.txt.big')
```

### Q3: 停用詞過濾

```python
# 載入繁中停用詞表
with open('../../shared_resources/stopwords/stopwords_zh_tw.txt', 'r') as f:
    stopwords = set(f.read().splitlines())

# 過濾
filtered_words = [w for w in words if w not in stopwords]
```

---

## 📈 擴展建議

### 初級擴展
- [ ] 分析不同歌手的用詞風格
- [ ] 比較不同年代的歌詞特色
- [ ] 統計情歌高頻主題詞

### 中級擴展
- [ ] 整合情感分析 (判斷歌詞情緒)
- [ ] 使用 LDA 主題建模
- [ ] 建立歌詞搜尋引擎 (TF-IDF)

### 進階擴展
- [ ] 訓練歌詞生成模型 (LSTM)
- [ ] 建立歌詞推薦系統
- [ ] 多維度分析儀表板

---

## 🏆 專案作品集建議

### 如何展示此專案

1. **GitHub README** 應包含:
   - 專案背景與動機
   - 技術架構圖
   - 核心代碼片段
   - 視覺化結果截圖
   - 主要發現 (Insights)

2. **展示重點**:
   - "分析 290+ 首情歌,發現'愛'字出現 1,234 次"
   - "使用 Jieba 處理繁體中文,準確率 XX%"
   - "生成互動式文字雲,展現情歌主題詞彙"

3. **技術亮點**:
   - 中文 NLP 處理經驗
   - 大規模文本處理能力
   - 數據視覺化技巧

---

## 📞 支援

- **Jieba 文檔**: https://github.com/fxsjy/jieba
- **WordCloud 文檔**: https://amueller.github.io/word_cloud/
- **課程相關**: 參考 `課程資料/03_文本預處理/`

---

**專案版本**: v1.0
**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team
