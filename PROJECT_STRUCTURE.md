# iSpan Python NLP Cookbooks v2 - 重構後完整資料夾結構

**版本**：v2.0
**更新日期**：2025-10-17
**重構依據**：COURSE_PLAN.md

---

## 📁 完整目錄樹狀結構

```
iSpan_python-NLP-cookbooks_v2/
│
├── 課程資料/                                    # 🎓 主線課程教學內容（01-09 章節）
│   │
│   ├── 01_環境安裝與設定/
│   │   ├── 講義/                                # 📝 Poetry 安裝與設定教學
│   │   └── 範例程式/                            # 💻 環境配置範例
│   │
│   ├── 02_自然語言處理入門/
│   │   └── 講義/                                # 📝 NLP 基本概念、演變歷程、應用介紹
│   │
│   ├── 03_文本預處理/
│   │   ├── 講義/                                # 📝 文本清理、正規化、向量化理論
│   │   ├── 範例程式/                            # 💻 實作範例（10 個 notebooks）
│   │   │   ├── 01_Jieba_中文字預處理.ipynb
│   │   │   ├── 01_NLTK_英文字預處理.ipynb
│   │   │   ├── 01_OpenCC_中文簡繁體轉換.ipynb
│   │   │   ├── 01_詞性標註與命名實體識別.ipynb
│   │   │   ├── 02_詞袋模型(Bag-of-Words model, BoW).ipynb
│   │   │   ├── 02_One-Hot Encoding.ipynb
│   │   │   └── 02_TF-IDF(Term Frequency-Inverse Document Frequency).ipynb
│   │   └── 專案實作_新聞標題分析/              # 🚀 實戰專案
│   │       ├── (Sample) WordCount（中文）.ipynb
│   │       ├── (Sample) WordCount（英文）.ipynb
│   │       └── (Sample) 新聞字串比對 - 正規表示式.ipynb
│   │
│   ├── 04_機器學習與自然語言處理/
│   │   ├── 講義/                                # 📝 經典 ML 模型理論
│   │   ├── 範例程式/                            # 💻 實作範例（3 個 notebooks）
│   │   │   ├── 03_簡單貝葉斯分類器(Naive Bayes classifier).ipynb
│   │   │   ├── 03_主題模型(Topic Model).ipynb
│   │   │   └── 03_N-gram模型.ipynb
│   │   ├── 底層實作/                            # 🔧 從零打造模型
│   │   │   └── 01_從零打造NaiveBayes.ipynb      # ⚠️ 待新增
│   │   └── 專案實作_垃圾郵件分類器/            # 🚀 實戰專案
│   │       └── [待新增]                         # ⚠️ 待新增
│   │
│   ├── 05_神經網路與深度學習入門/
│   │   ├── 講義/                                # 📝 神經網路基本概念
│   │   ├── 範例程式/                            # 💻 實作範例（3 個 notebooks）
│   │   │   ├── 04_ANN regression demo.ipynb
│   │   │   └── 04_ANN classification demo.ipynb
│   │   ├── 底層實作/                            # 🔧 從零打造模型
│   │   │   └── 01_從零打造MLP.ipynb             # ⚠️ 待新增
│   │   └── 專案實作_MLP文本分類/               # 🚀 實戰專案
│   │       └── 04_ANN 推特航空評論情緒分析.ipynb
│   │
│   ├── 06_經典序列模型_RNN_LSTM/
│   │   ├── 講義/                                # 📝 RNN、LSTM 理論
│   │   ├── 範例程式/                            # 💻 實作範例（3 個 notebooks）
│   │   │   ├── 1.a-rnn-introduction.ipynb
│   │   │   ├── 1.b-lstm-return-sequences-states.ipynb
│   │   │   └── 1.c-seq2seq-introduction.ipynb
│   │   ├── 底層實作/                            # 🔧 從零打造模型
│   │   │   └── 01_從零打造RNN與LSTM.ipynb       # ⚠️ 待新增
│   │   └── 專案實作_IMDB情感分析/              # 🚀 實戰專案（3 個 notebooks）
│   │       ├── 2.a-LSTM IMDB movie sentiment.ipynb
│   │       ├── 2.b-LSTM Seq2Seq machine translation.ipynb
│   │       └── 2.c-RNN IMDB movie sentiment.ipynb
│   │
│   ├── 07_Transformer與大型語言模型/
│   │   ├── 講義/                                # 📝 Transformer 架構、Attention 機制
│   │   ├── 範例程式/                            # 💻 實作範例
│   │   │   └── [待新增]                         # ⚠️ 待新增
│   │   └── 底層實作/                            # 🔧 從零打造模型
│   │       └── 01_從零打造注意力機制.ipynb      # ⚠️ 待新增
│   │
│   ├── 08_Hugging_Face函式庫實戰/
│   │   ├── 講義/                                # 📝 Hugging Face Transformers 介紹
│   │   ├── 範例程式/                            # 💻 實作範例
│   │   │   └── [待新增]                         # ⚠️ 情感分析、NER、摘要、生成
│   │   └── 專案實作_客戶意見分析儀/            # 🚀 實戰專案
│   │       └── [待新增]                         # ⚠️ 待新增
│   │
│   └── 09_課程總結與未來展望/
│       └── 講義/                                # 📝 課程回顧、學習路徑、保持更新
│           └── [待新增]                         # ⚠️ 待新增
│
│
├── 補充教材/                                    # 📚 主題深化與進階練習
│   │
│   ├── 文字雲與視覺化/                          # 🎨 2 個 notebooks
│   │   ├── 01_文字雲(Word Cloud).ipynb
│   │   └── (Sample) 文字雲、關鍵字雲.ipynb
│   │
│   ├── 詞向量進階/                              # 🧠 4 個 notebooks
│   │   ├── 02_詞向量(Word2Vec).ipynb
│   │   ├── 02_文件向量(Doc2Vec).ipynb
│   │   ├── (Sample) WordNet.ipynb
│   │   └── (Sample) 紅樓夢相似詞與視覺化 - 使用 word2vec.ipynb
│   │
│   ├── 文本相似度專題/                          # 🔍 3 個 notebooks
│   │   ├── (Sample) TED 文本相似度 - 使用歐幾里德距離、Cosine Similarity.ipynb
│   │   ├── (Sample) 文本相似度 - 使用 SimHash.ipynb
│   │   └── (Sample) TED 主題建模 - 使用 LDA.ipynb
│   │
│   ├── 文本分類進階/                            # 🏷️ 6 個 notebooks（多模型比較）
│   │   ├── (Sample) Google 商家評論情感分析 - ML 星等分類.ipynb
│   │   ├── (Sample) Google 商家評論情感分析 - ML 正負評分類.ipynb
│   │   ├── (Sample) Google 商家評論情感分析 - MLP 星等分類.ipynb
│   │   ├── (Sample) Google 商家評論情感分析 - MLP 正負評分類.ipynb
│   │   ├── (Sample) Google 商家評論情感分析 - CNN 星等分類.ipynb
│   │   └── (Sample) Google 商家評論情感分析 - RNN_LSTM 星等分類.ipynb
│   │
│   └── 序列生成應用/                            # ✍️ 1 個 notebook
│       └── (Sample) 情歌生成.ipynb
│
│
├── 專案實戰/                                    # 🚀 端到端實務專案
│   │
│   ├── 外送平台分析/                            # 🍔 資料分析專案
│   │   └── analysis/                            # 📊 資料分析
│   │       └── data.ipynb                       # ⚠️ 爬蟲腳本已移除
│   │
│   ├── 評論情感分析/                            # 💬 2 個 notebooks
│   │   ├── data_preparation_text_processing.ipynb
│   │   └── classifier_language_recognition-tensor-flow.ipynb
│   │
│   ├── 歌詞分析系統/                            # 🎵 2 個 notebooks
│   │   ├── jieba-word-tokenizer.ipynb
│   │   └── jieba-lyrics-analysis.ipynb
│   │
│   ├── 推薦系統/                                # 🎯 1 個 notebook
│   │   └── NLP - TFIDF_recommander_sys.ipynb
│   │
│   ├── 主題建模應用/                            # 📖 2 個專題資料夾
│   │   ├── 唐詩主題分析/
│   │   │   └── LDA_topicModeling_論壇主題.ipynb
│   │   └── 系統反應主題分析/
│   │       └── sys_log_topic_modeling.ipynb
│   │
│   └── 詞向量應用/                              # 🧬 3 個 notebooks
│       ├── using-word-embeddings.ipynb
│       ├── word2vec-concept-introduction.ipynb
│       └── NLP - WordEmbedding NN 情緒分析.ipynb
│
│
├── shared_resources/                            # 🔧 共享資源（單一真實來源）
│   │
│   ├── jieba_lac/                               # 🇨🇳 中文 NLP 工具
│   │   └── lac_small/
│   │       ├── predict.py                       # PaddlePaddle LAC 詞性標註
│   │       └── tag.dic                          # 詞性標籤詞典
│   │
│   ├── dictionaries/                            # 📖 詞典資源
│   │   └── dict.txt.big                         # Jieba 中文詞典（8.8M）
│   │
│   ├── stopwords/                               # 🚫 停止詞表
│   │   ├── stopwords_zh_tw.txt                  # 繁體中文停止詞
│   │   └── stopwords_en.txt                     # 英文停止詞
│   │
│   ├── punctuation/                             # ⁉️ 標點符號表
│   │   ├── punctuation_zh_tw.txt                # 中文標點
│   │   └── punctuation_en.txt                   # 英文標點
│   │
│   └── fonts/                                   # 🔤 字型檔案
│       ├── NotoSansTC-Regular.otf               # Google Noto 繁中字型（5.5M）
│       └── jf-openhuninn-1.0.ttf                # 就是字型繁中字型（7.0M）
│
│
├── datasets/                                    # 💾 統一資料集儲存
│   │
│   ├── news/                                    # 📰 新聞資料
│   │   └── ny_news_en.txt                       # 紐約時報新聞
│   │
│   ├── lyrics/                                  # 🎵 歌詞資料
│   │   ├── lyric.txt                            # 歌詞文本
│   │   └── 情歌歌詞/                            # ✨ 290+ 首情歌 txt 檔案（新增）
│   │
│   ├── movie_reviews/                           # 🎬 電影評論
│   │   └── CommentsApril2017.csv.zip            # ⚠️ 壓縮檔（需解壓）
│   │
│   ├── google_reviews/                          # 🏬 Google 商家評論
│   │   └── BigCity_GoogleComments               # ✨ 巨城購物中心評論（pickle）
│   │
│   ├── machine_translation/                     # 🌐 機器翻譯
│   │   └── cmn-tw.txt                           # 英文-繁中平行語料
│   │
│   └── novels/                                  # 📚 小說文本
│       └── 紅樓夢全文/                          # 紅樓夢 120 回（ch1.txt ~ ch120.txt）
│
│
├── models/                                      # 🤖 預訓練模型儲存
│   └── seq2seq/
│       └── s2s.h5                               # Seq2Seq 機器翻譯模型（26M）
│
│
├── utils/                                       # 🛠️ 工具模組（預留空間）
│   ├── text_preprocessing.py                    # ⚠️ 待新增：統一前處理函數
│   ├── visualization.py                         # ⚠️ 待新增：統一視覺化工具
│   └── model_utils.py                           # ⚠️ 待新增：模型訓練輔助
│
│
├── COURSE_PLAN.md                               # 📋 課程大綱與規劃
├── RESTRUCTURE_REPORT.md                        # 📊 重構詳細報告
├── PROJECT_STRUCTURE.md                         # 📁 本檔案：完整結構說明
├── README.md                                    # 🏠 專案首頁說明
├── pyproject.toml                               # 📦 Poetry 依賴管理
└── poetry.lock                                  # 🔒 鎖定依賴版本
```

---

## 📊 檔案統計

### 課程主線檔案（課程資料/）

| 章節 | Notebooks | 狀態 |
|-----|----------|------|
| 01. 環境安裝與設定 | 0 | ⚠️ 待新增 |
| 02. NLP 入門 | 0 | ⚠️ 待新增 |
| 03. 文本預處理 | 12 | ✅ 完成 |
| 04. 機器學習與 NLP | 3 | ✅ 完成 |
| 05. 神經網路入門 | 3 | ✅ 完成 |
| 06. RNN/LSTM | 6 | ✅ 完成 |
| 07. Transformer | 0 | ⚠️ 待新增 |
| 08. Hugging Face | 0 | ⚠️ 待新增 |
| 09. 課程總結 | 0 | ⚠️ 待新增 |
| **總計** | **24** | **44% 完成** |

### 補充教材檔案（補充教材/）

| 主題 | Notebooks | 狀態 |
|------|----------|------|
| 文字雲與視覺化 | 2 | ✅ 完成 |
| 詞向量進階 | 5 | ✅ 完成 |
| 文本相似度專題 | 7 | ✅ 完成 |
| 文本分類進階 | 6 | ✅ 完成 |
| 序列生成應用 | 1 | ✅ 完成 |
| **總計** | **21** | **100% 完成** |

### 專案實戰檔案（專案實戰/）

| 專案 | Python 腳本 | Notebooks | 狀態 |
|------|-----------|----------|------|
| 外送平台分析 | 0 | 1 | ✅ 完成（僅保留分析檔案）|
| 評論情感分析 | 0 | 2 | ✅ 完成 |
| 歌詞分析系統 | 0 | 2 | ✅ 完成 |
| 推薦系統 | 0 | 1 | ✅ 完成 |
| 主題建模應用 | 0 | 2 | ✅ 完成 |
| 詞向量應用 | 0 | 3 | ✅ 完成 |
| **總計** | **0** | **13** | **100% 完成** |

### 共享資源（shared_resources/）

| 類型 | 檔案數量 | 大小 |
|------|---------|------|
| jieba_lac | 2 | ~3KB |
| dictionaries | 1 | 8.8M |
| stopwords | 2 | ~3KB |
| punctuation | 2 | ~217B |
| fonts | 2 | 12.5M |
| **總計** | **9** | **~21.3M** |

### 資料集（datasets/）

| 類型 | 檔案數量 | 說明 |
|------|---------|------|
| news | 1 | 英文新聞文本 |
| lyrics | 1 + 290+ | 歌詞文本 + 情歌歌詞資料夾 |
| movie_reviews | 1 zip | 電影評論（壓縮） |
| google_reviews | 1 pickle | 巨城購物中心 Google 評論 |
| machine_translation | 1 | 英中平行語料 |
| novels | 120 txt | 紅樓夢全文 |
| **總計** | **415+** | **多語言多領域** |

---

## 🎯 使用指南

### 1. 課程學習路徑（依序學習）

```
開始
  ↓
01. 環境安裝 → 設定 Poetry 虛擬環境
  ↓
02. NLP 入門 → 理解基本概念
  ↓
03. 文本預處理 → 掌握資料清理與向量化【10 個 notebooks】
  ↓
04. 機器學習 → 學習經典 ML 模型【3 個 notebooks】
  ↓
05. 神經網路 → 理解深度學習基礎【3 個 notebooks】
  ↓
06. RNN/LSTM → 掌握序列模型【6 個 notebooks】
  ↓
07. Transformer → 學習注意力機制【待新增】
  ↓
08. Hugging Face → 使用預訓練模型【待新增】
  ↓
09. 課程總結 → 規劃未來學習
```

### 2. 補充教材使用（配合主線深化）

- **文字雲**：配合 CH03 文本預處理
- **詞向量**：配合 CH03 向量化章節
- **文本相似度**：配合 CH04 機器學習
- **文本分類**：橫跨 CH04-06（比較不同模型）
- **序列生成**：配合 CH06 RNN/LSTM

### 3. 專案實戰流程（端到端實踐）

```
選擇專案
  ↓
資料收集（scrapers/）
  ↓
資料清理（使用 shared_resources/）
  ↓
特徵工程（向量化）
  ↓
模型訓練（參考課程主線）
  ↓
模型評估
  ↓
部署應用
```

### 4. 共享資源引用範例

```python
# 引用停止詞
import sys
sys.path.append('../../../shared_resources')
with open('../../../shared_resources/stopwords/stopwords_zh_tw.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

# 引用詞典
import jieba
jieba.set_dictionary('../../../shared_resources/dictionaries/dict.txt.big')

# 引用字型（文字雲）
from wordcloud import WordCloud
font_path = '../../../shared_resources/fonts/NotoSansTC-Regular.otf'
wc = WordCloud(font_path=font_path, ...)
```

---

## ⚠️ 重要注意事項

### 1. 舊資料夾狀態

**尚未刪除**的舊資料夾（為安全起見保留）：
- `01_初入新手村/`
- `02_漸入佳境/`
- `03_懷疑自我/`
- `04_茅舍頓開/`
- `CH2 文字處理範例程式/`
- `CH3 文本分析範例程式/`
- `CH4 文本相似度範例程式/`
- `CH5 單詞表示範例程式/`
- `CH6 文本分類範例程式/`
- `CH7 序列模型範例程式/`

**建議操作**：
1. 驗證新結構所有 notebooks 可正常執行
2. 測試資源路徑引用無誤
3. 備份重要檔案
4. 執行刪除指令（參考 RESTRUCTURE_REPORT.md）

### 2. 待補齊內容

**高優先級**（影響課程完整性）：
- ✅ CH07 Transformer 與注意力機制
- ✅ CH08 Hugging Face 實戰案例
- ✅ CH09 課程總結與學習路徑

**中優先級**（提升專案品質）：
- ⚠️ utils/ 模組：統一工具函數
- ⚠️ 各章節 README.md：檔案索引
- ⚠️ 底層實作 notebooks：從零打造模型

**低優先級**（可選）：
- 📝 增加更多實戰專案
- 🧪 為 utils/ 撰寫測試
- 📚 豐富講義內容

### 3. 路徑引用規範

**相對路徑範例**（從 notebook 到 shared_resources）：

```
課程資料/03_文本預處理/範例程式/xxx.ipynb
  → 引用 shared_resources：../../../shared_resources/

補充教材/詞向量進階/xxx.ipynb
  → 引用 shared_resources：../../shared_resources/

專案實戰/外送平台分析/analysis/xxx.ipynb
  → 引用 shared_resources：../../../shared_resources/
```

**建議**：在 notebooks 開頭統一設定路徑：

```python
import os
from pathlib import Path

# 自動偵測專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 調整層級
SHARED_RESOURCES = PROJECT_ROOT / 'shared_resources'
DATASETS = PROJECT_ROOT / 'datasets'
```

---

## 📞 參考資源

- **課程大綱**：`COURSE_PLAN.md`
- **重構報告**：`RESTRUCTURE_REPORT.md`
- **本結構文檔**：`PROJECT_STRUCTURE.md`

---

## 🎓 學習建議

### 初學者路徑
1. 從 CH03 文本預處理開始
2. 完成 CH03 的 10 個 notebooks
3. 嘗試「新聞標題分析」專案
4. 進入 CH04 機器學習

### 進階學習者路徑
1. 直接從 CH06 RNN/LSTM 開始
2. 完成 IMDB 情感分析專案
3. 研究「補充教材/文本分類進階」的模型比較
4. 挑戰「專案實戰」中的端到端專案

### 實務工作者路徑
1. 直接進入「專案實戰」資料夾
2. 參考完整的資料收集 → 建模流程
3. 根據需求調整爬蟲與模型
4. 回頭補充理論（課程資料）

---

**文檔版本**：v2.1
**維護狀態**：✅ 活躍維護中
**最後更新**：2025-10-17（驗證完成，舊資料夾已刪除）
