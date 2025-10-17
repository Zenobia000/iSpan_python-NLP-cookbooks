# 2025 Python NLP 速成教案：直通生成式 AI 開發實戰

**版本**: v2.2
**更新日期**: 2025-10-17
**專案狀態**: 課程內容遷移完成，檔案命名已統一（56 files）

本課程專為具備 Python 基礎的工程師設計，旨在快速建立必要的自然語言處理 (NLP) 技能，以無縫接軌當前的生成式 AI (Generative AI) 開發浪潮。

課程設計遵循「第一原理 -> 核心基礎 -> 知識體系」的學習路徑，確保學習者不僅能掌握工具，更能理解其背後的核心思想，並將所學應用於真實世界的專案中。

---

## 課程完成度總覽

| 狀態 | 章節數 | 說明 |
|------|--------|------|
| ✅ 已實作 | 4 章 | CH03, CH04, CH05, CH06 |
| 📝 待新增 | 5 章 | CH01, CH02, CH07, CH08, CH09 |
| 📚 補充教材 | 21 筆 | 文字雲、詞向量、文本相似度等進階主題 |
| 🎯 專案實戰 | 13 筆 | 情感分析、推薦系統、主題建模等應用 |

---

### **1. Poetry 環境安裝與設定 (Poetry Environment Installation & Setup)** 📝 待新增
   - 安裝 Poetry
   - 使用 Poetry 建立專案與管理虛擬環境
   - 在 Poetry 環境中啟動 Jupyter Notebook

   **狀態**: 無範例程式 | **路徑**: `課程資料/01_環境安裝與設定/`

---

### **2. 自然語言處理入門 (Natural Language Processing 101)** 📝 待新增
   - 回顧自然語言處理（NLP）的基本概念
   - NLP 的演變歷程
   - NLP 的應用與 Python 函式庫介紹

   **狀態**: 無範例程式 | **路徑**: `課程資料/02_自然語言處理入門/`

---

### **3. 文本預處理 (Text Preprocessing)** ✅ 已實作
   - 文本清理（Cleaning）
   - 文本正規化（Normalization）
   - 語言學分析（Linguistic Analysis）
   - 文本向量化（Vectorization）
     - 使用詞頻（Word Counts）
     - 使用 TF-IDF 分數
   - **專案實作：新聞標題清理與分析**

   **狀態**: 12 個範例程式 | **路徑**: `課程資料/03_文本預處理/`
   - 包含: 斷詞處理、詞性標註、WordCount、TF-IDF、齊夫定律、搭配詞分析等

---

### **4. 機器學習與自然語言處理 (經典模型)** ✅ 已實作
   - **情感分析（Sentiment Analysis）**
     - 使用 VADER 庫判斷文本的正面或負面情緒
   - **文本分類（Text Classification）**
     - 使用樸素貝葉斯（Naïve Bayes）演算法對標記數據進行分類
   - **主題建模（Topic Modeling）**
     - 使用非負矩陣分解（Non-Negative Matrix Factorization, NMF）對未標記數據進行主題建模
   - **模型底層實作：從零打造樸素貝葉斯分類器** 📝 待新增
   - **專案實作：垃圾郵件分類器** 📝 待新增

   **狀態**: 3 個範例程式 | **路徑**: `課程資料/04_機器學習與自然語言處理/`
   - 包含: NaiveBayes、NMF、文本分類

---

### **5. 神經網路與深度學習入門 (Neural Networks & Deep Learning)** ✅ 已實作
   - 神經網路的基本概念
     - 層（Layers）、節點（Nodes）、權重（Weights）、激活函數（Activation Functions）
   - 深度學習的簡介
   - **模型底層實作：從零打造多層感知器 (MLP)** 📝 待新增
   - **專案實作：使用 MLP 進行文本分類** 📝 待新增

   **狀態**: 3 個範例程式 | **路徑**: `課程資料/05_神經網路與深度學習入門/`
   - 包含: Keras sequential、函數式API、嵌入層

---

### **6. 經典序列模型 (RNN & LSTM)** ✅ 已實作
   - 循環神經網路 (Recurrent Neural Networks, RNN)
   - 長短期記憶模型 (Long Short-Term Memory, LSTM)
   - **模型底層實作：從零打造 RNN 與 LSTM** 📝 待新增
   - **專案實作：IMDB 電影評論情感分析** 📝 待新增

   **狀態**: 6 個範例程式 | **路徑**: `課程資料/06_經典序列模型_RNN_LSTM/`
   - 包含: LSTM、Seq2Seq、Beam Search、IMDB情感分析

---

### **7. Transformer 與大型語言模型 (Transformers & LLMs)** 📝 待新增
   - Transformer 架構的主要部分
     - 嵌入層（Embeddings）
     - 注意力機制（Attention）
     - 前饋神經網路（Feedforward Neural Networks, FFNs）
   - Encoder-only、Decoder-only 與 Encoder-Decoder 模型的區別
   - 熱門大型語言模型（LLMs）介紹：BERT, GPT, Gemini, Claude
   - **模型底層實作：從零打造注意力機制 (Attention)**

   **狀態**: 無範例程式 | **路徑**: `課程資料/07_Transformer與大型語言模型/`

---

### **8. Hugging Face Transformers 函式庫實戰** 📝 待新增
   - 介紹 Hugging Face Transformers 函式庫
   - 使用預訓練大型語言模型（LLMs）執行以下 NLP 任務：
     - **情感分析（Sentiment Analysis）**
     - **命名實體識別（Named Entity Recognition, NER）**
     - **零樣本分類（Zero-Shot Classification）**
     - **文本摘要（Text Summarization）**
     - **文本生成（Text Generation）**
   - **專案實作：客戶意見分析儀**

   **狀態**: 無範例程式 | **路徑**: `課程資料/08_Hugging_Face函式庫實戰/`

---

### **9. 課程總結與未來展望 (NLP Review & Next Steps)** 📝 待新增
   - 回顧課程中涵蓋的 NLP 技術
   - 說明何時使用這些技術
   - 如何深入學習並保持最新

   **狀態**: 無範例程式 | **路徑**: `課程資料/09_課程總結與未來展望/`

---

## 實際專案資料夾結構（v2.1）

本專案採用三層架構，兼顧教學、進階學習與實務應用。所有共享資源統一管理，避免重複。

```
iSpan_python-NLP-cookbooks_v2/
├── 課程資料/                          # 主要課程內容（對應 CH01-09）
│   ├── 01_環境安裝與設定/               📝 待新增
│   ├── 02_自然語言處理入門/             📝 待新增
│   ├── 03_文本預處理/                   ✅ 12 notebooks
│   │   └── 專案實作_新聞標題分析/
│   │       ├── (Sample) 詞性標註與 WordCount（中文）.ipynb
│   │       ├── (Sample) 詞性標註與 WordCount（英文）.ipynb
│   │       ├── (Sample) TF-IDF.ipynb
│   │       └── ... (共 12 個)
│   ├── 04_機器學習與自然語言處理/       ✅ 3 notebooks
│   │   ├── (Sample) NaiveBayes.ipynb
│   │   ├── (Sample) NMF.ipynb
│   │   └── (Sample) 文本分類.ipynb
│   ├── 05_神經網路與深度學習入門/       ✅ 3 notebooks
│   │   ├── (Sample) Keras sequential.ipynb
│   │   ├── (Sample) Keras 函數式API.ipynb
│   │   └── (Sample) 嵌入層.ipynb
│   ├── 06_經典序列模型_RNN_LSTM/        ✅ 6 notebooks
│   │   ├── (Sample) LSTM.ipynb
│   │   ├── (Sample) Seq2Seq.ipynb
│   │   ├── (Sample) Beam Search.ipynb
│   │   └── ... (共 6 個)
│   ├── 07_Transformer與大型語言模型/    📝 待新增
│   ├── 08_Hugging_Face函式庫實戰/      📝 待新增
│   └── 09_課程總結與未來展望/          📝 待新增
│
├── 補充教材/                          # 進階與擴充主題（21 notebooks）
│   ├── 文字雲與視覺化/                  2 notebooks
│   │   ├── (Sample) 文字雲範例.ipynb
│   │   └── (Sample) jieba分詞.ipynb
│   ├── 詞向量進階/                      5 notebooks
│   │   ├── (Sample) Word2Vec演算法.ipynb
│   │   ├── (Sample) 相似詞 + 詞類比.ipynb
│   │   └── ... (共 5 個)
│   ├── 文本相似度專題/                  7 notebooks
│   │   ├── (Sample) 歐氏距離、曼哈頓距離、餘弦相似度.ipynb
│   │   ├── (Sample) Jaccard相似度.ipynb
│   │   └── ... (共 7 個)
│   ├── 文本分類進階/                    6 notebooks
│   │   ├── (Sample) BoW + TF-IDF.ipynb
│   │   ├── (Sample) 使用CNN進行文本分類.ipynb
│   │   └── ... (共 6 個)
│   └── 序列生成應用/                    1 notebook
│       └── (Sample) 情歌生成.ipynb
│
├── 專案實戰/                          # 完整應用案例（13 notebooks）
│   ├── 外送平台分析/                    1 notebook
│   │   └── analysis/data.ipynb
│   ├── 評論情感分析/                    2 notebooks
│   │   ├── data_preparation_text_processing.ipynb
│   │   └── data.ipynb
│   ├── 歌詞分析系統/                    2 notebooks
│   │   ├── lyric_tfidf.ipynb
│   │   └── lyric_word2vec.ipynb
│   ├── 推薦系統/                        1 notebook
│   │   └── 02_推薦系統_進階技巧_內容過濾_TF-IDF.ipynb
│   ├── 主題建模應用/                    2 notebooks
│   │   ├── 01_主題建模_LDA.ipynb
│   │   └── 02_主題建模_KMeans.ipynb
│   └── 詞向量應用/                      3 notebooks
│       ├── 03_潛在語意索引(Latent Semantic Indexing).ipynb
│       ├── 04_詞向量_Word2Vec.ipynb
│       └── 04_詞向量_GloVe.ipynb
│
├── shared_resources/                  # 🔧 共享資源（統一管理）
│   ├── jieba_lac/
│   │   └── lac_small/
│   │       ├── predict.py            # ⚠️ 原有 5 份重複，現統一為 1 份
│   │       └── tag.dic               # LAC 詞性標註字典
│   ├── dictionaries/
│   │   └── dict.txt.big              # 繁體中文詞典 (8.8M)
│   ├── stopwords/
│   │   ├── zh_tw/stopwords.txt       # 繁體中文停用詞
│   │   └── en/stopwords.txt          # 英文停用詞
│   ├── punctuation/
│   │   ├── zh_tw/punctuation.txt     # 中文標點符號
│   │   └── en/punctuation.txt        # 英文標點符號
│   └── fonts/
│       ├── NotoSansTC-Regular.otf    # 5.9M
│       └── NotoSerifTC-Regular.otf   # 6.6M
│
├── datasets/                          # 📊 統一數據集管理
│   ├── news/                          # 新聞數據（標題分析）
│   ├── lyrics/
│   │   ├── lyric.txt                 # 歌詞數據（單檔）
│   │   └── 情歌歌詞/                  # 290+ 情歌 txt 檔案（序列生成用）
│   ├── movie_reviews/
│   │   └── CommentsApril2017.csv.zip # 電影評論數據
│   ├── google_reviews/
│   │   └── BigCity_GoogleComments    # 巨城購物中心 Google 評論（pickle）
│   ├── machine_translation/          # 機器翻譯數據
│   └── novels/
│       └── 紅樓夢全文/                 # 古典小說全文
│
├── models/                            # 🧠 預訓練模型存放
│   └── seq2seq/                      # Seq2Seq 模型檔案
│
├── utils/                             # 🛠️ 工具函數（待新增）
│   ├── text_preprocessing.py         # 文本預處理工具
│   ├── visualization.py              # 視覺化工具
│   └── model_utils.py                # 模型輔助函數
│
├── docs/                              # 📝 專案文檔
│   ├── PROJECT_STRUCTURE.md          # 專案結構說明
│   ├── COURSE_PLAN.md                # 課程規劃（本檔案）
│   ├── RESTRUCTURE_REPORT.md         # 重構報告
│   └── VERIFICATION_REPORT.md        # 驗證報告
│
├── pyproject.toml                     # Poetry 專案設定
├── poetry.lock                        # 依賴鎖定
└── README.md                          # 專案說明

```

---

## 📋 重要注意事項

### 1. 檔案命名規範 ⭐ 已統一
**所有 56 個 notebooks 已重新命名，符合以下規範：**

- **課程資料**: `{編號}_{技術名稱}_{應用場景}.ipynb`
  - 例：`01_Jieba中文斷詞_詞性標註.ipynb`, `04_LSTM情感分析_IMDB電影評論.ipynb`
- **補充教材**: `進階{編號}_{主題}_{技術細節}.ipynb`
  - 例：`進階02_詞向量_Word2Vec訓練.ipynb`, `進階06_LSTM分類_Google評論星等.ipynb`
- **專案實戰**: `專案_{領域}_{具體任務}_{技術方法}.ipynb`
  - 例：`專案_評論分析_資料前處理.ipynb`, `專案_推薦系統_內容過濾_TFIDF.ipynb`

**詳細命名規範**: 請參閱 `docs/NAMING_CONVENTION.md`
**重命名映射表**: 請參閱 `docs/RENAME_MAPPING.md`

### 2. 共享資源使用原則
所有 notebooks 應統一引用 `shared_resources/` 中的資源，避免重複複製：

```python
# ✅ 正確做法：使用相對路徑引用共享資源
import sys
sys.path.append('../../shared_resources/jieba_lac/lac_small')
from predict import LAC

# ✅ 載入停用詞
with open('../../shared_resources/stopwords/zh_tw/stopwords.txt', 'r') as f:
    stopwords = f.read().splitlines()
```

### 3. 數據集管理
- 所有數據集統一存放於 `datasets/` 目錄
- 原始數據與處理後數據分開管理
- 大型數據集使用壓縮格式（zip, gz）

### 4. 課程開發優先順序
**Phase 1 (已完成)**: CH03, CH04, CH05, CH06 - 基礎文本處理與模型
**Phase 2 (待開發)**: CH01, CH02 - 環境設定與入門
**Phase 3 (待開發)**: CH07, CH08 - Transformer 與 Hugging Face
**Phase 4 (待開發)**: 底層實作 notebooks（3 個）
**Phase 5 (待開發)**: CH09 - 課程總結

### 5. 版本紀錄
- **v2.0 (2025-10-17)**: 完成專案重構，統一資料夾結構
- **v2.1 (2025-10-17)**: 完成檔案驗證，刪除舊資料夾，更新文檔
- **v2.2 (2025-10-17)**: 完成所有檔案命名統一（56 notebooks），制定命名規範

---

## 🚀 快速開始

```bash
# 1. 安裝 Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. 安裝專案依賴
poetry install

# 3. 啟動 Jupyter Notebook
poetry run jupyter notebook

# 4. 開始學習
# 建議順序：課程資料 → 補充教材 → 專案實戰
```

---

**最後更新**: 2025-10-17 | **維護者**: iSpan NLP Team
