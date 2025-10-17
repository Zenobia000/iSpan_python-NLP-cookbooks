# 專案結構指南

本文件詳細說明 **iSpan Python NLP Cookbooks** 專案的資料夾結構，幫助您快速定位所需的資源。

---

## 📁 資料夾核心功能概覽

```
/ (根目錄)
├── 課程資料/         # 🎓 主要學習路徑，包含從 CH01 到 CH09 的所有教學 Notebooks。
├── 補充教材/         # 📚 針對特定主題的深入學習材料，適合進階者。
├── 專案實戰/         # 🚀 端到端的完整專案，展示如何將 NLP 技術應用於真實世界。
├── datasets/         # 💾 所有課程與專案會用到的數據集。
├── shared_resources/ # 🔧 跨專案共用的資源，如自訂詞典、停用詞表、字體等。
├── scripts/          # 🛠️ 用於環境設定與資料準備的輔助腳本。
└── docs/             # 📝 存放本專案的所有說明文件。
```

---

## 🌳 詳細目錄樹狀結構

```
iSpan_python-NLP-cookbooks/
│
├── 課程資料/
│   ├── 01_環境安裝與設定/      # (Poetry, Jupyter)
│   ├── 02_自然語言處理入門/      # (NLP 基礎概念)
│   ├── 03_文本預處理/            # (Jieba, NLTK, TF-IDF)
│   ├── 04_機器學習與自然語言處理/  # (Naive Bayes, N-gram)
│   ├── 05_神經網路與深度學習入門/  # (MLP, Keras, Embedding)
│   ├── 06_經典序列模型_RNN_LSTM/   # (RNN, LSTM, Seq2Seq)
│   ├── 07_Transformer與大型語言模型/ # (Attention, BERT, GPT)
│   ├── 08_Hugging_Face函式庫實戰/ # (Pipeline, Fine-tuning)
│   └── 09_課程總結與未來展望/      # (技術選型, 職涯規劃)
│
├── 補充教材/
│   ├── 文本分類進階/            # (比較 ML, MLP, CNN, LSTM)
│   ├── 文本相似度專題/          # (Cosine, SimHash, LSI)
│   ├── 文字雲與視覺化/          # (WordCloud)
│   ├── 序列生成應用/            # (LSTM 情歌產生器)
│   └── 詞向量進階/              # (Word2Vec, Doc2Vec)
│
├── 專案實戰/
│   ├── 問答系統/                # (RAG, LangChain, FAISS)
│   ├── 文本分類系統/            # (BERT Fine-tuning)
│   ├── 推薦系統/                # (TF-IDF 內容過濾)
│   ├── 主題建模應用/            # (LDA, 唐詩分析)
│   ├── 新聞自動標籤系統/        # (Zero-Shot Classification)
│   └── ... (更多實戰專案)
│
├── datasets/                 # 數據集 (依專案或主題分類)
│   ├── google_reviews/          # (Google 商家評論)
│   ├── lyrics/                  # (歌詞數據)
│   ├── news/                    # (新聞文章)
│   └── ... (更多數據集)
│
├── shared_resources/         # 共用資源庫
│   ├── dictionaries/            # (Jieba 自訂詞典)
│   ├── fonts/                   # (文字雲使用的字體)
│   └── stopwords/               # (中英文停用詞)
│
├── scripts/                  # 輔助腳本
│   ├── check_environment.py     # (環境檢查)
│   └── download_datasets.py   # (數據集下載)
│
├── docs/                     # 專案文件
│   ├── COURSE_PLAN.md         # (課程大綱)
│   ├── PROJECT_STRUCTURE.md   # (本文件)
│   └── QUICKSTART.md          # (快速啟動指南)
│
├── pyproject.toml            # 📦 Poetry 專案設定檔，定義所有依賴
├── poetry.lock               # 🔒 鎖定依賴版本，確保環境一致性
└── README.md                 # 🏠 專案入口，提供整體介紹
```

---

## 💡 如何使用此結構

*   **學習者**：建議您從 `課程資料/` 開始，依照章節順序學習。當您對某個主題想有更深入的了解時，可以到 `補充教材/` 中尋找相關內容。
*   **開發者**：如果您想尋找特定問題的解決方案，可以直接查看 `專案實戰/` 中的範例，並從中借鑒程式碼與架構。
*   **資源尋找**：所有 Notebooks 中用到的數據集都存放在 `datasets/`，而共用的設定檔（如停用詞表）則在 `shared_resources/`。
