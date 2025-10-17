# 深入淺出 Python NLP：從基礎到生成式 AI 實戰

歡迎來到 **iSpan Python NLP Cookbooks**！這不僅僅是一份程式碼合輯，而是一套精心設計的互動式學習課程，旨在帶領您從零開始，一步步掌握自然語言處理（NLP）的核心技術，最終具備開發生成式 AI 應用的實戰能力。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/Poetry-1.2%2B-purple.svg)](https://python-poetry.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 這門課為何與眾不同？

在這個資訊爆炸的時代，我們相信最好的學習方式是「做中學」。本課程圍繞一系列真實世界的專案和案例，讓您不只學會「如何做」，更能理解「為何這麼做」。

*   **🎯 專案導向**：從歌詞分析、評論情感判斷，到智能問答系統，每個章節都圍繞一個實用專案。
*   **🤖 接軌前沿**：課程內容涵蓋最新的 Transformer、BERT 及檢索增強生成（RAG）技術，直通當前最熱門的生成式 AI 領域。
*   **🧱 循序漸進**：從環境設定、文本預處理，到經典機器學習、深度學習，再到大型語言模型，學習路徑清晰，適合所有程度的學習者。
*   **👐 開源實踐**：所有程式碼、教材、數據集完全開源，鼓勵您動手修改、實驗，甚至貢獻您的學習成果。

---

## 🚀 5 分鐘快速開始

準備好踏上 NLP 的奇妙旅程了嗎？只需幾個簡單步驟：

1.  **取得專案**：
    ```bash
    git clone https://github.com/iSpan/python-NLP-cookbooks.git
    cd python-NLP-cookbooks
    ```

2.  **安裝環境** (我們使用 [Poetry](https://python-poetry.org/) 管理依賴)：
    ```bash
    poetry install
    ```
    > 首次安裝可能需要一些時間下載模型與數據集，請耐心等候。

3.  **啟動 Jupyter Notebook**：
    ```bash
    poetry run jupyter notebook
    ```

4.  **開始學習**：
    打開瀏覽器，從 `課程資料/01_環境安裝與設定/` 開始您的第一堂課！

> 👉 想要更詳細的指引？請參考 [**QUICKSTART.md**](QUICKSTART.md)。

---

## 🗺️ 學習路徑圖

我們為您規劃了三條學習路徑，您可以根據自己的需求和時間安排來選擇：

| **路徑** | **適合對象** | **預計時間** | **學習重點** |
| :--- | :--- | :--- | :--- |
| 🎓 **系統學習** | NLP 新手、學生、想打好基礎的開發者 | 40-50 小時 | 從文本預處理到 Transformer，全面掌握 NLP 知識體系。 |
| 🚀 **快速實戰** | 有經驗的開發者、想快速應用的產品經理 | 10-15 小時 | 直接挑戰 `專案實戰` 中的文本分類、問答系統等專案。 |
| 🧠 **主題鑽研** | 對特定技術有興趣的進階學習者 | 5-10 小時 | 深入 `補充教材`，鑽研詞向量、文本相似度、CNN/LSTM 等主題。 |

> 👉 完整的課程大綱與內容，請見 [**COURSE_PLAN.md**](COURSE_PLAN.md)。

---

## 🛠️ 專案結構概覽

本專案採用模組化結構，讓您輕鬆找到所需資源：

```
iSpan_python-NLP-cookbooks/
│
├── 課程資料/         # 🎓 主線課程 (CH01-CH09)，從基礎到進階
│
├── 補充教材/         # 📚 針對特定主題的深入探討
│
├── 專案實戰/         # 🚀 端到端的真實世界專案
│
├── datasets/         # 💾 所有教學與專案所需的數據集
│
├── shared_resources/ # 🔧 共用的字典、字體、停用詞等資源
│
├── scripts/          # 🛠️ 用於環境檢查、數據下載的輔助腳本
│
└── README.md         # 📍 就是您正在閱讀的這份文件！
```

> 👉 想要了解每個資料夾的詳細用途？請參考 [**PROJECT_STRUCTURE.md**](PROJECT_STRUCTURE.md)。

---

## ✨ 專案亮點展示

在本課程中，您將親手打造以下酷炫的專案：

1.  **智能問答系統 (RAG)**
    *   **技術**：`Hugging Face`, `FAISS`, `LangChain`
    *   **成果**：打造一個能根據給定知識庫回答問題的智能助手，是構建企業級知識庫的核心技術。

2.  **新聞自動分類器 (BERT Fine-Tuning)**
    *   **技術**：`BERT`, `Transformers Trainer API`
    *   **成果**：微調大型語言模型，實現對新聞文章的高精度自動分類。

3.  **Google 評論情感分析**
    *   **技術**：`CNN`, `LSTM`, `Scikit-learn`
    *   **成果**：分析用戶評論，自動判斷其情感傾向（正面/負面/中性）與星等。

4.  **歌詞分析與生成**
    *   **技術**：`Jieba`, `Word2Vec`, `LSTM`
    *   **成果**：從周杰倫的歌詞中分析詞頻與主題，甚至嘗試生成新的歌詞。

---

## 🤝 如何貢獻

我們歡迎任何形式的貢獻！無論是修正錯字、優化程式碼、補充教材，或是提出新的專案想法，都對我們意義重大。

1.  **Fork** 這個專案。
2.  建立您的分支 (`git checkout -b feature/YourFeature`)。
3.  提交您的變更 (`git commit -m 'Add some feature'`)。
4.  推送到您的分支 (`git push origin feature/YourFeature`)。
5.  開啟一個 **Pull Request**。

---

## 📜 授權

本專案採用 MIT 授權。詳情請見 [LICENSE](LICENSE) 文件。

> **“The journey of a thousand miles begins with a single step.”**
>
> **現在，就讓我們一起邁出 NLP 學習的第一步吧！**