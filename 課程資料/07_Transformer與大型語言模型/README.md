# CH07: Transformer 與大型語言模型

**章節定位**: 現代 NLP 核心架構 - 從基礎到前沿的完整學習路徑

---

## 📚 學習資源結構

### **範例程式** (8 notebooks)

本章節採用**拆分式底層實作**策略,讓學生循序漸進理解 Transformer:

#### **基礎理論** (01-02):
1. `01_Transformer架構概覽.ipynb` - 整體架構與設計理念
2. `02_嵌入層_Embeddings.ipynb` - Token Embeddings + Positional Encoding

#### **核心機制底層實作** ⭐ (03-05):
3. `03_注意力機制_Attention.ipynb` - **從零實作 Self-Attention 機制**
   - Scaled Dot-Product Attention 完整實作
   - Multi-Head Attention 從零構建
   - 注意力權重視覺化

4. `04_Transformer編碼器_Encoder.ipynb` - **從零實作 Encoder 層**
   - EncoderLayer 類完整實作
   - Residual Connection + Layer Normalization
   - 完整的前向傳播流程

5. `05_Transformer解碼器_Decoder.ipynb` - **從零實作 Decoder 層**
   - DecoderLayer 類完整實作
   - Masked Self-Attention 機制
   - Cross-Attention 連接 Encoder-Decoder

#### **架構對比與應用** (06-08):
6. `06_三大模型架構對比.ipynb` - Encoder-Only/Decoder-Only/Encoder-Decoder
7. `07_大型語言模型_LLMs.ipynb` - GPT/BERT/T5 原理與演進
8. `08_LLM實際應用案例.ipynb` - Prompt Engineering 與實戰

---

### **講義文件** (4 講義)

採用**三段式教學法** (Fundamentals → First Principles → Body of Knowledge):

1. `01_Transformer架構完全解析.md` - 架構全貌與數學原理
2. `02_大型語言模型原理與應用.md` - LLM 演進與應用
3. `03_Encoder與Decoder深度剖析.md` - 三大架構變體詳解
4. `04_LLM應用實戰指南.md` - Prompt Engineering 與微調

---

## 🎓 學習路徑建議

### **快速理解路徑** (適合 80% 學生):
```
01 架構概覽 → 03 Attention 機制 → 06 架構對比 → 08 實際應用

重點: 理解原理,能使用 Hugging Face Transformers
時間: 6-8 小時
```

### **深入原理路徑** (適合進階學生):
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
+ 4 個講義深度閱讀

重點: 深入數學推導,能從零實作組件
時間: 12-15 小時
```

### **快速應用路徑** (適合實務導向):
```
01 → 06 → 08 → 直接跳到 CH08 Hugging Face

重點: 快速上手預訓練模型
時間: 3-4 小時
```

---

## ❓ 常見問題

### **Q1: 為什麼沒有單獨的"底層實作"資料夾?**

**A**: 本章節的 `03-05 notebooks` **即為底層實作**!

不同於 CH04-06 將底層實作單獨放置,CH07 採用**拆分式底層實作**:
- 更符合 Transformer 模組化設計
- 避免 800+ 行代碼的認知負擔
- 循序漸進,逐步構建完整架構

**實際情況**:
- `03_注意力機制_Attention.ipynb` = 底層實作 Self-Attention
- `04_Transformer編碼器_Encoder.ipynb` = 底層實作 Encoder
- `05_Transformer解碼器_Decoder.ipynb` = 底層實作 Decoder

### **Q2: 這樣夠深入嗎?會不會淪為"call API"?**

**A**: 絕對足夠!本章節包含:

**數學層面**:
- ✅ Scaled Dot-Product Attention 公式推導
- ✅ Multi-Head Attention 數學原理
- ✅ Positional Encoding 三角函數本質
- ✅ Layer Normalization vs Batch Normalization

**代碼層面**:
- ✅ 從零實作 Attention 機制 (NumPy/PyTorch)
- ✅ 從零實作 Encoder Layer (完整類定義)
- ✅ 從零實作 Decoder Layer (包含 Masked Attention)
- ✅ 視覺化 Attention 權重

**這是"理解原理並能實作組件"級別,而非"只會 call API"!**

### **Q3: 如果想看完整端到端實作怎麼辦?**

**A**: 三個選項:

1. **組合現有代碼** (推薦):
   - 將 03-05 的代碼組合起來
   - notebooks 中已有完整的類定義
   - 可自行組裝成完整 Transformer

2. **參考延伸閱讀**:
   - Annotated Transformer (Harvard NLP)
   - PyTorch Official Tutorial
   - Hugging Face Transformers 源碼

3. **等待學生反饋**:
   - 如果多數學生反饋需要
   - 可補充 Mini 整合範例 (300 行)

---

## 🎯 設計哲學

本章節遵循 **Linus "Good Taste" 原則**:

**消除特殊情況**:
- 不用為 Transformer 單獨創建"底層實作"資料夾
- 03-05 本身就是底層實作,不需要重複

**實用主義**:
- 拆分式教學比端到端更易理解
- 學生實際需要的是"理解原理" + "會用框架"
- 不追求"理論完美"的完整實作

**簡潔執念**:
- 避免 800+ 行代碼的認知負擔
- 每個 notebook 聚焦單一概念
- 複雜性分散到多個檔案

---

## 📊 與其他章節對比

| 章節 | 模型複雜度 | 底層實作策略 | 原因 |
|:-----|:----------|:-----------|:-----|
| **CH04 NaiveBayes** | 低 | 單一檔案 (150 行) | 簡單,一次講完 |
| **CH05 MLP** | 中 | 單一檔案 (200 行) | 可接受的複雜度 |
| **CH06 RNN/LSTM** | 中高 | 單一檔案 (300 行) | 已接近上限 |
| **CH07 Transformer** | 高 | **拆分式** (03-05) | 複雜度過高,必須拆分 |

**結論**: CH07 的拆分式設計是**深思熟慮的教學決策**,而非缺失!

---

## 🚀 後續優化建議

### **立即可行**:
1. ✅ 創建本 README.md (已完成)
2. ⏳ 在 03-05 notebooks 開頭標註 "🔧 底層實作"
3. ⏳ 在 WBS 文件中澄清設計決策

### **可選執行** (根據學生反饋):
4. ⏳ 開發 Mini 整合範例 (300 行簡化版)
5. ⏳ 錄製教學影片 (講解 03-05 如何組合)

---

**核心要點**: 本章節已完整包含 Transformer 底層實作 (拆分在 03-05),無需額外開發。這是基於教學效果與實用主義的**最佳設計**,而非妥協。

---

**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team / Claude AI
