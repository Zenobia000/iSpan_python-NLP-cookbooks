# 評論情感分析專案

**專案類型**: 商業應用 - 情感分析與語言識別
**難度**: ⭐⭐⭐ 中級
**預計時間**: 3-5 小時

---

## 📋 專案概述

本專案展示如何構建**商業級評論分析系統**,包含:
1. 數據預處理流程
2. 語言自動識別
3. TensorFlow 模型訓練
4. 實際應用部署

---

## 🎯 學習目標

完成本專案後,您將能夠:

- ✅ 掌握大規模文本數據的預處理流程
- ✅ 實作多語言文本分類
- ✅ 使用 TensorFlow/Keras 訓練深度學習模型
- ✅ 建立端到端的情感分析系統
- ✅ 理解生產環境的數據處理規範

---

## 📊 專案檔案

### Notebook 1: 資料前處理
**檔名**: `專案_評論分析_資料前處理.ipynb`

**內容**:
1. 數據載入與探索
2. 缺失值處理
3. 文本清理 (去除 HTML, emoji, 特殊字符)
4. 語言偵測與分類
5. 數據標準化與匯出

**技術棧**:
- Pandas: 數據處理
- Regex: 文本清理
- langdetect: 語言識別

**輸入**: 原始評論 CSV
**輸出**: 清理後的評論數據

---

### Notebook 2: 語言識別模型
**檔名**: `專案_評論分析_語言識別_TensorFlow.ipynb`

**內容**:
1. 數據準備與 One-hot 編碼
2. 字符級 CNN 模型構建
3. 模型訓練與評估
4. 混淆矩陣分析
5. 實際應用範例

**技術棧**:
- TensorFlow 2.x
- Keras Sequential API
- Scikit-learn: 評估指標

**模型架構**:
```
Input (字符序列)
    ↓
Embedding Layer (字符嵌入)
    ↓
Conv1D (卷積提取特徵)
    ↓
MaxPooling (池化)
    ↓
Dense (全連接層)
    ↓
Softmax (語言分類)
```

---

## 🚀 快速開始

### 前置需求

```bash
# 確保已安裝必要套件
poetry add pandas tensorflow scikit-learn langdetect matplotlib seaborn
```

### 執行步驟

```bash
# 1. 進入專案目錄
cd 專案實戰/評論情感分析

# 2. 啟動 Jupyter
poetry run jupyter notebook

# 3. 按順序執行 notebooks:
#    ① 先執行資料前處理
#    ② 再執行語言識別
```

---

## 📁 數據說明

### 數據來源
- **來源**: Google 商家評論、電影評論
- **位置**: `datasets/google_reviews/` 或 `datasets/movie_reviews/`
- **格式**: CSV 或 Pickle
- **大小**: ~5,000 - 10,000 條評論

### 數據格式

```python
# 期望的 CSV 格式
{
    'text': '評論內容文本',
    'rating': 1-5,
    'language': 'en' or 'zh',
    'timestamp': '2024-01-01'
}
```

---

## 🎨 結果展示

### 預期輸出

#### 1. 資料前處理結果
```
✅ 原始數據: 10,000 筆
✅ 清理後: 9,500 筆
✅ 語言分布:
   - 英文: 6,800 (71.6%)
   - 中文: 2,700 (28.4%)
✅ 情感分布:
   - 正面: 5,700 (60%)
   - 負面: 3,800 (40%)
```

#### 2. 模型性能
```
語言識別準確率: 97.8%
訓練時間: ~5 分鐘 (CPU)
推理速度: ~10ms/樣本

分類報告:
              precision    recall  f1-score
English         0.98      0.98      0.98
Chinese         0.97      0.96      0.96
```

#### 3. 視覺化
- 語言分布餅圖
- 混淆矩陣熱圖
- 訓練歷史曲線 (Loss/Accuracy)

---

## 🔧 常見問題

### Q1: 數據集在哪裡?

```python
# 檢查數據路徑
import os
data_path = "../../datasets/google_reviews/BigCity_GoogleComments"

if os.path.exists(data_path):
    print(f"✅ 數據集存在: {data_path}")
else:
    print(f"❌ 數據集不存在")
    print("請檢查 datasets/ 目錄")
```

### Q2: 執行時記憶體不足

```python
# 減少數據量 (開發階段)
df = df.sample(n=5000, random_state=42)  # 只用 5000 筆

# 減少 batch size
model.fit(X_train, y_train, batch_size=32)  # 改為 16 或 8

# 使用 Google Colab
# 提供免費 GPU,記憶體更大
```

### Q3: TensorFlow 版本問題

```bash
# 確認 TensorFlow 版本
python -c "import tensorflow as tf; print(tf.__version__)"

# 如果版本不符,重新安裝
poetry add tensorflow==2.14.0
```

---

## 📈 擴展建議

### 進階功能

1. **多語言擴展**
   - 增加更多語言支持 (日文、韓文)
   - 使用 XLM-RoBERTa 多語言模型

2. **細粒度情感**
   - 1-5 星評分預測
   - 情感強度分析
   - 主客觀分類

3. **主題提取**
   - 整合 LDA 主題建模
   - 識別評論關注點 (價格、品質、服務)

4. **實時應用**
   - 建立 FastAPI 服務
   - 前端展示介面
   - 資料庫整合

### 作品集優化

```markdown
## 優化方向

1. **代碼品質**
   - 重構為模組化類別
   - 添加完整註解
   - 使用 Type Hints

2. **文檔完善**
   - 撰寫專案說明
   - 添加架構圖
   - 記錄實驗結果

3. **視覺化提升**
   - 互動式圖表 (Plotly)
   - 儀表板 (Streamlit)
   - 動態報告

4. **部署示範**
   - Docker 容器化
   - API 文檔 (Swagger)
   - 性能測試報告
```

---

## 📚 相關資源

### 延伸閱讀
- [TensorFlow Text Classification Tutorial](https://www.tensorflow.org/tutorials/keras/text_classification)
- [語言識別技術綜述](https://arxiv.org/abs/1708.04811)
- [情感分析最佳實踐](https://github.com/bentrevett/pytorch-sentiment-analysis)

### 相關課程章節
- CH03: 文本預處理
- CH05: 神經網路與深度學習入門
- CH08: Hugging Face 實戰

---

## ✅ 檢查清單

完成專案後,確認您已經:

- [ ] 成功執行所有 Cells
- [ ] 理解每個步驟的作用
- [ ] 嘗試修改參數並觀察影響
- [ ] 完成錯誤分析
- [ ] 思考如何應用到實際問題
- [ ] (選) 擴展功能
- [ ] (選) 撰寫技術報告

---

**專案版本**: v1.0
**建立日期**: 2025-10-17
**維護者**: iSpan NLP Team
**授權**: MIT License
