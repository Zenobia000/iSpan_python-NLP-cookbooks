# 文本分類系統 - BERT 微調實戰

**專案類型**: 深度學習 - Transformer 模型微調
**難度**: ⭐⭐⭐⭐ 進階
**預計時間**: 4-5 小時
**技術棧**: BERT, DistilBERT, Trainer API, Hugging Face

---

## 📋 專案概述

本專案展示如何使用 **BERT 模型微調** 技術構建生產級文本分類系統,應用於新聞文章自動分類場景。

### 核心技術

- **遷移學習**: 使用預訓練 BERT 模型
- **Fine-Tuning**: Trainer API 完整訓練流程
- **多類別分類**: 4 類新聞分類 (World, Sports, Business, Sci/Tech)
- **模型優化**: 量化、加速、部署

### 商業價值

- 📰 **新聞媒體**: 自動分類文章,提升編輯效率
- 📧 **郵件系統**: 智能郵件路由
- 🎫 **客服工單**: 自動工單分派
- 📚 **文檔管理**: 智能文檔歸檔

---

## 🎯 學習目標

- ✅ 掌握 BERT 模型微調完整流程
- ✅ 使用 Hugging Face Trainer API
- ✅ 實作多類別文本分類
- ✅ 評估模型性能 (混淆矩陣、F1 Score)
- ✅ 優化與部署模型

---

## 📊 數據集說明

### AG News Dataset

- **來源**: Hugging Face Datasets
- **規模**:
  - 訓練集: 120,000 samples
  - 測試集: 7,600 samples
- **類別**: 4 類 (World, Sports, Business, Sci/Tech)
- **語言**: 英文
- **平衡性**: 完全平衡 (每類 30,000 訓練樣本)

### 數據格式

```python
{
    'text': 'Wall St. Bears Claw Back Into the Black...',
    'label': 2  # 0:World, 1:Sports, 2:Business, 3:Sci/Tech
}
```

### 載入方式

```python
from datasets import load_dataset

dataset = load_dataset(\"ag_news\")
# 自動下載並快取到 ~/.cache/huggingface/datasets/
```

---

## 🚀 快速開始

### 環境需求

```bash
# 必需套件
poetry add transformers datasets torch
poetry add evaluate accelerate

# 可視化 (選用)
poetry add matplotlib seaborn scikit-learn

# API 部署 (選用)
poetry add fastapi uvicorn
```

### 運行專案

```bash
# 啟動 Jupyter
poetry run jupyter notebook

# 開啟 notebook
專案_文本分類_BERT微調實戰.ipynb

# 執行所有 cells (預計 30-60 分鐘,取決於硬體)
```

### 硬體需求

| 硬體 | 最低 | 推薦 | 說明 |
|------|------|------|------|
| **RAM** | 8GB | 16GB+ | 避免 OOM |
| **GPU** | 無 (可用 CPU) | GTX 1060+ | 加速 10-20x |
| **磁碟** | 2GB | 5GB+ | 模型 + 數據集 |
| **訓練時間** | 2-3 hr (CPU) | 10-15 min (GPU) | DistilBERT |

---

## 🎨 預期結果

### 模型性能

```
測試集結果:
================
Accuracy:  94.2%
F1-Score:  0.941
Precision: 0.943
Recall:    0.942

各類別表現:
            precision    recall  f1-score   support
World          0.93      0.95      0.94      1900
Sports         0.98      0.98      0.98      1900
Business       0.91      0.89      0.90      1900
Sci/Tech       0.94      0.95      0.94      1900
```

### 分類範例

```
Input: "Apple announces new iPhone with advanced AI features."
Output: Sci/Tech (Confidence: 98.7%)

Input: "Manchester United defeats Barcelona 3-1 in semifinals."
Output: Sports (Confidence: 99.2%)

Input: "Stock market reaches all-time high amid economic recovery."
Output: Business (Confidence: 96.5%)
```

---

## 🔧 技術亮點

### 1. Trainer API 優勢

- ✅ 自動化訓練循環
- ✅ 內建梯度累積
- ✅ 混合精度訓練 (FP16)
- ✅ 多 GPU 支持
- ✅ 檢查點管理
- ✅ Early Stopping
- ✅ TensorBoard 整合

### 2. 模型選擇對比

| 模型 | 參數量 | 訓練時間 | 準確率 | 推薦 |
|------|--------|---------|-------|------|
| BERT-base | 110M | 長 | 95.2% | 追求極致準確率 |
| DistilBERT | 66M | 中 | 94.2% | **平衡之選** ⭐ |
| ALBERT-base | 12M | 短 | 93.1% | 資源受限 |
| RoBERTa-base | 125M | 長 | 95.8% | 最佳性能 |

### 3. 超參數調優

```python
# 關鍵超參數
learning_rate: 2e-5      # BERT 推薦範圍: 2e-5 ~ 5e-5
batch_size: 16           # 依據 GPU 記憶體調整
epochs: 3                # 通常 2-5 epoch 即可
weight_decay: 0.01       # L2 正則化
warmup_steps: 500        # 學習率預熱
```

---

## 📈 性能對比

### DistilBERT vs Baseline

| 指標 | Baseline (Naive Bayes) | DistilBERT (微調後) | 提升 |
|------|---------------------|------------------|------|
| Accuracy | 88.3% | 94.2% | +5.9% |
| F1-Score | 0.875 | 0.941 | +7.5% |
| 推理速度 | 2ms | 15ms | -13ms |
| 模型大小 | 1MB | 260MB | - |

**結論**: 準確率大幅提升,但需權衡速度與大小

---

## 🔧 常見問題

### Q1: CUDA Out of Memory

```python
# 解決方案 1: 減少 batch size
per_device_train_batch_size=8  # 從 16 降到 8

# 解決方案 2: 梯度累積
gradient_accumulation_steps=4  # 模擬大 batch

# 解決方案 3: 使用梯度檢查點
model.gradient_checkpointing_enable()

# 解決方案 4: 使用更小的模型
model_name = "distilbert-base-uncased"  # 而非 bert-large
```

### Q2: 訓練太慢

```python
# 使用 Google Colab (免費 GPU)
# 或 Kaggle Notebooks

# 啟用混合精度訓練
training_args = TrainingArguments(
    fp16=True  # 需要 GPU
)

# 減少數據量 (開發階段)
train_dataset = dataset['train'].select(range(10000))  # 只用 10k
```

### Q3: 過擬合問題

```python
# Early Stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 增加 Dropout
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

# 數據增強 (回譯、同義詞替換)
```

---

## 📈 進階優化

### 1. 超參數搜索

```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        \"distilbert-base-uncased\",
        num_labels=4
    )

def hp_space(trial):
    return {
        \"learning_rate\": trial.suggest_float(\"learning_rate\", 1e-5, 5e-5, log=True),
        \"num_train_epochs\": trial.suggest_int(\"num_train_epochs\", 2, 5),
        \"per_device_train_batch_size\": trial.suggest_categorical(
            \"per_device_train_batch_size\", [8, 16, 32]
        )
    }

best_run = trainer.hyperparameter_search(
    direction=\"maximize\",
    backend=\"optuna\",
    hp_space=hp_space,
    n_trials=10
)
```

### 2. 集成學習

```python
# 訓練多個模型並投票
models = [
    \"distilbert-base-uncased\",
    \"roberta-base\",
    \"albert-base-v2\"
]

predictions_list = []
for model_name in models:
    # Train and predict
    predictions = train_and_predict(model_name)
    predictions_list.append(predictions)

# Majority voting
final_predictions = mode(predictions_list, axis=0)
```

---

## 🏆 作品集展示

### 技術亮點

1. **完整微調流程**: \"從零開始微調 BERT 模型,達到 94% 準確率\"
2. **生產部署**: \"FastAPI 服務化部署,支援實時分類\"
3. **性能優化**: \"模型量化減少 75% 大小,保持 93%+ 準確率\"
4. **錯誤分析**: \"深入分析 misclassifications,提出改進方向\"

### Demo 建議

- 展示訓練過程 (TensorBoard 曲線)
- 對比微調前後的性能
- 實時分類 Demo (API 或 Web 介面)
- 展示混淆矩陣與錯誤分析

---

**專案版本**: v1.0
**數據集**: AG News (120k samples)
**最佳準確率**: 94.2%
**最後更新**: 2025-10-17
