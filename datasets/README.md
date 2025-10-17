# iSpan Python NLP Cookbooks v2 - 數據集資源目錄

本目錄統一管理 NLP 課程所需的所有數據集資源。

---

## 📂 目錄結構

```
datasets/
├── README.md                    # 本說明文檔
├── honglou.txt                  # 紅樓夢全文 (中文語料)
├── glove/                       # GloVe 詞向量
│   ├── glove.6B.50d.txt
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   └── glove.6B.300d.txt
├── sms_spam/                    # SMS 垃圾簡訊數據集
├── twitter_sentiment/           # Twitter 情感分析數據集
├── food_delivery/               # 外送平台數據集
├── movielens/                   # MovieLens 推薦系統數據集
├── google_reviews/              # Google 商家評論 (專案自建)
└── lyrics/                      # 歌詞數據集
    └── 情歌歌詞/                # 290+ 首情歌歌詞 (專案自建)
```

---

## 📥 數據集下載方式

### 方法 1: 使用統一下載腳本 (推薦)

```bash
# 下載所有數據集
python scripts/download_datasets.py --all

# 只下載 Keras 內建數據集
python scripts/download_datasets.py --keras

# 只下載 Hugging Face 數據集
python scripts/download_datasets.py --huggingface

# 只下載中文語料
python scripts/download_datasets.py --chinese

# 查看完整選項
python scripts/download_datasets.py --help
```

### 方法 2: 手動下載

詳見各章節 notebook 內的數據載入指令。

---

## 📊 數據集清單

### 內建數據集 (自動下載)

這些數據集會在首次使用時自動下載到系統快取目錄:

| 數據集 | 來源 | 大小 | 快取位置 | 用途章節 |
|--------|------|------|----------|----------|
| IMDB | Keras | 80MB | `~/.keras/datasets/` | CH05, CH06, CH07 |
| Reuters | Keras | 2MB | `~/.keras/datasets/` | CH06 |
| 20 Newsgroups | sklearn | 18MB | `~/scikit_learn_data/` | CH04 |
| AG News | Hugging Face | 30MB | `~/.cache/huggingface/` | CH08 |
| CoNLL-2003 | Hugging Face | 3MB | `~/.cache/huggingface/` | CH08 |
| SQuAD 2.0 | Hugging Face | 50MB | `~/.cache/huggingface/` | 專案實戰 |

### 需手動下載數據集

| 數據集 | 來源 | 大小 | 下載方式 | 用途章節 |
|--------|------|------|----------|----------|
| **GloVe 6B** | Stanford NLP | 822MB | `python scripts/download_datasets.py --glove` | CH07 |
| **紅樓夢全文** | GitHub | 1MB | `python scripts/download_datasets.py --chinese` | CH03 |
| **Twitter Sentiment** | Kaggle | 238MB | `python scripts/download_datasets.py --kaggle` | CH08 |
| **Food Delivery** | Kaggle | 50MB | `python scripts/download_datasets.py --kaggle` | 專案實戰 |
| **MovieLens** | GroupLens | 1MB | `python scripts/download_datasets.py --movielens` | 專案實戰 |
| **CNN/DailyMail** | Hugging Face | 1.4GB | 使用時手動載入 `load_dataset('cnn_dailymail')` | CH08 |

### 專案自建數據集 (已存在)

| 數據集 | 位置 | 大小 | 說明 |
|--------|------|------|------|
| **情歌歌詞** | `datasets/lyrics/情歌歌詞/` | 5MB | 290+ 首繁體中文情歌 (用於 CH06 序列生成) |
| **Google 商家評論** | `datasets/google_reviews/` | 20MB | 真實商家評論數據 (用於 CH08 專案實戰) |

---

## 🔧 Kaggle 數據集設定

使用 Kaggle 數據集需要先設定 API Token:

### 步驟 1: 獲取 API Token

1. 前往 [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. 點擊 "Create New API Token"
3. 下載 `kaggle.json` 檔案

### 步驟 2: 設定 Token

**Windows**:
```bash
# 創建 .kaggle 目錄
mkdir %USERPROFILE%\.kaggle

# 複製 kaggle.json 到該目錄
copy kaggle.json %USERPROFILE%\.kaggle\

# 確認權限
```

**macOS/Linux**:
```bash
# 創建 .kaggle 目錄
mkdir -p ~/.kaggle

# 複製 kaggle.json
cp kaggle.json ~/.kaggle/

# 設定權限
chmod 600 ~/.kaggle/kaggle.json
```

### 步驟 3: 安裝 Kaggle API

```bash
pip install kaggle
```

### 步驟 4: 下載數據集

```bash
python scripts/download_datasets.py --kaggle
```

---

## 📝 使用範例

### 載入 IMDB 數據集 (Keras)

```python
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
print(f"訓練樣本: {len(x_train)}, 測試樣本: {len(x_test)}")
```

### 載入 AG News 數據集 (Hugging Face)

```python
from datasets import load_dataset

ag_news = load_dataset("ag_news")
print(f"訓練集: {len(ag_news['train'])} 新聞")
print(f"類別: {ag_news['train'].features['label'].names}")
```

### 載入紅樓夢語料 (本地檔案)

```python
from pathlib import Path

corpus_path = Path("datasets/honglou.txt")
text = corpus_path.read_text(encoding="utf-8")
print(f"紅樓夢全文字數: {len(text)}")
```

### 載入情歌歌詞 (本地目錄)

```python
from pathlib import Path

lyrics_dir = Path("datasets/lyrics/情歌歌詞/")
lyrics_files = list(lyrics_dir.glob("*.txt"))

all_lyrics = []
for file in lyrics_files:
    with open(file, 'r', encoding='utf-8') as f:
        all_lyrics.append(f.read())

print(f"載入 {len(all_lyrics)} 首情歌歌詞")
```

### 載入 GloVe 詞向量

```python
import numpy as np
from pathlib import Path

glove_path = Path("datasets/glove/glove.6B.100d.txt")
embeddings_index = {}

with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f"載入 {len(embeddings_index)} 個詞向量")
```

---

## ⚖️ 數據集授權聲明

**重要提醒**: 請遵守各數據集的授權條款

| 數據集 | 授權類型 | 商業使用 | 教學使用 | 連結 |
|--------|---------|---------|---------|------|
| IMDB | Academic Use | ❌ | ✅ | [Stanford](http://ai.stanford.edu/~amaas/data/sentiment/) |
| AG News | CC BY-SA 3.0 | ✅ | ✅ | [Hugging Face](https://huggingface.co/datasets/ag_news) |
| CoNLL-2003 | Research Use | ❌ | ✅ | [ACL](https://www.clips.uantwerpen.be/conll2003/ner/) |
| SQuAD 2.0 | CC BY-SA 4.0 | ✅ | ✅ | [Stanford](https://rajpurkar.github.io/SQuAD-explorer/) |
| GloVe | Public Domain | ✅ | ✅ | [Stanford NLP](https://nlp.stanford.edu/projects/glove/) |
| 20 Newsgroups | Public Domain | ✅ | ✅ | [UCI ML](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) |
| CNN/DailyMail | Research Use | ❌ | ✅ | [Papers with Code](https://paperswithcode.com/dataset/cnn-daily-mail-1) |
| MovieLens | Research Use | ❌ | ✅ | [GroupLens](https://grouplens.org/datasets/movielens/) |
| Twitter Sentiment140 | Research Use | ❌ | ✅ | [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) |

**本專案聲明**: 所有數據集僅用於教學與學術研究目的，不得用於商業用途。

---

## 🚨 問題排解

### 問題 1: Keras 數據集下載失敗

**解決方案**:
```python
# 手動指定鏡像站
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# 或使用代理
os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
```

### 問題 2: Hugging Face 數據集下載緩慢

**解決方案**:
```python
# 使用國內鏡像 (清華大學)
export HF_ENDPOINT=https://hf-mirror.com

# 或在代碼中設定
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 問題 3: Kaggle API 無法使用

**檢查清單**:
1. 確認 `kaggle.json` 位於 `~/.kaggle/` (Linux/Mac) 或 `%USERPROFILE%\.kaggle\` (Windows)
2. 確認檔案權限 (Linux/Mac 需要 `chmod 600 ~/.kaggle/kaggle.json`)
3. 確認已安裝 `kaggle` 套件: `pip install kaggle`

### 問題 4: 磁碟空間不足

**總數據量估算**:
- **必備數據** (P0): ~500MB (IMDB, Reuters, AG News, CoNLL-2003, 紅樓夢, 情歌歌詞)
- **推薦數據** (P1): ~1.2GB (+ GloVe)
- **完整數據** (P2): ~3GB (+ CNN/DailyMail, Twitter Sentiment, MovieLens)

**建議**:
- 根據需求選擇性下載
- 使用雲端環境 (Google Colab, Kaggle Notebooks)
- 定期清理不再使用的數據集

---

## 📚 相關文檔

- **完整數據規劃**: [docs/16_wbs_development_plan_template.md](../docs/16_wbs_development_plan_template.md) - Section 6
- **下載腳本**: [scripts/download_datasets.py](../scripts/download_datasets.py)
- **課程規劃**: [COURSE_PLAN.md](../COURSE_PLAN.md)
- **專案結構**: [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)

---

## 🤝 貢獻

如果您有推薦的優質 NLP 數據集，歡迎提出建議:

1. Fork 本專案
2. 新增數據集資訊到本文檔
3. 更新下載腳本 (如適用)
4. 提交 Pull Request

---

**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team
**版本**: v1.0
