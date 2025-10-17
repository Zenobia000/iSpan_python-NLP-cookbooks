# iSpan Python NLP Cookbooks v2 - 專案使用指南

**版本**: v1.0
**最後更新**: 2025-10-17
**適用對象**: 學生、講師、開發者

---

## 📋 目錄

1. [快速開始](#1-快速開始)
2. [專案結構導覽](#2-專案結構導覽)
3. [課程學習路徑](#3-課程學習路徑)
4. [環境設置完整指南](#4-環境設置完整指南)
5. [常見問題排解](#5-常見問題排解)
6. [進階使用技巧](#6-進階使用技巧)
7. [教師指南](#7-教師指南)
8. [貢獻指南](#8-貢獻指南)

---

## 1. 快速開始

### 1.1 5 分鐘快速啟動

```bash
# Step 1: Clone 專案 (或從 USB 複製)
git clone <repository-url>
cd iSpan_python-NLP-cookbooks_v2

# Step 2: 安裝 Poetry (如果尚未安裝)
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Step 3: 安裝專案依賴
poetry install

# Step 4: 啟動 Jupyter Notebook
poetry run jupyter notebook

# Step 5: 開啟任一 notebook 開始學習!
```

### 1.2 第一個 Notebook 體驗

推薦從這些 notebooks 開始:

1. **入門體驗**: `課程資料/02_自然語言處理入門/範例程式/01_什麼是NLP.ipynb`
2. **快速實戰**: `課程資料/03_文本預處理/範例程式/06_Jieba中文斷詞_基礎操作.ipynb`
3. **進階應用**: `課程資料/08_Hugging_Face函式庫實戰/01_Hugging_Face生態簡介.ipynb`

---

## 2. 專案結構導覽

### 2.1 三層架構概覽

```
iSpan_python-NLP-cookbooks_v2/
│
├── 📚 課程資料/              # 核心教學內容 (CH01-09)
│   ├── 01_環境安裝與設定/
│   ├── 02_自然語言處理入門/
│   ├── 03_文本預處理/
│   ├── 04_機器學習與自然語言處理/
│   ├── 05_神經網路與深度學習入門/
│   ├── 06_經典序列模型_RNN_LSTM/
│   ├── 07_Transformer與大型語言模型/
│   ├── 08_Hugging_Face函式庫實戰/
│   └── 09_課程總結與未來展望/
│
├── 🎓 補充教材/              # 進階主題與擴展
│   ├── 文字雲與視覺化/
│   ├── 詞向量進階/
│   ├── 文本相似度專題/
│   ├── 文本分類進階/
│   └── 序列生成應用/
│
├── 🚀 專案實戰/              # 完整商業應用案例
│   ├── 評論情感分析/
│   ├── 歌詞分析系統/
│   ├── 推薦系統/
│   ├── 主題建模應用/
│   └── 詞向量應用/
│
├── 🔧 shared_resources/      # 共享資源
│   ├── jieba_lac/           # Jieba 與 LAC
│   ├── dictionaries/        # 詞典
│   ├── stopwords/           # 停用詞表
│   ├── punctuation/         # 標點符號表
│   └── fonts/               # 中文字型
│
├── 📊 datasets/              # 統一數據集管理
│   ├── news/                # 新聞數據
│   ├── lyrics/              # 歌詞數據
│   ├── movie_reviews/       # 電影評論
│   ├── google_reviews/      # Google 評論
│   └── novels/              # 小說語料
│
└── 📝 docs/                  # 專案文檔
    ├── PROJECT_STRUCTURE.md
    ├── COURSE_PLAN.md
    ├── NAMING_CONVENTION.md
    └── PROJECT_USAGE_GUIDE.md (本文件)
```

### 2.2 各層級功能說明

#### **課程資料 (Main Course)**
- **目的**: 系統化教學,從基礎到前沿
- **內容**: CH01-09,共 57 notebooks + 20 講義
- **特色**: 每章包含:
  - `範例程式/`: 可執行的教學範例
  - `講義/`: Markdown 格式完整講義
  - `底層實作/`: 從零實作核心算法 (NaiveBayes, MLP, RNN/LSTM)
  - `專案實作_*/`: 章節綜合應用專案

#### **補充教材 (Advanced Topics)**
- **目的**: 深化特定主題
- **內容**: 21 notebooks
- **適合**: 完成課程資料後進階學習

#### **專案實戰 (Real-World Projects)**
- **目的**: 商業應用訓練
- **內容**: 10 個完整專案
- **適合**: 準備求職作品集

---

## 3. 課程學習路徑

### 3.1 零基礎學習路徑 (8-12 週)

```
Week 1-2: 環境與基礎
├─ CH01: 環境安裝與設定
├─ CH02: NLP 入門概念
└─ CH03: 文本預處理 (重點!)

Week 3-4: 傳統機器學習
├─ CH04: 機器學習與 NLP
│   └─ 底層實作: NaiveBayes
└─ 專案: 垃圾郵件分類器

Week 5-6: 深度學習基礎
├─ CH05: 神經網路入門
│   └─ 底層實作: MLP
└─ CH06: RNN/LSTM
    └─ 底層實作: RNN & LSTM

Week 7-8: 前沿技術
├─ CH07: Transformer 與 LLMs
└─ CH08: Hugging Face 實戰 ⭐ 重點!

Week 9-10: 綜合應用
├─ 補充教材: 選擇感興趣主題
└─ 專案實戰: 完成 2-3 個專案

Week 11-12: 總結與求職準備
├─ CH09: 課程總結
└─ 建立個人作品集
```

### 3.2 有基礎快速通道 (4-6 週)

```
Week 1: 快速複習
└─ CH03-06: 跳過基礎,重點看底層實作

Week 2-3: 前沿技術
├─ CH07: Transformer (重點理解 Attention)
└─ CH08: Hugging Face (務必精通!)

Week 4: 專案實戰
└─ 完成 3 個專案實戰

Week 5-6: 作品集與求職
└─ 建立 GitHub 作品集
```

### 3.3 講師教學建議路徑

```
第 1 堂課 (3hr): 環境與入門
├─ CH01: 環境設置 (1hr)
├─ CH02: NLP 概念 (1hr)
└─ CH03: 文本預處理示範 (1hr)

第 2 堂課 (3hr): 傳統 NLP
├─ CH04: 機器學習 (1.5hr)
└─ NaiveBayes 底層實作 (1.5hr)

第 3 堂課 (3hr): 深度學習
├─ CH05: 神經網路 (1hr)
├─ MLP 底層實作 (1hr)
└─ CH06: RNN/LSTM 簡介 (1hr)

第 4 堂課 (3hr): Transformer
└─ CH07: 完整講解 (3hr) ⭐ 重點!

第 5-6 堂課 (6hr): Hugging Face
├─ Pipeline API (2hr)
├─ 模型微調 (2hr)
└─ 實戰專案 (2hr)

第 7 堂課 (3hr): 專案實戰
└─ 學生分組完成專案

第 8 堂課 (3hr): 總結與求職
└─ CH09 + 作品集指導
```

---

## 4. 環境設置完整指南

### 4.1 Poetry 安裝與配置

#### **Windows 安裝**

```powershell
# PowerShell (管理員權限)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# 添加到 PATH (如果未自動添加)
$env:Path += ";$env:APPDATA\Python\Scripts"

# 驗證安裝
poetry --version
```

#### **macOS/Linux 安裝**

```bash
# 安裝 Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 添加到 PATH (添加到 ~/.bashrc 或 ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# 驗證安裝
poetry --version
```

### 4.2 專案依賴安裝

```bash
# 進入專案目錄
cd iSpan_python-NLP-cookbooks_v2

# 安裝所有依賴 (首次執行需要 5-10 分鐘)
poetry install

# 如果遇到錯誤,先更新 Poetry
poetry self update

# 查看已安裝套件
poetry show

# 進入虛擬環境
poetry shell
```

### 4.3 Jupyter Notebook 啟動

```bash
# 方法 1: 使用 Poetry 運行 (推薦)
poetry run jupyter notebook

# 方法 2: 進入虛擬環境後運行
poetry shell
jupyter notebook

# 方法 3: 使用 VS Code Jupyter 擴展
# 1. 安裝 "Jupyter" 擴展
# 2. 選擇 Poetry 虛擬環境作為 kernel
```

### 4.4 常用套件說明

| 套件 | 版本 | 用途 | 必需 |
|------|------|------|------|
| `transformers` | 4.35+ | Hugging Face 模型 | ✅ 必需 |
| `datasets` | 2.14+ | 數據集加載 | ✅ 必需 |
| `torch` | 2.1+ | PyTorch 深度學習 | ✅ 必需 |
| `tensorflow` | 2.14+ | TensorFlow 深度學習 | ⚠️ 可選 |
| `jieba` | 0.42+ | 中文分詞 | ✅ 必需 |
| `nltk` | 3.8+ | 英文 NLP 工具 | ✅ 必需 |
| `spacy` | 3.7+ | 進階 NLP | ⚠️ 可選 |
| `gensim` | 4.3+ | 詞向量與主題建模 | ✅ 必需 |
| `scikit-learn` | 1.3+ | 機器學習 | ✅ 必需 |

---

## 5. 常見問題排解

### 5.1 環境問題

#### **Q1: Poetry 安裝後找不到指令**

```bash
# Windows: 檢查 PATH
echo $env:Path
# 應包含: C:\Users\<YourName>\AppData\Roaming\Python\Scripts

# macOS/Linux: 檢查 PATH
echo $PATH
# 應包含: /Users/<YourName>/.local/bin

# 手動添加到 PATH (永久)
# Windows: 系統變數 → Path → 新增
# macOS/Linux: 編輯 ~/.bashrc 或 ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

#### **Q2: `poetry install` 失敗**

```bash
# 解決方案 1: 清除快取
poetry cache clear pypi --all

# 解決方案 2: 更新 Poetry
poetry self update

# 解決方案 3: 使用國內鏡像 (中國大陸用戶)
poetry config repositories.pypi https://pypi.tuna.tsinghua.edu.cn/simple

# 解決方案 4: 手動安裝核心套件
poetry add transformers datasets torch jieba nltk
```

#### **Q3: Jupyter Kernel 無法連接**

```bash
# 方案 1: 重新安裝 ipykernel
poetry run pip install --force-reinstall ipykernel

# 方案 2: 手動註冊 kernel
poetry run python -m ipykernel install --user --name=ispan-nlp

# 方案 3: 重啟 Jupyter
poetry run jupyter notebook --no-browser
```

### 5.2 執行問題

#### **Q4: 找不到共享資源 (shared_resources)**

```python
# 問題: FileNotFoundError: shared_resources/stopwords/...

# 解決方案 1: 使用絕對路徑
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
STOPWORDS_PATH = PROJECT_ROOT / "shared_resources" / "stopwords" / "stopwords_zh_tw.txt"

# 解決方案 2: 確認當前目錄
print(os.getcwd())  # 應該在專案根目錄

# 解決方案 3: 使用相對路徑 (從 notebook 位置)
# 課程資料 notebooks: ../../shared_resources/
# 補充教材 notebooks: ../../shared_resources/
```

#### **Q5: NLTK 數據下載失敗**

```python
# 解決方案: 手動下載 NLTK 數據
import nltk

# 方法 1: 互動式下載器
nltk.download()  # 會開啟 GUI

# 方法 2: 指定數據包
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# 方法 3: 下載所有 (需要時間)
nltk.download('all')
```

#### **Q6: Hugging Face 模型下載慢/失敗**

```python
# 解決方案 1: 使用鏡像站 (中國大陸用戶)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 解決方案 2: 手動下載後本地加載
# 1. 從 https://hf-mirror.com 下載模型
# 2. 解壓到本地目錄
# 3. 使用本地路徑加載
from transformers import AutoModel
model = AutoModel.from_pretrained("./local_models/bert-base-uncased")

# 解決方案 3: 設置代理 (如果有)
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
```

### 5.3 數據問題

#### **Q7: 找不到數據集檔案**

```python
# 檢查數據集路徑
import os
from pathlib import Path

# 列出可用數據集
datasets_path = Path("datasets")
for dataset_dir in datasets_path.iterdir():
    if dataset_dir.is_dir():
        print(f"📁 {dataset_dir.name}")
        for file in dataset_dir.glob("*"):
            print(f"  - {file.name}")
```

#### **Q8: 數據集版本不匹配**

```python
# 問題: 某些 notebooks 使用的數據格式可能不同

# 解決方案: 查看 notebook 開頭的數據說明
# 通常會註明數據來源與格式

# 如果數據缺失,可以:
# 1. 查看 docs/16_wbs_development_plan_template.md 的數據資源規劃
# 2. 使用類似的公開數據集替代
# 3. 使用 Hugging Face datasets 函式庫
from datasets import load_dataset
dataset = load_dataset("imdb")  # 替代本地數據
```

---

## 6. 進階使用技巧

### 6.1 高效 Notebook 使用

#### **快捷鍵**

| 快捷鍵 | 功能 | 模式 |
|--------|------|------|
| `Shift + Enter` | 執行當前 Cell 並移到下一個 | 任何 |
| `Ctrl + Enter` | 執行當前 Cell | 任何 |
| `A` | 在上方插入 Cell | Command |
| `B` | 在下方插入 Cell | Command |
| `DD` | 刪除 Cell | Command |
| `M` | 轉換為 Markdown | Command |
| `Y` | 轉換為 Code | Command |
| `L` | 顯示/隱藏行號 | Command |

#### **Jupyter Magic Commands**

```python
# 顯示執行時間
%time result = some_function()
%timeit some_function()  # 多次執行取平均

# 顯示變數內容
%whos  # 列出所有變數
%who   # 簡單列表

# 執行外部 Python 檔案
%run script.py

# 載入外部程式碼
%load script.py

# 查看函數文檔
?function_name
??function_name  # 查看原始碼

# 自動重新載入模組 (開發時有用)
%load_ext autoreload
%autoreload 2
```

### 6.2 Git 版本控制

```bash
# 初始化 Git (如果尚未初始化)
git init

# 添加 .gitignore (忽略不需要版本控制的檔案)
cat > .gitignore << EOF
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Poetry
.venv/
poetry.lock

# Data
datasets/**/*.csv
datasets/**/*.zip
*.pkl

# Models
models/
*.pt
*.pth
*.h5

# OS
.DS_Store
Thumbs.db
EOF

# 提交變更
git add .
git commit -m "feat: complete CH08 Hugging Face notebooks"

# 推送到遠端 (如果有)
git remote add origin <your-repo-url>
git push -u origin main
```

### 6.3 建立個人分支學習

```bash
# 建立個人學習分支
git checkout -b learning/your-name

# 在個人分支上自由實驗
# 修改 notebooks、添加註解、完成練習

# 提交進度
git add .
git commit -m "docs: complete CH03 exercises"

# 定期合併主分支的更新
git checkout main
git pull origin main
git checkout learning/your-name
git merge main
```

### 6.4 匯出與分享

#### **匯出 Notebook 為 HTML/PDF**

```bash
# 安裝 nbconvert
poetry add nbconvert

# 匯出為 HTML
poetry run jupyter nbconvert --to html notebook.ipynb

# 匯出為 PDF (需要 LaTeX)
poetry run jupyter nbconvert --to pdf notebook.ipynb

# 批量匯出
for file in 課程資料/08_*/*.ipynb; do
    poetry run jupyter nbconvert --to html "$file"
done
```

#### **建立作品集**

```markdown
# 建立 GitHub 作品集

1. 選擇 3-5 個最佳專案
2. 為每個專案建立 README.md
3. 添加執行結果截圖
4. 說明技術棧與學習收穫
5. 發布到 GitHub Pages
```

---

## 7. 教師指南

### 7.1 課堂準備清單

#### **課前 1 週**
- [ ] 確認所有學生環境設置完成
- [ ] 準備課程 Slides (可從講義轉換)
- [ ] 測試當週 notebooks 執行無誤
- [ ] 準備額外練習題

#### **課前 1 天**
- [ ] 檢查投影設備
- [ ] 準備 USB 隨身碟 (離線安裝包)
- [ ] 列印重點講義
- [ ] 準備 Q&A 時段

#### **課後**
- [ ] 收集學生反饋
- [ ] 整理常見問題
- [ ] 更新課程內容
- [ ] 批改作業/專案

### 7.2 教學建議

#### **理論與實作比例**
- **入門課程 (CH01-03)**: 30% 理論 + 70% 實作
- **核心課程 (CH04-06)**: 40% 理論 + 60% 實作
- **前沿技術 (CH07-08)**: 50% 理論 + 50% 實作
- **專案實戰**: 10% 理論 + 90% 實作

#### **互動技巧**
1. **每 20 分鐘一次互動**
   - 提問
   - 小測驗
   - 即時練習

2. **分組討論**
   - 3-4 人一組
   - 討論技術問題
   - 分享解決方案

3. **即時編碼 (Live Coding)**
   - 邊講邊寫
   - 故意犯錯並修正
   - 展示 Debug 流程

### 7.3 評量方式

#### **平時成績 (60%)**
- 出席率: 10%
- 課堂練習: 20%
- 作業: 30%

#### **期末評量 (40%)**
- 個人專案: 30%
- 書面報告: 10%

#### **專案評分標準**
| 項目 | 權重 | 說明 |
|------|------|------|
| 技術難度 | 30% | 使用的技術深度與廣度 |
| 代碼品質 | 25% | 可讀性、註解、結構 |
| 創新性 | 20% | 問題解決的創意 |
| 完整性 | 15% | 功能完整度 |
| 文檔 | 10% | README、使用說明 |

---

## 8. 貢獻指南

### 8.1 如何貢獻

歡迎提交:
- 🐛 Bug 修復
- ✨ 新功能/新範例
- 📝 文檔改進
- 🌐 翻譯

### 8.2 貢獻流程

```bash
# 1. Fork 專案
# 2. 建立功能分支
git checkout -b feature/new-example

# 3. 進行修改
# 4. 提交變更
git add .
git commit -m "feat: add sentiment analysis example"

# 5. 推送到 Fork
git push origin feature/new-example

# 6. 建立 Pull Request
```

### 8.3 代碼規範

#### **Notebook 命名規範**
```
{編號}_{技術名稱}_{應用場景}.ipynb

✅ 01_Jieba中文斷詞_詞性標註.ipynb
✅ 02_LSTM情感分析_IMDB電影評論.ipynb
❌ test.ipynb
❌ untitled1.ipynb
```

#### **代碼風格**
- Python: 遵循 PEP 8
- 變數命名: `snake_case`
- 類別命名: `PascalCase`
- 常數: `UPPER_CASE`

#### **註解規範**
```python
# ✅ 好的註解: 解釋為什麼
# 使用 Laplace 平滑處理零概率問題
smoothed_prob = (count + 1) / (total + vocab_size)

# ❌ 壞的註解: 重複代碼
# 將 count 加 1 然後除以 total 加 vocab_size
smoothed_prob = (count + 1) / (total + vocab_size)
```

---

## 9. 附錄

### 9.1 推薦學習資源

#### **書籍**
- 《Speech and Language Processing》 - Dan Jurafsky
- 《Natural Language Processing with Python》 - Steven Bird
- 《深度學習》 - Ian Goodfellow

#### **線上課程**
- Coursera: Natural Language Processing Specialization
- fast.ai: Practical Deep Learning for Coders
- Hugging Face Course: https://huggingface.co/course

#### **社群**
- Hugging Face 論壇
- Stack Overflow
- Reddit: r/LanguageTechnology

### 9.2 工具推薦

| 工具 | 用途 | 推薦度 |
|------|------|--------|
| VS Code | 編輯器 | ⭐⭐⭐⭐⭐ |
| PyCharm | Python IDE | ⭐⭐⭐⭐ |
| Colab | 雲端 GPU | ⭐⭐⭐⭐⭐ |
| Weights & Biases | 實驗追蹤 | ⭐⭐⭐⭐ |
| GitHub Copilot | AI 輔助編碼 | ⭐⭐⭐⭐ |

### 9.3 數據集資源

| 平台 | 說明 | 網址 |
|------|------|------|
| Hugging Face Datasets | 50,000+ 數據集 | https://huggingface.co/datasets |
| Kaggle | 數據競賽平台 | https://www.kaggle.com/datasets |
| Papers with Code | 論文+代碼+數據 | https://paperswithcode.com/datasets |
| UCI ML Repository | 經典數據集 | https://archive.ics.uci.edu/ml |

---

## 📞 支援與聯繫

- **技術問題**: 開 GitHub Issue
- **課程諮詢**: ispan-nlp@example.com
- **Bug 回報**: 使用 Issue Template

---

**文件版本**: v1.0
**維護者**: iSpan NLP Team
**授權**: CC BY-NC-SA 4.0
**最後更新**: 2025-10-17
