# 🚀 iSpan Python NLP Cookbooks - 快速啟動指南

---

**歡迎來到 iSpan Python NLP 速成教案!**

這份指南將幫助你在 **5 分鐘內**開始學習 NLP。

---

## 📋 前置檢查清單

在開始之前,請確認你已具備:

- [ ] Python 3.8+ 已安裝
- [ ] 基礎 Python 語法知識
- [ ] 文字編輯器或 IDE (推薦 VS Code, PyCharm, Jupyter)
- [ ] 網路連線 (用於下載套件)

---

## ⚡ 5 分鐘快速開始

### Step 1: 驗證環境 (1 分鐘)

```bash
# 檢查 Python 版本
python --version
# 或
python3 --version

# 應該顯示 3.8 或更高版本
```

### Step 2: 安裝核心套件 (2 分鐘)

```bash
# 基礎套件
pip install jupyter notebook

# NLP 核心套件 (擇一即可快速開始)
pip install numpy pandas matplotlib

# 或使用 requirements.txt (推薦)
pip install -r requirements.txt
```

### Step 3: 啟動 Jupyter Notebook (1 分鐘)

```bash
# 進入專案目錄
cd iSpan_python-NLP-cookbooks_v2

# 啟動 Jupyter
jupyter notebook
```

瀏覽器會自動開啟 `http://localhost:8888`

### Step 4: 開始第一個 Notebook (1 分鐘)

導航到: `課程資料/01_環境安裝與設定/01_環境檢查與套件安裝.ipynb`

點擊並執行第一個 Cell! 🎉

---

## 📚 推薦學習路徑

### 🎯 路徑 1: 快速入門 (適合趕時間的你)

**總時間**: 2-3 小時

```
1. CH01: 環境安裝 (30min)
   └─ 01_環境檢查與套件安裝.ipynb

2. CH02: NLP 入門 (1h)
   └─ 01_自然語言處理基礎概念.ipynb

3. CH08: Hugging Face 快速上手 (1.5h)
   ├─ 01_Hugging_Face生態簡介.ipynb
   └─ 02_Pipeline_API快速入門.ipynb
```

### 🏃 路徑 2: 系統學習 (適合想全面掌握的你)

**總時間**: 40-50 小時

```
Week 1: 基礎建立
├─ CH01: 環境安裝與設定
└─ CH02: NLP 入門概念

Week 2-3: 核心技術
├─ CH03: 文本預處理
├─ CH04: 機器學習與 NLP
├─ CH05: 神經網路基礎
└─ CH06: RNN/LSTM

Week 4-5: 前沿技術 ⭐
├─ CH07: Transformer 與 LLMs
└─ CH08: Hugging Face 實戰

Week 6: 總結與實戰
└─ CH09: 課程總結與未來展望
```

### 💡 路徑 3: 實戰導向 (適合想快速應用的你)

**總時間**: 10-15 小時

```
核心實戰章節:
├─ CH08-03: 情感分析實戰
├─ CH08-08: 模型微調
├─ CH08-09: 客戶意見分析儀 (完整專案)
└─ CH08-10: 生產環境部署
```

---

## 🛠️ 常見安裝問題解決

### 問題 1: Python 版本過舊

```bash
# macOS/Linux
brew install python@3.10

# Windows
# 下載並安裝: https://www.python.org/downloads/
```

### 問題 2: pip 安裝失敗

```bash
# 升級 pip
python -m pip install --upgrade pip

# 使用國內鏡像 (中國用戶)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
```

### 問題 3: Jupyter 無法啟動

```bash
# 重新安裝 Jupyter
pip uninstall jupyter notebook
pip install jupyter notebook

# 或使用 JupyterLab (推薦)
pip install jupyterlab
jupyter lab
```

### 問題 4: GPU 相關錯誤

```bash
# 如果沒有 GPU,使用 CPU 版本即可
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 有 NVIDIA GPU,安裝 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 核心套件列表

### 必裝套件 (Level 1)

```bash
# 資料處理
numpy==1.24.3
pandas==2.0.3

# 可視化
matplotlib==3.7.2
seaborn==0.12.2

# Jupyter
jupyter==1.0.0
notebook==7.0.0
```

### NLP 基礎套件 (Level 2)

```bash
# 文本處理
nltk==3.8.1
spacy==3.6.0
jieba==0.42.1  # 中文分詞

# 機器學習
scikit-learn==1.3.0
```

### 深度學習套件 (Level 3)

```bash
# 深度學習框架
torch==2.0.1
tensorflow==2.13.0  # 可選

# Hugging Face
transformers==4.35.0
datasets==2.14.0
```

### 進階套件 (Level 4)

```bash
# 模型優化
optimum==1.14.0
onnx==1.14.1
onnxruntime==1.16.0

# API 部署
fastapi==0.104.1
uvicorn==0.24.0
```

---

## 🎓 章節內容快速導覽

### CH01: 環境安裝與設定 ✅
- **難度**: ⭐ 入門
- **時間**: 1-2 小時
- **重點**: 開發環境設置、套件安裝

### CH02: NLP 入門概念 ✅
- **難度**: ⭐ 入門
- **時間**: 2-3 小時
- **重點**: NLP 基本概念、應用場景

### CH03: 文本預處理 ✅
- **難度**: ⭐⭐ 基礎
- **時間**: 4-6 小時
- **重點**: 文本清理、分詞、向量化

### CH04: 機器學習與 NLP ✅
- **難度**: ⭐⭐ 基礎
- **時間**: 4-6 小時
- **重點**: 傳統 ML 算法應用於 NLP

### CH05: 神經網路基礎 ✅
- **難度**: ⭐⭐⭐ 中級
- **時間**: 4-6 小時
- **重點**: DNN、CNN 基礎

### CH06: RNN/LSTM ✅
- **難度**: ⭐⭐⭐ 中級
- **時間**: 6-8 小時
- **重點**: 序列模型、LSTM、GRU

### CH07: Transformer 與 LLMs ✅ ⭐ 重點
- **難度**: ⭐⭐⭐⭐ 進階
- **時間**: 8-12 小時
- **重點**: Transformer 架構、注意力機制、LLM

### CH08: Hugging Face 實戰 ✅ ⭐ 重點
- **難度**: ⭐⭐⭐ 中級
- **時間**: 10-15 小時
- **重點**: Pipeline API、模型微調、部署

### CH09: 課程總結 ✅
- **難度**: ⭐⭐ 基礎
- **時間**: 2-4 小時
- **重點**: 技術回顧、職涯規劃

---

## 💻 開發環境推薦

### 選項 1: VS Code (推薦)

**優點**: 輕量、擴展豐富、支援 Jupyter

**安裝**:
1. 下載 [VS Code](https://code.visualstudio.com/)
2. 安裝 Python 擴展
3. 安裝 Jupyter 擴展

### 選項 2: PyCharm

**優點**: 專業 Python IDE、強大的調試

**安裝**:
1. 下載 [PyCharm Community](https://www.jetbrains.com/pycharm/download/)
2. 配置 Python 解釋器

### 選項 3: Google Colab (雲端)

**優點**: 免費 GPU、無需本地安裝

**使用**:
1. 訪問 [Google Colab](https://colab.research.google.com/)
2. 上傳 `.ipynb` 檔案
3. 開始運行!

---

## 🔍 自我檢查清單

開始學習前,確認你能完成以下檢查:

### 基礎環境檢查

```bash
# 1. Python 版本
python --version
# 預期: Python 3.8.0 或更高

# 2. pip 可用
pip --version
# 預期: pip 20.0 或更高

# 3. Jupyter 可用
jupyter --version
# 預期: 顯示版本信息

# 4. 核心套件
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### 進階環境檢查 (可選)

```bash
# PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 📝 學習建議

### DO ✅

1. **按順序學習**: 前面章節是基礎
2. **動手實作**: 每個 notebook 都要運行
3. **做筆記**: 記錄重點與疑問
4. **改代碼**: 嘗試修改參數、數據
5. **提問題**: 不懂就問 (GitHub Issues / 社群)

### DON'T ❌

1. **跳章節**: 跳過基礎會很吃力
2. **只看不做**: 光看無法真正理解
3. **死記硬背**: 理解原理比記代碼重要
4. **追求完美**: 先完成,再完美
5. **孤軍奮戰**: 善用社群資源

---

## 🆘 獲取幫助

### 官方資源

- **課程倉庫**: [GitHub Repo]
- **問題回報**: [GitHub Issues]
- **課程文檔**: `docs/` 目錄

### 社群資源

- **Discord / Slack**: [連結]
- **課程論壇**: [連結]
- **學習社群**: [連結]

### 推薦學習資源

#### NLP 入門

- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)
- [Hugging Face Course](https://huggingface.co/course)
- [Fast.ai NLP](https://www.fast.ai/)

#### 深度學習基礎

- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

#### 實戰專案

- [Kaggle NLP Competitions](https://www.kaggle.com/competitions?search=nlp)
- [Papers With Code](https://paperswithcode.com/area/natural-language-processing)

---

## 🎯 學習里程碑

追蹤你的學習進度:

- [ ] **Week 1**: 完成 CH01-02 (環境與入門)
- [ ] **Week 2**: 完成 CH03-04 (預處理與 ML)
- [ ] **Week 3**: 完成 CH05-06 (DL 基礎與 RNN)
- [ ] **Week 4**: 完成 CH07 (Transformer)
- [ ] **Week 5**: 完成 CH08 (Hugging Face)
- [ ] **Week 6**: 完成 CH09 + 個人專案

---

## 🎉 準備好了嗎?

如果你已完成上述檢查,恭喜! 你已做好學習準備。

**現在就開始吧!**

```bash
# 1. 啟動 Jupyter
jupyter notebook

# 2. 開啟第一個 notebook
# 課程資料/01_環境安裝與設定/01_環境檢查與套件安裝.ipynb

# 3. 開始你的 NLP 學習之旅! 🚀
```

---

## 📞 需要幫助?

如果遇到任何問題:

1. **先查看**: `docs/FAQ.md` (常見問題)
2. **搜尋**: GitHub Issues (可能已有解答)
3. **提問**: 建立新的 Issue (詳細描述問題)

---

**祝學習順利! Let's build amazing NLP applications together!** 🤗

---

**文件版本**: v1.0
**最後更新**: 2025-10-17
**維護者**: iSpan NLP Team
