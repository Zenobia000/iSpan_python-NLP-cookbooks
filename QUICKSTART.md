# 🚀 快速啟動指南：5 分鐘設定您的 NLP 開發環境

歡迎來到 iSpan Python NLP Cookbooks！本指南將引導您快速完成環境設定，並開始運行您的第一個 NLP 程式。

---

## 📋 Step 0: 前置檢查

在開始之前，請確保您的電腦已安裝以下軟體：

-   **Python 3.8+**：[點此下載](https://www.python.org/downloads/)
-   **Git**：[點此下載](https://git-scm.com/downloads/)

> **如何檢查？**
> 打開您的終端機（Terminal 或命令提示字元），輸入以下指令：
> ```sh
> python --version
> git --version
> ```
> 如果能看到版本號，代表您已安裝成功！

---

## 🛠️ Step 1: 取得專案並設定環境

### 1. 下載專案程式碼

打開終端機，執行以下指令，將專案複製到您的電腦：

```bash
# 使用 git clone 下載專案
git clone https://github.com/iSpan/python-NLP-cookbooks.git

# 進入專案目錄
cd python-NLP-cookbooks
```

### 2. 安裝 Poetry

本專案使用 [Poetry](https://python-poetry.org/) 來管理 Python 套件與虛擬環境，它能確保您的開發環境乾淨且一致。

**macOS / Linux**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (使用 PowerShell)**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

安裝完成後，請重啟您的終端機。

### 3. 安裝專案依賴

在專案根目錄下，執行以下指令，Poetry 會自動為您建立虛擬環境並安裝所有必要的套件。

```bash
poetry install
```

> ☕ **溫馨提示**：首次安裝會下載所有課程用到的模型與數據集，可能需要 10-20 分鐘，具體時間取決於您的網路速度。這是正常現象，請耐心等候。

---

##  notebooks Step 2: 啟動您的 NLP 實驗室

### 1. 啟動 Jupyter Notebook

環境安裝完成後，執行以下指令來啟動 Jupyter Notebook：

```bash
poetry run jupyter notebook
```

執行後，您的預設瀏覽器將會自動開啟一個新分頁，網址為 `http://localhost:8888`。這就是您的 NLP 實驗室！

### 2. 運行您的第一個 NLP 程式

1.  在瀏覽器頁面中，點擊進入 `課程資料/` 資料夾。
2.  接著進入 `01_環境安裝與設定/`。
3.  點擊打開 `03_開發環境測試.ipynb`。
4.  在打開的 Notebook 中，點擊上方的 `▶ Run` 按鈕，或使用快捷鍵 `Shift + Enter` 來執行第一個程式碼區塊。

如果您看到「✅ 開發環境測試成功！」的輸出，恭喜您！您的 NLP 開發環境已準備就緒。

---

## ❓ 常見問題與解決方案

-   **問題：`poetry` 指令找不到？**
    -   **解決**：請確認您已將 Poetry 的路徑添加到系統的 `PATH` 環境變數中。通常在安裝結束時會有提示。您也可以參考 [Poetry 官方文件](https://python-poetry.org/docs/#installation) 的說明。

-   **問題：`poetry install` 下載很慢或失敗？**
    -   **解決**：可能是網路問題。您可以嘗試更換 Poetry 的套件來源，指向離您較近的鏡像站。執行以下指令更換來源：
        ```bash
        poetry config repositories.tsinghua https://pypi.tuna.tsinghua.edu.cn/simple
        ```

-   **問題：Jupyter Notebook 無法啟動？**
    -   **解決**：請確認您是在專案的根目錄下執行 `poetry run jupyter notebook`。如果仍然失敗，可以嘗試手動安裝：
        ```bash
        poetry run pip install jupyter notebook
        ```

---

## 🎉 恭喜！

您已經成功搭建了專業的 NLP 開發環境。現在，您可以開始您的學習之旅了！

**下一步建議**：

*   回到 [**README.md**](README.md) 查看我們為您規劃的學習路徑。
*   從 `課程資料/02_自然語言處理入門/` 開始，正式進入 NLP 的世界。

**祝您學習愉快！**
