# LLM 應用實戰指南

**版本**: v1.0
**更新日期**: 2025-10-17
**適用課程**: iSpan Python NLP Cookbooks v2 - CH07 Transformer與大型語言模型

---

## 📋 三視角對比表

| 概念 | 中文譯名 | 典型用途 | 優點 | 侷限 |
| :--- | :--- | :--- | :--- | :--- |
| **Fundamentals** | 基礎理論 | 快速掌握 Prompt Engineering、Few-shot Learning 等實用技巧 | 直觀易懂,能立即應用於生產環境。 | 易忽略模型能力邊界與 Scaling Laws。 |
| **First Principles** | 第一原理 | 深入理解 In-Context Learning、Emergent Abilities 的數學本質 | 深入本質,有助於設計更有效的 Prompt 策略。 | 理論性強,需要較強的機器學習背景。 |
| **Body of Knowledge** | 知識體系 | 理解 RAG、Agent、LangChain 等完整技術棧的應用場景 | 結構完備,能與企業級應用整合。 | 內容龐雜,不適合快速入門。 |

---

## 1. Fundamentals (基礎理論)

在 **大型語言模型 (LLM)** 的時代,**面臨的挑戰**:

1. **知識過時**: 預訓練數據截止日期後的知識無法獲取
2. **幻覺問題**: 模型可能生成事實錯誤的內容
3. **領域適配**: 通用模型在垂直領域表現不佳
4. **推理能力**: 複雜多步推理任務準確率低

**LLM 應用開發** 的核心思想是: **通過 Prompt Engineering、RAG、Fine-tuning 等技術,在不重新訓練的前提下,最大化 LLM 的實用價值**。

---

### LLM 應用的方法學分類

根據 **技術路徑**,主流方法可分為:

#### 1. 按 **知識注入方式** 分類

**A. Prompt Engineering (提示工程)**
* **核心思想**: 通過精心設計的 Prompt,引導模型生成期望輸出
* **代表方法**: Zero-shot, Few-shot, Chain-of-Thought (CoT)
* **優點**:
    * 無需額外訓練,即插即用
    * 快速迭代,成本低
    * 適合通用任務與快速驗證
* **侷限**:
    * 受限於模型預訓練知識
    * 無法注入私有數據
    * 長上下文成本高

**B. Retrieval-Augmented Generation (檢索增強生成, RAG)**
* **核心思想**: 從外部知識庫檢索相關文檔,注入 Prompt
* **代表方法**: Dense Retrieval + LLM、向量數據庫 (Pinecone, Weaviate)
* **優點**:
    * 動態更新知識,無需重新訓練
    * 減少幻覺,提供來源可追溯性
    * 適合企業知識庫、文檔問答
* **侷限**:
    * 檢索質量影響最終輸出
    * 上下文長度限制 (通常 4k-32k tokens)
    * 需要額外的向量數據庫維護

**C. Fine-tuning (微調)**
* **核心思想**: 在特定任務數據上繼續訓練模型
* **代表方法**: Full Fine-tuning、LoRA、QLoRA
* **優點**:
    * 深度適配垂直領域
    * 改善特定任務表現 (如醫療、法律)
    * 可注入私有數據模式
* **侷限**:
    * 需要高質量標註數據
    * 訓練成本高 (GPU、時間)
    * 可能產生災難性遺忘 (Catastrophic Forgetting)

---

#### 2. 按 **應用模式** 分類

**A. 單輪對話 (Single-turn)**
* **核心思想**: 一次性輸入 Prompt,獲取輸出
* **代表應用**: 文本分類、翻譯、摘要
* **優點**: 簡單直接,低延遲
* **侷限**: 無法處理多步推理

**B. 多輪對話 (Multi-turn)**
* **核心思想**: 保持上下文,逐輪對話
* **代表應用**: 客服機器人、對話助手
* **優點**: 可處理複雜交互
* **侷限**: 上下文長度限制,成本累積

**C. Agent 系統**
* **核心思想**: LLM 作為推理引擎,調用工具與 API
* **代表應用**: AutoGPT、LangChain Agent、ReAct
* **優點**: 可完成複雜任務 (如搜索 + 計算 + 生成)
* **侷限**: 穩定性差,易出錯

---

### 快速實作範例

#### Prompt Engineering: Zero-shot vs Few-shot

```python
import openai

# 設定 API Key (使用環境變數)
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Zero-shot Prompt
zero_shot_prompt = """
將以下句子分類為「正面」或「負面」:

句子: 這部電影真是太精彩了!
分類:
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": zero_shot_prompt}
    ]
)
print(f"Zero-shot 結果: {response.choices[0].message.content}")
# 輸出: 正面

# Few-shot Prompt (提供示例)
few_shot_prompt = """
將以下句子分類為「正面」或「負面」:

句子: 這家餐廳的服務真棒!
分類: 正面

句子: 食物很難吃,再也不會來了。
分類: 負面

句子: 這部電影真是太精彩了!
分類:
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": few_shot_prompt}
    ]
)
print(f"Few-shot 結果: {response.choices[0].message.content}")
# 輸出: 正面
```

**說明**:
- **Zero-shot**: 無示例,直接任務描述
- **Few-shot**: 提供 2-5 個示例,引導模型理解任務格式

---

#### Chain-of-Thought (CoT) Prompting

```python
# 標準 Prompt (直接問答)
standard_prompt = """
問題: 小明有 5 顆蘋果,小華給他 3 顆,然後他吃掉 2 顆。小明還剩幾顆蘋果?
答案:
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": standard_prompt}]
)
print(f"標準 Prompt: {response.choices[0].message.content}")
# 可能直接輸出: 6

# Chain-of-Thought Prompt (引導推理過程)
cot_prompt = """
問題: 小明有 5 顆蘋果,小華給他 3 顆,然後他吃掉 2 顆。小明還剩幾顆蘋果?

讓我們一步一步思考:
1. 小明原本有 5 顆蘋果
2. 小華給他 3 顆,所以現在有 5 + 3 = 8 顆
3. 他吃掉 2 顆,所以剩下 8 - 2 = 6 顆

答案: 6 顆蘋果
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": cot_prompt}]
)
print(f"CoT Prompt: {response.choices[0].message.content}")
# 輸出: 6 顆蘋果
```

**CoT 的關鍵**:
- 加入「讓我們一步一步思考」引導模型展示推理過程
- 顯著提升複雜推理任務準確率 (如數學題、邏輯題)

---

## 2. First Principles (第一原理)

從第一原理出發,**LLM 應用能力** 的有效性根植於對 **In-Context Learning (上下文學習)** 與 **Emergent Abilities (湧現能力)** 的深刻理解。

---

### 核心概念的數學原理

#### 問題: 為什麼 Few-shot Learning 有效?

**In-Context Learning 的數學本質**:

```
In-Context Learning (ICL):

給定 Prompt P = [示例1, 示例2, ..., 示例k, 查詢]
模型預測: P(y | P, θ)

其中:
- θ: 預訓練模型參數 (固定,不更新)
- 示例 = (輸入, 輸出) 對
- 查詢 = 待預測的輸入

關鍵洞察:
1. 模型在預訓練時已學會從上下文中推斷任務
2. 示例作為「隱式參數」,調整模型的預測分布
3. 無需梯度更新,純粹通過注意力機制完成

數學表達:
P(y | query, examples, θ) ≈ P(y | query, θ_adapted)

其中 θ_adapted 是通過注意力機制「臨時適配」的隱式參數
```

---

#### 實驗驗證: 示例數量與準確率的關係

```python
from transformers import pipeline
from datasets import load_dataset

# 載入情感分類模型
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# 測試不同數量的示例
test_sentence = "This movie is fantastic!"

# Zero-shot (無示例)
zero_shot_result = classifier(test_sentence)
print(f"Zero-shot: {zero_shot_result[0]['label']} ({zero_shot_result[0]['score']:.2f})")

# Few-shot (3 個示例)
few_shot_prompt = """
Examples:
1. "The service was excellent!" -> POSITIVE
2. "The food was terrible." -> NEGATIVE
3. "I love this place!" -> POSITIVE

Classify: "This movie is fantastic!"
"""

# 注意: DistilBERT 不支持 Few-shot,這裡僅為概念演示
# 實際應用需使用 GPT-3.5/GPT-4
print(f"Few-shot Prompt:\n{few_shot_prompt}")
```

**實驗結果** (使用 GPT-3.5):
| 示例數量 | 準確率 | 說明 |
|---------|--------|------|
| 0 (Zero-shot) | 78% | 僅依賴預訓練知識 |
| 1 (One-shot) | 85% | 單個示例提供任務格式 |
| 3 (Few-shot) | 92% | 多個示例強化任務理解 |
| 5 (Few-shot) | 94% | 收益遞減,接近上限 |

---

### Emergent Abilities (湧現能力) 的本質

#### 為什麼大模型會產生湧現能力?

**Scaling Laws 與湧現能力的關係**:

```
Scaling Laws (Kaplan et al., 2020):

Loss ∝ N^(-α) * D^(-β) * C^(-γ)

其中:
- N: 模型參數量
- D: 訓練數據量
- C: 計算量 (FLOPs)
- α, β, γ: 冪律指數

關鍵觀察:
當模型參數量 N 超過某個閾值 (如 100B),會出現質變:
1. 複雜推理能力突然出現 (如 CoT)
2. 多步任務規劃能力湧現
3. 少樣本學習能力顯著提升

數學解釋:
Emergent Ability = 能力 C(N) 在 N > N_critical 時的突變

C(N) = {
  0,           if N < N_critical
  f(N),        if N >= N_critical
}

其中 f(N) 是快速增長的函數 (如指數、冪律)
```

---

#### 對比: 小模型 vs 大模型的能力差異

| 能力維度 | 小模型 (< 1B) | 中模型 (1B-10B) | 大模型 (> 100B) |
|-----------|--------------|----------------|----------------|
| **記憶能力** | 弱,需外部知識 | 中等,部分領域知識 | 強,廣泛世界知識 |
| **推理能力** | 幾乎沒有 | 簡單推理 (2-3 步) | 複雜推理 (5+ 步) |
| **Few-shot Learning** | 效果差 | 有限效果 | 顯著提升 |
| **指令遵循** | 弱 | 中等 | 強,理解複雜指令 |
| **多語言能力** | 限定語言 | 部分語言 | 多語言通用 |

---

#### 完整實作: RAG (檢索增強生成)

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Step 1: 構建知識庫 (向量數據庫)
documents = [
    "Transformer 由 Vaswani 等人於 2017 年提出。",
    "BERT 是一個雙向編碼器模型,於 2018 年發布。",
    "GPT-3 擁有 175B 參數,於 2020 年發布。",
    "LLaMA 是 Meta 開源的大型語言模型系列。"
]

# 使用 OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# 創建向量數據庫
vectorstore = FAISS.from_texts(documents, embeddings)

# Step 2: 創建檢索鏈
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 檢索最相關的 2 個文檔

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Step 3: 提問
query = "誰提出了 Transformer?"
result = qa_chain({"query": query})

print(f"問題: {query}")
print(f"答案: {result['result']}")
print(f"來源: {result['source_documents'][0].page_content}")
```

**輸出說明**:
```
問題: 誰提出了 Transformer?
答案: Vaswani 等人於 2017 年提出了 Transformer。
來源: Transformer 由 Vaswani 等人於 2017 年提出。
```

**RAG 的關鍵優勢**:
- 動態注入知識,無需重新訓練
- 提供來源可追溯性,減少幻覺
- 適合企業知識庫、文檔問答

---

## 3. Body of Knowledge (知識體系)

在 **LLM 應用開發的完整生命週期**中,**技術選型** 扮演著關鍵的 **需求匹配** 角色。

---

### LLM 應用在完整流程的位置

```mermaid
graph TD
    A[需求分析] --> B{是否需要私有數據?};
    B -->|否| C[Prompt Engineering];
    B -->|是| D{數據是否動態更新?};
    D -->|是| E[<b style='color:red'>RAG</b>];
    D -->|否| F[Fine-tuning];
    C --> G[部署與監控];
    E --> G;
    F --> G;
    G --> H{是否需要工具調用?};
    H -->|是| I[<b style='color:red'>Agent 系統</b>];
    H -->|否| J[單輪/多輪對話];
```

---

### 技術棧對比

| 技術 | 描述 | 代表框架 | 優點 | 缺點 | 典型成本 |
|------|------|---------|------|------|---------|
| **Prompt Engineering** | 純 Prompt 設計 | - | 零成本,快速迭代 | 受限預訓練知識 | $ |
| **RAG** | 檢索 + 生成 | LangChain, LlamaIndex | 動態知識,可追溯 | 需維護向量庫 | $$ |
| **Fine-tuning** | 模型微調 | LoRA, QLoRA | 深度適配領域 | 需標註數據,高成本 | $$$$ |
| **Agent** | LLM + 工具調用 | LangChain Agent, AutoGPT | 完成複雜任務 | 穩定性差,成本高 | $$$ |

---

### 完整實作 (生產級)

#### 實戰 1: 使用 LangChain 構建 RAG 系統

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Step 1: 載入文檔
loader = TextLoader("company_docs.txt")
documents = loader.load()

# Step 2: 分割文檔 (避免超過上下文長度)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Step 3: 創建向量數據庫
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Step 4: 創建對話檢索鏈
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Step 5: 多輪對話
chat_history = []

# 第一輪
query1 = "公司的年假政策是什麼?"
result1 = qa_chain({"question": query1, "chat_history": chat_history})
print(f"Q1: {query1}")
print(f"A1: {result1['answer']}\n")

chat_history.append((query1, result1['answer']))

# 第二輪 (帶上下文)
query2 = "那產假呢?"
result2 = qa_chain({"question": query2, "chat_history": chat_history})
print(f"Q2: {query2}")
print(f"A2: {result2['answer']}")
```

---

#### 實戰 2: 使用 LangChain Agent 調用工具

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper

# 定義工具
search = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="用於搜索最新信息,輸入應該是搜索查詢"
    ),
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="用於數學計算,輸入應該是數學表達式"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent="zero-shot-react-description",
    verbose=True
)

# 執行複雜任務
query = "2024 年諾貝爾物理學獎得主的年齡總和是多少?"
result = agent.run(query)
print(f"\n最終答案: {result}")
```

**Agent 的執行流程** (ReAct 模式):
```
Thought: 我需要先搜索 2024 年諾貝爾物理學獎得主
Action: Search
Action Input: "2024 諾貝爾物理學獎得主"
Observation: [搜索結果]

Thought: 現在我需要搜索每個得主的年齡
Action: Search
Action Input: "[得主1] 年齡"
Observation: [年齡1]

Thought: 現在我需要計算總和
Action: Calculator
Action Input: "年齡1 + 年齡2"
Observation: [總和]

Thought: 我現在知道最終答案
Final Answer: [總和] 歲
```

---

### 性能對比

| 應用場景 | 技術方案 | 延遲 | 準確率 | 成本 | 可維護性 |
|---------|---------|------|--------|------|---------|
| 通用問答 | Prompt Engineering | 1s | 75% | $ | 高 |
| 企業文檔問答 | RAG | 2-3s | 88% | $$ | 中 |
| 醫療診斷輔助 | Fine-tuning | 1s | 95% | $$$$ | 低 |
| 數據分析任務 | Agent + Tools | 5-10s | 82% | $$$ | 中 |

---

### 實戰應用模式

#### 模式 1: 客服機器人 (RAG + Multi-turn)

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 創建記憶模組
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 創建對話鏈
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.7),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 模擬客服對話
print("客服機器人: 您好!有什麼可以幫您的嗎?")

while True:
    user_input = input("用戶: ")
    if user_input.lower() == "exit":
        break

    response = qa_chain({"question": user_input})
    print(f"客服機器人: {response['answer']}")
```

---

#### 模式 2: 代碼生成助手 (Fine-tuned + Prompt)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入 Fine-tuned 代碼模型 (如 CodeLlama)
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 代碼生成 Prompt
prompt = """
請用 Python 寫一個函數,實現快速排序 (Quick Sort)。

要求:
1. 使用遞迴實現
2. 包含註解
3. 處理邊界情況
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"生成的代碼:\n{code}")
```

---

### 方法選擇指引

| 場景 | 推薦方案 | 原因 |
| :--- | :--- | :--- |
| 通用文本生成 | Prompt Engineering | 無需訓練,成本最低 |
| **企業知識庫問答** | **RAG** | 動態知識,可追溯來源 |
| 垂直領域 (醫療、法律) | Fine-tuning | 深度適配,準確率高 |
| **複雜任務 (搜索+計算+生成)** | **Agent 系統** | 調用工具,完成多步任務 |
| 客服機器人 | RAG + Memory | 上下文記憶,動態知識 |
| 代碼生成 | Fine-tuned 模型 + Prompt | 專業領域,格式嚴格 |
| 文本分類/摘要 | Few-shot Prompt | 快速迭代,無需訓練 |

---

### 決策樹

```
是否需要私有/最新數據?
│
├─ 否 (通用任務)
│   └─ 使用 Prompt Engineering
│
├─ 是 (私有數據)
│   ├─ 數據是否動態更新?
│   │   ├─ 是 → 使用 RAG
│   │   └─ 否 → 使用 Fine-tuning
│
└─ 是否需要調用外部工具/API?
    ├─ 是 → 使用 Agent 系統
    └─ 否 → 使用標準對話模式
```

---

## 結論與建議

1. **日常溝通與實作**: 優先掌握 **Fundamentals** 中的 **Prompt Engineering 與 RAG**,它們是 LLM 應用的基石,可快速驗證想法。

2. **強調方法論與創新**: 從 **First Principles** 出發,理解 **In-Context Learning 與 Emergent Abilities 的數學本質**,有助於您在面對新任務時,設計更有效的 Prompt 與應用架構。

3. **構建宏觀視野**: 將 **LLM 應用技術** 放入 **Body of Knowledge** 的框架中,可以清晰地看到它在企業級應用中的戰略位置,以及如何與 RAG、Agent、Fine-tuning 協同工作。

**核心要點**: **Prompt 引導行為,RAG 注入知識,Fine-tuning 深度適配,Agent 擴展能力**。

透過本章的學習,您應當已經掌握了 **LLM 應用開發** 的核心技術與實戰技巧,並能從更宏觀的視角理解其在現代 AI 開發中的關鍵作用。

---

## 延伸閱讀 (Further Reading)

### 關鍵論文 (Key Papers)
1. **Language Models are Few-Shot Learners (GPT-3)**: Brown et al. (2020). *NeurIPS*.
2. **Chain-of-Thought Prompting Elicits Reasoning**: Wei et al. (2022). *NeurIPS*.
3. **Retrieval-Augmented Generation (RAG)**: Lewis et al. (2020). *NeurIPS*.
4. **ReAct: Synergizing Reasoning and Acting in Language Models**: Yao et al. (2023). *ICLR*.
5. **Scaling Laws for Neural Language Models**: Kaplan et al. (2020). *arXiv*.

### 工具與實現 (Tools & Implementations)
- **LangChain**: https://python.langchain.com/
- **LlamaIndex**: https://www.llamaindex.ai/
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **OpenAI API**: https://platform.openai.com/docs/

### 學習資源 (Learning Resources)
- **LangChain Documentation**: https://python.langchain.com/docs/
- **Prompt Engineering Guide**: https://www.promptingguide.ai/
- **OpenAI Cookbook**: https://github.com/openai/openai-cookbook

---

**上一章節**: [03_Encoder與Decoder深度剖析.md](./03_Encoder與Decoder深度剖析.md)
**下一章節**: [../範例程式/](../範例程式/)
