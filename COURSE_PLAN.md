# 2025 Python NLP 速成教案：直通生成式 AI 開發實戰

本課程專為具備 Python 基礎的工程師設計，旨在快速建立必要的自然語言處理 (NLP) 技能，以無縫接軌當前的生成式 AI (Generative AI) 開發浪潮。

課程設計遵循「第一原理 -> 核心基礎 -> 知識體系」的學習路徑，確保學習者不僅能掌握工具，更能理解其背後的核心思想，並將所學應用於真實世界的專案中。

---

## Part 0: Introduction

*   **Chapter 0: Course Overview (C0-S1)**
    *   [Unit 1: Course Overview](./nlp-course/part-0_introduction/chapter-0_course-overview/C0-S1-U1_course-overview.ipynb)
    *   [Unit 2: Course Logistics](./nlp-course/part-0_introduction/chapter-0_course-overview/C0-S1-U2_course-logistics.ipynb)

---

## Part 1: Fundamentals

#### Module 1: Data to Vector
*   **Chapter 1: Text Cleaning and Normalization (C1-S1)**
    *   [Unit 1: Chinese Text Preprocessing with Jieba](./nlp-course/part-1_fundamentals/module-1_data-to-vector/chapter-1_text-cleaning-and-normalization/C1-S1-U1_chinese-text-preprocessing-with-jieba.ipynb)
    *   [Unit 2: English Text Preprocessing with NLTK](./nlp-course/part-1_fundamentals/module-1_data-to-vector/chapter-1_text-cleaning-and-normalization/C1-S1-U2_english-text-preprocessing-with-nltk.ipynb)
    *   [Unit 3: Chinese Character Conversion with OpenCC](./nlp-course/part-1_fundamentals/module-1_data-to-vector/chapter-1_text-cleaning-and-normalization/C1-S1-U3_chinese-character-conversion-with-opencc.ipynb)
*   **Chapter 2: Tokenization**
    *   **Sub-Chapter 2.1: Basic Tokenization (C2-S1)**
        *   [Unit 1: Visualization with Word Cloud](./nlp-course/part-1_fundamentals/module-1_data-to-vector/chapter-2_tokenization/C2-S1-U1_visualization-with-word-cloud.ipynb)
    *   **Sub-Chapter 2.2: Tokenization Materials (C2-S2)**
        *   [Unit 1: Tokenization Concepts](./nlp-course/part-1_fundamentals/module-1_data-to-vector/chapter-2_tokenization/chapter-2_tokenization-materials/C2-S2-U1_tokenization-concepts.ipynb)
        *   [Unit 2: Tokenization Slides](./nlp-course/part-1_fundamentals/module-1_data-to-vector/chapter-2_tokenization/chapter-2_tokenization-materials/C2-S2-U2_tokenization-slides.ipynb)

#### Module 2: Understanding Structure and Semantics
*   **Chapter 3: Parsing (C3-S1)**
    *   [Unit 1: Dependency Parsing Slides](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-3_parsing/C3-S1-U1_dependency-parsing-slides.ipynb)
    *   [Unit 2: Transition-Based Dependency Parsing](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-3_parsing/C3-S1-U2_transition-based-dependency-parsing.ipynb)
*   **Chapter 4: POS & NER**
    *   **Sub-Chapter 4.1: Basic POS & NER (C4-S1)**
        *   [Unit 1: POS Tagging and NER](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-4_pos-and-ner/C4-S1-U1_pos-tagging-and-ner.ipynb)
    *   **Sub-Chapter 4.2: Sequence Labeling Materials (C4-S2)**
        *   [Unit 1: Sequence Labeling Slides](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-4_pos-and-ner/chapter-4_sequence-labeling-materials/C4-S2-U1_sequence-labeling-slides.ipynb)
*   **Chapter 5: Word Embeddings**
    *   **Sub-Chapter 5.1: Count-Based Models (C5-S1)**
        *   [Unit 1: One-Hot Encoding](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.1_count-based-models/C5-S1-U1_one-hot-encoding.ipynb)
        *   [Unit 2: Bag-of-Words](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.1_count-based-models/C5-S1-U2_bag-of-words.ipynb)
        *   [Unit 3: TF-IDF](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.1_count-based-models/C5-S1-U3_tf-idf.ipynb)
        *   [Unit 4: Static Embedding Slides](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.1_count-based-models/C5-S1-U4_static-embedding-slides.ipynb)
    *   **Sub-Chapter 5.2: Prediction-Based Models (C5-S2)**
        *   [Unit 1: Word2Vec](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.2_prediction-based-models/C5-S2-U1_word2vec.ipynb)
        *   [Unit 2: Doc2Vec](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.2_prediction-based-models/C5-S2-U2_doc2vec.ipynb)
        *   [Unit 3: Contextual Embedding Slides](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.2_prediction-based-models/C5-S2-U3_contextual-embedding-slides.ipynb)
        *   [Unit 4: Transformer Embedding Slides](./nlp-course/part-1_fundamentals/module-2_structure-and-semantics/chapter-5_word-embeddings/sub-chapter-5.2_prediction-based-models/C5-S2-U4_transformer-embedding-slides.ipynb)

---

## Part 2: Applications

*   **Chapter 6: Classification (C6-S1)**
    *   [Unit 1: Document Classification Slides](./nlp-course/part-2_applications/chapter-6_classification/C6-S1-U1_document-classification-slides.ipynb)
*   **Chapter 7: Language Modeling (C7-S1)**
    *   [Unit 1: Language Modeling Slides](./nlp-course/part-2_applications/chapter-7_language-modeling/C7-S1-U1_language-modeling-slides.ipynb)
*   **Chapter 8: Information Extraction (C8-S1)**
    *   [Unit 1: Information Extraction Slides](./nlp-course/part-2_applications/chapter-8_information-extraction/C8-S1-U1_information-extraction-slides.ipynb)
    *   [Unit 2: Relation Extraction Notes](./nlp-course/part-2_applications/chapter-8_information-extraction/C8-S1-U2_relation-extraction-notes.ipynb)
*   **Chapter 9: Question Answering (C9-S1)**
    *   [Unit 1: Question Answering Slides](./nlp-course/part-2_applications/chapter-9_question-answering/C9-S1-U1_question-answering-slides.ipynb)
*   **Chapter 10: Sequence-to-Sequence (C10-S1)**
    *   [Unit 1: NMT Slides](./nlp-course/part-2_applications/chapter-10_sequence-to-sequence/C10-S1-U1_nmt-slides.ipynb)
    *   [Unit 2: RNN for Seq2Seq Slides](./nlp-course/part-2_applications/chapter-10_sequence-to-sequence/C10-S1-U2_rnn-for-seq2seq-slides.ipynb)

---

## Part 3: Advanced Topics

*   **Chapter 11: Transfer Learning (C11-S1)**
    *   [Unit 1: Transfer Learning Slides](./nlp-course/part-3_advanced-topics/chapter-11_transfer-learning/C11-S1-U1_transfer-learning-slides.ipynb)
    *   [Unit 2: Cross-Lingual Transfer Learning Slides](./nlp-course/part-3_advanced-topics/chapter-11_transfer-learning/C11-S1-U2_cross-lingual-transfer-learning-slides.ipynb)
*   **Chapter 12: Interpretability (C12-S1)**
    *   [Unit 1: Interpretability Slides](./nlp-course/part-3_advanced-topics/chapter-12_interpretability/C12-S1-U1_interpretability-slides.ipynb)

---

## Part 4: Projects

*   **Project 1: Ticket Classifier**
    *   [./nlp-course/part-4_projects/project-1_ticket-classifier/](./nlp-course/part-4_projects/project-1_ticket-classifier/)
*   **Project 2: Resume Parser**
    *   (尚未建立)
*   **Project 3: RAG Chatbot**
    *   (尚未建立)

---

## Archived Old Materials

此目錄 `archived_nlp_materials` 包含專案的舊版教材、未整理的筆記本或重複的檔案。這些檔案已被移出主要課程路徑，以保持結構清晰，但仍然可供參考。
