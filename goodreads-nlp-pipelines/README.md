# 📚 Goodreads NLP Pipeline: Classification, Clustering, Embeddings & BERT Fine-Tuning

This repository contains my end-to-end engineering pipelines for advanced text mining, vector representation learning, and transformer fine-tuning using a Goodreads book metadata dataset. The codebase bridges classic machine learning methodologies with modern parameter-efficient deep learning architectures.

### 👤 Author
* **Developer:** [Arvin Baghal Asl](https://github.com/arvinasli)
* **Institution:** Sharif University of Technology (SUT)
* **Course:** Modern Information Retrieval (Spring 2026)  
* **Instructor:** [Dr. Mahdieh Soleymani Baghshah](https://scholar.google.com/citations?user=S1U0KlgAAAAJ&hl=en)

---

## 🏗 Repository Architecture & Components

The repository is structured into four distinct machine learning pipelines, spanning supervised learning, unsupervised clustering, custom semantic representations, and deep transformer architectures.

### 1️⃣ Book Genre Classification (Supervised Learning)
This pipeline implements a multi-class text classification framework designed to predict book genres from raw text descriptions. 
* **Text Representations Compared:** High-dimensional sparse vectors via **TF-IDF** versus dense semantic vectors via pre-computed **Sentence Embeddings**.
* **Classifiers Evaluated:** Naive Bayes, k-Nearest Neighbors (k-NN), Logistic Regression, and Multi-Layer Perceptrons (MLP).
* **Imbalance Engineering:** Implemented and benchmarked systematic class-imbalance mitigation strategies on the development set prior to final test-set deployment.
* **Analysis Suite:** Performance evaluated via comprehensive model comparison plots, multi-class confusion matrices, per-class F1-scores, and semantic error-case inspections.

### 2️⃣ Book Description Clustering (Unsupervised Learning)
This module explores the capacity of unsupervised learning algorithms to recover underlying latent themes and semantic groupings from text data without exposure to categorical labels.
* **Feature Spaces:** Parallel evaluations run across both text-based **TF-IDF matrices** and dense **Sentence Embedding spaces**.
* **Algorithms Implemented:** **K-Means**, **Spherical K-Means** (cosine distance optimized), **DBSCAN**, and **Hierarchical Agglomerative Clustering**.
* **Evaluation Framework:** True genre classifications are held out entirely during training and utilized exclusively post-clustering to assess cluster purity, alignment, and semantic coherence.

### 3️⃣ Custom & Subword Embeddings (Representation Learning)
A self-contained representation learning framework that processes book metadata (titles, authors, genres, descriptions, and regions) into customized, continuous semantic spaces.
* **Word2Vec from Scratch:** Implemented a native **Skip-Gram Word2Vec model with Negative Sampling** from the ground up to handle custom vocabulary builds, token normalization, and training pair generation.
* **Subword Representations (Gensim FastText):** Trained a subword-aware FastText architecture using the `gensim` engine to preserve morphological properties and manage out-of-vocabulary (OOV) tokens.
* **Native FastText:** Leveraged the official native `fasttext` package to establish an optimization and behavioural benchmark against the Gensim implementation.
* **Evaluation & Vector Mapping:** Explored linguistic properties using analogy tests and vectorized structural projections down to a 2D space for comparative visualizations.

### 4️⃣ Deep Learning: Transformer Fine-Tuning Scenarios
A PyTorch-driven optimization workspace examining performance profiles and efficiency tradeoffs across distinct fine-tuning paradigms using a pre-trained `bert-base-uncased` backbone.

| Fine-Tuning Scenario | Architectural Profile | Trainable Parameters |
| :--- | :--- | :--- |
| **Full Fine-Tuning** | Backpropagates through all layers of the pre-trained BERT model alongside an added linear classification head. | Maximum (100% of weights) |
| **LoRA Fine-Tuning** | Injects low-rank decomposition matrices into the transformer layers while freezing the base parameters using HuggingFace `peft`. | Highly Reduced |
| **Few-Shot Learning** | Freezes the entire base BERT encoder and exclusively trains a small, custom downstream neural head over the static `[CLS]` token embedding. | Minimal |

* **Engineering Infrastructure:** Built clean, reusable data components featuring a custom `TitleLabelEncoder`, a robust PyTorch `GoodreadsDataset` handling tokenization batching, and an isolated `Trainer` engine coordinating loss calculation, gradient steps, validation tracking, and accuracy evaluation.
* **Data Scarcity Analysis:** Evaluated model convergence, training stability, and accuracy scaling profiles under constrained, micro-dataset conditions to evaluate LoRA vs. Few-Shot resilience.

---

## 🚀 Getting Started

### 1. Prerequisites & Dataset
Ensure your local data configuration maps to the pipeline inputs:
* **Target File:** `preprocessed.json` (Ensure this file is placed inside your designated data path before running the execution cells).

### 2. Environment Installation
Install the core deep learning engines, tokenizers, and parameter-efficient adapters:
```bash
pip install torch torchao transformers peft gensim scikit-learn matplotlib
