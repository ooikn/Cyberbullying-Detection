# Cyberbullying-Detection
This repository implements a multimodal cyberbullying detection pipeline.
It uses:
1. Text embeddings (Word2Vec, GloVe, BERT)

2. Hybrid feature representation (combining GloVe + BERT)

3. Classification models (Logistic Regression, fine‑tuned transformer)

The goal is to detect online harassment and aggressive language in social media posts.

Installation Guide

**Clone the repository**
git clone https://github.com/ooikn/Cyberbullying-Detection.git
cd Cyberbullying-Detection

**Dataset Setup**
Due to GitHub’s file size limits, the images and videos dataset folder (live_media_downloads) is placed into Google Drive.
1. Download from Google Drive:
Cyberbullying-Detection/media_folder_in_drive/

2. Move the live_media_downloads folder to the Cyberbullying Detection directory.

**Install dependencies**
pip install numpy pandas scikit-learn gensim torch transformers datasets joblib

##  Deduplication

The script **`deduplication.py`** ensures dataset quality by checking for repeated post and eliminating duplicates.  
This step guarantees that the dataset contains **unique samples only**, preventing bias during training and evaluation.

Run:
```bash
python deduplication.py
```

##  Preprocessing

The script **`preprocessing.py`** cleans the text by removing emojis, special characters, and URLs, converting to lowercase, and tokenizing into word lists.  
The output is saved in **`processed_dataset.json`**.

Run:
```bash
python preprocessing.py
```

## Implementation

The core script **`implementation.py`** integrates the hybrid feature representation and classification pipeline:

- **Feature Extraction**
  - Generates embeddings using **Word2Vec** (via Gensim), **GloVe**, and **BERT** (via HuggingFace Transformers).
  - Combines GloVe and BERT embeddings into a **hybrid representation** to capture both semantic and contextual features.

- **Model Training**
  - **Logistic Regression** is used as a baseline classifier on the hybrid features.
  - A **fine‑tuned transformer model (BERT)** is trained for deeper contextual understanding of cyberbullying language.

- **Evaluation**
  - Dataset split into **80:20 (train:test)** ratio.
  - Predictions compared against true labels.
  - Metrics reported: **Accuracy, Precision, Recall, F1‑score**.

- **Reproducibility**
  - Preprocessing scripts ensure consistent cleaning (emoji removal, lowercasing, tokenization).
  - Processed datasets are saved in both **CSV** and **JSON** formats for reuse.
  - Media files (images/videos) are linked to posts via a dedicated folder.

Run:
```bash
python implementation.py
```


