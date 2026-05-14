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

**Deduplication**
The script deduplication.py checks for repeated text entries and eliminates them. This ensure the dataset contain unique samples only.

**Preprocessing**
The script preprocessing.py clean the text by removing the emojis, special characters, URLS, convert to lower case and tokenize into word lists. The output is saved in processed_dataset.json.

**Implementation**
Embeddings: Word2Vec (via Gensim), GloVe, and BERT (via Transformers).
Hybrid Representation: Combination of GloVe + BERT embeddings.
Models:
Logistic Regression (baseline)
Fine‑tuned Transformer (BERT)


