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

**Install dependencies**
pip install numpy pandas scikit-learn gensim torch transformers datasets joblib

**Dataset Setup**
Due to GitHub’s file size limits, the images and videos dataset is hosted externally.
Download from Google Drive:
Cyberbullying-Detection/media_folder_in_drive/
