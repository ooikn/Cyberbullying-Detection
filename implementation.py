import json
import pandas as pd
import numpy as np
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
import joblib

# 1. Load dataset
with open('processed_dataset_labeled.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
labels = df['cyberbullying_label'].values.astype(int)

# Check available columns
print("Available columns:", df.columns)

# 2. Prepare text for BERT
if 'text' in df.columns:
    texts = df['text'].tolist()
else:
    # Fall back to tokenized_text if raw text not available
    texts = [' '.join(tokens) for tokens in df['tokenized_text']]

tokenized_sentences = df['tokenized_text'].tolist()

# 3. Load pre-trained GloVe embeddings
print("Loading GloVe embeddings...")
glove_model = api.load("glove-wiki-gigaword-100")

def get_mean_vector(tokens, model, vector_size):
    words = [word for word in tokens if word in model.key_to_index]
    if len(words) >= 1:
        return np.mean([model[word] for word in words], axis=0)
    else:
        return np.zeros(vector_size)

vector_size_glove = 100
glove_features = np.array([get_mean_vector(text, glove_model, vector_size_glove)
                           for text in tokenized_sentences])

# 4. Load pre-trained BERT model and tokenizer
print("Extracting BERT embeddings...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

bert_features = np.array([get_bert_embedding(t) for t in texts])

# 5. Combine GloVe + BERT features
combined_features = np.concatenate((glove_features, bert_features), axis=1)
print(f"Combined Feature Shape: {combined_features.shape}")

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# 7. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 8. Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Save model and features
np.save('features_hybrid.npy', combined_features)
np.save('labels.npy', labels)
joblib.dump(clf, 'cyberbullying_hybrid_classifier.pkl')

print("Hybrid Model Training Complete.")
