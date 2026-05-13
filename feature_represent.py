import json
import pandas as pd
import numpy as np
import gensim.downloader as api

# 1. Load your dataset
with open('processed_dataset_labeled.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Tokenized sentences (already processed in your dataset)
tokenized_sentences = df['tokenized_text'].tolist()

# 2. Load pre-trained GloVe embeddings (100 dimensions)
print("Loading GloVe embeddings...")
glove_model = api.load("glove-wiki-gigaword-100")

# 3. Function to compute mean vector for a sentence
def get_mean_vector(tokens, model, vector_size):
    words = [word for word in tokens if word in model.key_to_index]
    if len(words) >= 1:
        return np.mean([model[word] for word in words], axis=0)
    else:
        return np.zeros(vector_size)

# 4. Generate feature matrix
vector_size = 100
features = np.array([get_mean_vector(text, glove_model, vector_size) 
                     for text in tokenized_sentences])

# 5. Extract labels
labels = df['cyberbullying_label'].values.astype(int)

# 6. Save features and labels
np.save('features_glove.npy', features)
np.save('labels.npy', labels)

print("GloVe Feature Representation Complete.")
print(f"Feature Matrix Shape: {features.shape}")  # Example: (511, 100)
