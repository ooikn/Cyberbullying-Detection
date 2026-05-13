import pandas as pd
import re
import json
from pathlib import Path

DATASET_FILE = "threads_posts_dataset.csv"
MEDIA_DIR = Path("live_media_downloads")

OUTPUT_JSON = "processed_dataset.json"

df = pd.read_csv(DATASET_FILE)

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub("", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize(text):
    return text.split()

def get_media_files(index):

    post_num = str(index + 1).zfill(2)

    images = []
    videos = []

    for file in MEDIA_DIR.iterdir():

        if file.name.startswith(f"image_{post_num}_"):
            images.append(file.name)

        elif file.name.startswith(f"video_{post_num}_"):
            videos.append(file.name)

    return images if images else None, videos if videos else None

processed_data = []

for idx, row in df.iterrows():

    cleaned_text = clean_text(row["original_post"])
    tokenized_text = tokenize(cleaned_text)

    image_files, video_files = get_media_files(idx)

    processed_data.append({
        "tokenized_text": tokenized_text,
        "image_file": image_files,
        "video_file": video_files,
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print("Done!")
print(f"Saved: {OUTPUT_JSON}")