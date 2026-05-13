import pandas as pd
import re
import json
from pathlib import Path

DATASET_FILE = "threads_posts_dataset.csv"
MEDIA_DIR = Path("live_media_downloads")

OUTPUT_JSON = "processed_dataset_labeled.json"

df = pd.read_csv(DATASET_FILE, encoding="utf-8")

df = df.drop_duplicates(subset="id", keep="first")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", "", text)             # remove emojis/non-ASCII
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)            # remove special chars
    return text.lower().strip()

df["cleaned_text"] = df["original_post"].apply(clean_text)

df["tokenized_text"] = df["cleaned_text"].apply(lambda t: t.split())

def get_media_files(index):
    post_num = str(index + 1).zfill(2)
    images, videos = [], []
    for file in MEDIA_DIR.iterdir():
        if file.name.startswith(f"image_{post_num}_"):
            images.append(str(file))
        elif file.name.startswith(f"video_{post_num}_"):
            videos.append(str(file))
    return images if images else None, videos if videos else None

df["image_file"], df["video_file"] = zip(*[get_media_files(idx) for idx in range(len(df))])

keywords = [
    "stupid","idiot","dumb","ugly","loser","trash","garbage","worthless",
    "pathetic","useless","moron","fool","clown","disgusting","failure",
    "nobody","annoying","hopeless","irritating","brainless","fake","awful",
    "terrible","disaster","joke","ridiculous","lame","gross","scum","creep",
    "rat","pig","animal","monster","degenerate","weirdo","psycho","reject",
    "outcast","burden","parasite","waste","space","fat","skinny","hideous",
    "deformed","unattractive","clueless","dense","mid","dogwater","npc","simp",
    "noob","cringe"
]

expanded_keywords = set()
for phrase in keywords:
    for word in phrase.split():
        expanded_keywords.add(word.lower())

def classify(text):
    text = str(text).lower()
    for kw in expanded_keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            return True
    return False

df["cyberbullying_label"] = df["cleaned_text"].apply(classify)

df_final = df[["tokenized_text","image_file","video_file","cyberbullying_label"]]

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(df_final.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

count_true = df["cyberbullying_label"].sum()
count_false = len(df) - count_true

print("✅ Final JSON dataset created.")
print(f"Saved: {OUTPUT_JSON}")
print(f"Cyberbullying posts: {count_true}")
print(f"Non-cyberbullying posts: {count_false}")
