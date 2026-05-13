import pandas as pd
import json

df = pd.read_csv("threads_posts_dataset.csv")

# Keep only the first occurrence of each unique post ID
df_unique = df.drop_duplicates(subset="id", keep="first")

records = df_unique.to_dict(orient="records")
with open("threads_posts_dataset_deduplicated.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

print(f"Deduplication complete. {len(df) - len(df_unique)} duplicate posts removed.")
print(f"Saved: threads_posts_dataset_deduplicated.csv and threads_posts_dataset_deduplicated.json")
