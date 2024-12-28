import json
import pandas as pd
from sklearn.model_selection import train_test_split

with open('train_data.json', 'r', encoding="utf-8") as f: 
    data = json.load(f)  

texts = []
labels = []
full_records = []

for post in data:  
    record = {"post_id": post["post_id"], "title": post.get("title", ""), "sentiment": post.get("sentiment", {})}
    if "sentiment" in post and "actual_polarity" in post["sentiment"]:
        texts.append(post["title"])
        labels.append(post["sentiment"]["actual_polarity"] + 1)  # Convert to 0, 1, 2 for training
        record["actual_polarity"] = post["sentiment"]["actual_polarity"]

    comments_list = []
    if "comments" in post:
        for comment in post["comments"]:
            if "text" in comment and "sentiment" in comment and "actual_polarity" in comment["sentiment"]:
                texts.append(comment["text"])
                labels.append(comment["sentiment"]["actual_polarity"] + 1)
                comments_list.append({
                    "comment_id": comment["comment_id"],
                    "text": comment["text"],
                    "sentiment": comment["sentiment"],
                })
    record["comments"] = comments_list
    full_records.append(record)

if len(texts) == 0 or len(labels) == 0:
    raise ValueError("No valid data found in the JSON file. Check the structure or content.")

df = pd.DataFrame({"text": texts, "label": labels})  # For CSV export
full_df = pd.DataFrame(full_records)  # For JSON export

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_json, test_json = train_test_split(full_df, test_size=0.2, random_state=42)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

train_json_records = train_json.to_dict(orient="records")
test_json_records = test_json.to_dict(orient="records")

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_json_records, f, indent=4, ensure_ascii=False)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_json_records, f, indent=4, ensure_ascii=False)

total_actual_polarity = len(labels)
print(f"Total occurrences of actual polarity: {total_actual_polarity}")

print("Data successfully saved to train.csv, test.csv, train.json, and test.json!")