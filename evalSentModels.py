import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import nltk
from nltk.tokenize import word_tokenize 
nltk.download('punkt', force=True)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_metrics_to_tsv(file_path, metrics):
    df = pd.DataFrame(metrics)
    df.to_csv(file_path, sep="\t", index=False)
    print(f"Metrics saved to {file_path}")

def make_int(val):
    if (val > 0):
        return 1
    elif (val < 0):
        return -1
    else:
        return 0

def process_sentiments(data, output_file):
    actual_polarity = []
    predicted_polarity = []

    actual_subjectivity = []
    predicted_subjectivity = []

    num_posts = 0
    num_comments = 0

    for post in data:
        num_posts += 1
        sentiment = post.get("sentiment", {})
        
        if "actual polarity" in sentiment and "polarity" in sentiment:
            actual_value = sentiment["actual polarity"]
            predicted_value = make_int(sentiment["polarity"])
            actual_polarity.append(actual_value)
            predicted_polarity.append(predicted_value)

        # Handle post subjectivity
        if "actual subjectivity" in sentiment and "subjectivity" in sentiment:
            actual_value = sentiment["actual subjectivity"]
            predicted_value = make_int(sentiment["subjectivity"])
            actual_subjectivity.append(actual_value)
            predicted_subjectivity.append(predicted_value)

        for comment in post.get("comments", []):
            num_comments += 1
            comment_sentiment = comment.get("sentiment", {})
            
            if "actual polarity" in comment_sentiment and "polarity" in comment_sentiment:
                actual_value = comment_sentiment["actual polarity"]
                predicted_value = make_int(comment_sentiment["polarity"])
                actual_polarity.append(actual_value)
                predicted_polarity.append(predicted_value)

            if "actual subjectivity" in comment_sentiment and "subjectivity" in comment_sentiment:
                actual_value = comment_sentiment["actual subjectivity"]
                predicted_value = make_int(comment_sentiment["subjectivity"])
                actual_subjectivity.append(actual_value)
                predicted_subjectivity.append(predicted_value)

    Eval_metric = ["Precision", "Recall", "Accuracy", "F1 Score"]

    
    Polarity_metrics = [
                precision_score(actual_polarity, predicted_polarity, average="macro"),
                recall_score(actual_polarity, predicted_polarity, average="macro"),
                accuracy_score(actual_polarity, predicted_polarity),
                f1_score(actual_polarity, predicted_polarity, average="macro"),
            ]

    Subjectivity_metric = [
                precision_score(actual_subjectivity, predicted_subjectivity, average="macro"),
                recall_score(actual_subjectivity, predicted_subjectivity, average="macro"),
                accuracy_score(actual_subjectivity, predicted_subjectivity),
                f1_score(actual_subjectivity, predicted_subjectivity, average="macro"),
            ]
    
    metrics = {
        "Evaluation Metrics": Eval_metric,
        "Polarity": Polarity_metrics,
        "Subjectivity": Subjectivity_metric

    }

    
    metrics_df = pd.DataFrame(metrics)

    save_metrics_to_tsv(output_file, metrics_df)

    print("\nTotal Posts Processed:", num_posts)
    print("Total Comments Processed:", num_comments)

if __name__ == "__main__":
    json_file = "post_comments_tweet_labeled.json"
    output_file = "metrics_tweetSent_finetuned2.tsv"

    data = load_json(json_file)
    process_sentiments(data, output_file)