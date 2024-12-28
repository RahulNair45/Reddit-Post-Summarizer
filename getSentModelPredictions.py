from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import json

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def find_text_sent(text, tokenizer, model, config):
    text = preprocess(text)
    max_length = 512 
    stride = 256  


    tokenized = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length", 
    )
    

    chunk_sentiments = []
    for i, chunk in enumerate(tokenized["input_ids"]):
        output = model(input_ids=chunk.unsqueeze(0))  
        scores = output.logits[0].detach().numpy() 
        scores = softmax(scores) 
        predicted_index = np.argmax(scores)  
        sentiment_mapping = {0: -1, 1: 0, 2: 1} 
        chunk_sentiments.append(sentiment_mapping[predicted_index])


    overall_sentiment = round(sum(chunk_sentiments) / len(chunk_sentiments))
    return overall_sentiment

def give_post_sents(filename, tokenizer, model, config):

    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)


    for post in data:
        if 'title' in post:
            text = post['title']
            if 'sentiment' in post and isinstance(post['sentiment'], dict) and 'polarity' in post['sentiment']:
                sent_val = find_text_sent(text, tokenizer, model, config)
                post['sentiment']['polarity'] = sent_val

                print(f"Processed Title: {text}\nSentiment: {sent_val}\n")
        if 'comments' in post and isinstance(post['comments'], list):
            for comment in post['comments']:
                if 'text' in comment:
                    text = comment['text']
                    if 'sentiment' in comment and isinstance(comment['sentiment'], dict) and 'polarity' in comment['sentiment']:
                        sent_val = find_text_sent(text, tokenizer, model, config)
                        comment['sentiment']['polarity'] = sent_val

                        print(f"Processed Comment: {text}\nSentiment: {sent_val}\n")


    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Main function
def main():
    # Use your fine-tuned model's directory
    # ORIGINAL_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Replace with the original model ID
    # ORIGINAL_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    ORIGINAL_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # ORIGINAL_MODEL = "siebert/sentiment-roberta-large-english"
    # FINE_TUNED_MODEL = "fine_tuned_roberta_sentiment"  # Replace with the path to your fine-tuned model
    FINE_TUNED_MODEL = "fine_tuned_twitter" 

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)

    config = AutoConfig.from_pretrained(ORIGINAL_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL, config=config)

    filename = "post_comments_tweet_labeled.json"
    give_post_sents(filename, tokenizer, model, config)

if __name__ == "__main__":
    main()