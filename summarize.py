import json
import csv
from datetime import datetime
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
import numpy as np
import logging
import warnings
import stanza
from transformers import pipeline
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

stanza.download('en')  

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)


class ExtractiveSummarizer:
    def __init__(self):
        #sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0,  
        )

        #TweebankNLP pipeline
        config = {
            'processors': 'tokenize,lemma,pos,depparse,ner',
            'lang': 'en',
            'tokenize_model_path': './twitter-stanza/saved_models/tokenize/en_tweet_tokenizer.pt',
            'lemma_model_path': './twitter-stanza/saved_models/lemma/en_tweet_lemmatizer.pt',
            'pos_model_path': './twitter-stanza/saved_models/pos/en_tweet_tagger.pt',
            'depparse_model_path': './twitter-stanza/saved_models/depparse/en_tweet_parser.pt',
            'ner_model_path': './twitter-stanza/saved_models/ner/en_tweet_nertagger.pt',
        }
        self.tweebank_nlp = stanza.Pipeline(**config)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english") + list(punctuation) + ["game", "update", "patch", "play"])

    @staticmethod
    def flatten_comments(comments):
        flat_comments = []
        for item in comments:
            if isinstance(item, list):
                flat_comments.extend(ExtractiveSummarizer.flatten_comments(item))
            elif isinstance(item, dict):
                flat_comments.append(item)
            else:
                logging.warning(f"Unexpected comment format: {item}")
        return flat_comments
    
    def preprocess_text(self, text):
        text = text.replace("#", "").replace("_", " ") 
        doc = self.tweebank_nlp(text)
        tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
        return [token for token in tokens if token not in self.stop_words]

    def extract_entities(self, text):
        """Extract entities using TweebankNLP NER."""
        doc = self.tweebank_nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "type": ent.type})
        return entities
    
    def parse_dependencies(self, text):
        """Extract dependency relations using TweebankNLP."""
        doc = self.tweebank_nlp(text)
        dependencies = []
        for sentence in doc.sentences:
            for word in sentence.words:
                dependencies.append({
                    "word": word.text,
                    "head": sentence.words[word.head - 1].text if word.head > 0 else "ROOT",
                    "relation": word.deprel
                })
        return dependencies
    
    def get_sentence_scores(self, sentences, word_freq, comments):
        sentence_scores = defaultdict(float)
        for i, sentence in enumerate(sentences):
            words = self.preprocess_text(sentence)
            word_count = len(words)
            if word_count <= 0:
                continue
            for word in words:
                if word in word_freq:
                    sentence_scores[sentence] += word_freq[word]
            sentence_scores[sentence] /= word_count
    
            #boosting scores based on sentiment instead of word frequency
            sentiment_label = comments[i]["sentiment"]["label"]
            if sentiment_label == "positive":
                sentence_scores[sentence] *= 1.1
            elif sentiment_label == "negative":
                #critiques shouls impact the score more
                sentence_scores[sentence] *= 1.3
        return sentence_scores



    def cluster_sentences(self, sentences):
        # Train Word2Vec on tokenized sentences
        tokenized_sentences = [self.preprocess_text(sentence) for sentence in sentences]
        word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        # Compute sentence embeddings by averaging word embeddings
        def sentence_embedding(sentence):
            words = [word for word in sentence if word in word2vec_model.wv]
            if not words: return np.zeros(word2vec_model.vector_size)
            return np.mean([word2vec_model.wv[word] for word in words], axis=0)
        
        embeddings = np.array([sentence_embedding(sentence) for sentence in tokenized_sentences])
        
        # Cluster embeddings using KMeans
        n_clusters = min(len(sentences), 5)  # Cap clusters at 5 or number of sentences
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
        return kmeans.labels_


    def summarize(self, comments, title="", num_sentences=5):
        comments = self.flatten_comments(comments)
        comments = [
            comment
            for comment in comments
            if isinstance(comment, dict) and comment.get("text", "").strip() not in ["[deleted]", "[removed]"]
        ]
    
        if not comments:
            logging.warning("No valid comments found for summarization.")
            return {
                "title": title,
                "summary": ["No valid text found for summarization"],
                "metrics": {
                    "total_comments": 0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                },
            }
    
        # Analyze sentiment for each comment
        for comment in comments:
            try:
                sentiment_result = self.sentiment_analyzer(comment["text"][:512])
                if sentiment_result:
                    comment["sentiment"] = {
                        "label": sentiment_result[0]["label"].lower(),
                        "score": sentiment_result[0]["score"],
                    }
                else:
                    comment["sentiment"] = {"label": "neutral", "score": 0.0}
            except Exception as e:
                logging.warning(f"Error analyzing sentiment: {str(e)}")
                comment["sentiment"] = {"label": "neutral", "score": 0.0}
    
        all_text = " ".join(comment.get("text", "") for comment in comments)
        if not all_text.strip():
            logging.warning("No valid text found after combining comments.")
            return {"title": title, "summary": ["No valid text for summarization"], "metrics": {}}
    
        sentences = nltk.sent_tokenize(all_text)
        if len(sentences) != len(comments):
            sentences = sentences[:len(comments)] if len(sentences) > len(comments) else sentences
            comments = comments[:len(sentences)] if len(comments) > len(sentences) else comments
    
        # Step 1: Calculate word frequency and sentence scores
        word_freq = nltk.FreqDist(self.preprocess_text(all_text))
        sentence_scores = self.get_sentence_scores(sentences, word_freq, comments)
    
        if not sentence_scores:
            return {"title": title, "summary": ["No sentences scored for summarization"], "metrics": {}}
    
        # Step 2: Cluster sentences and pick representatives
        cluster_labels = self.cluster_sentences(sentences)
        cluster_representatives = []
        for cluster in range(max(cluster_labels) + 1):
            # Find all sentences in the current cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
            # Pick the highest-scoring sentence from the cluster
            best_sentence_idx = max(cluster_indices, key=lambda idx: sentence_scores.get(sentences[idx], 0))
            cluster_representatives.append(sentences[best_sentence_idx])
    
        # Step 3: Sort representatives to preserve original order in the text
        top_sentences = sorted(cluster_representatives, key=lambda x: sentences.index(x))
    
        # Step 4: Format the summary
        formatted_summary = "\n".join(f"- {s}" for s in top_sentences)
    
        # Step 5: Sentiment metrics
        return {
            "title": title,
            "summary": formatted_summary,
            "metrics": {
                "total_comments": len(comments),
                "sentiment_distribution": {
                    "positive": sum(c["sentiment"]["label"] == "positive" for c in comments),
                    "negative": sum(c["sentiment"]["label"] == "negative" for c in comments),
                    "neutral": sum(c["sentiment"]["label"] == "neutral" for c in comments),
                },
            },
        }


def generate_and_save_reports(input_file, output_dir="reports"):
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        with open(input_file, "r") as f:
            data = json.load(f)

        summarizer = ExtractiveSummarizer()
        tsv_rows = []
        text_report = []

        if isinstance(data, dict):
            data = [data]

        # Process each discussion
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue

            title = entry.get("title", f"Discussion {idx + 1}")
            comments = entry.get("comments", [])
            summary_result = summarizer.summarize(comments, title)

            tsv_row = {
                "Title": title,
                "Total Comments": summary_result["metrics"]["total_comments"],
                "Positive Comments": summary_result["metrics"]["sentiment_distribution"]["positive"],
                "Negative Comments": summary_result["metrics"]["sentiment_distribution"]["negative"],
                "Neutral Comments": summary_result["metrics"]["sentiment_distribution"]["neutral"],
                "Key Points": summary_result["summary"],
            }
            tsv_rows.append(tsv_row)

            text_report.append(
                f"""
Discussion: {title}
{'=' * len(title)}
{summary_result["summary"]}

Metrics:
Total Comments: {summary_result["metrics"]["total_comments"]}
Positive Comments: {summary_result["metrics"]["sentiment_distribution"]["positive"]}
Negative Comments: {summary_result["metrics"]["sentiment_distribution"]["negative"]}
Neutral Comments: {summary_result["metrics"]["sentiment_distribution"]["neutral"]}
"""
            )

        tsv_file = Path(output_dir) / f"discussion_summary_{timestamp}.tsv"
        with open(tsv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tsv_rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(tsv_rows)

        text_file = Path(output_dir) / f"discussion_detailed_{timestamp}.txt"
        with open(text_file, "w") as f:
            f.write("\n".join(text_report))

        return {
            "tsv_file": str(tsv_file),
            "text_file": str(text_file),
            "processed_discussions": len(tsv_rows),
        }

    except Exception as e:
        logging.error(f"Error generating reports: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        input_file = "post_comments_labeled.json" 
        result = generate_and_save_reports(input_file)
        print(f"Reports generated successfully!\n- TSV: {result['tsv_file']}\n- Text: {result['text_file']}\n")
    except Exception as e:
        print(f"Error: {str(e)}")