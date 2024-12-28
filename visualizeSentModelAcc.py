import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_process_tsv(file_paths, model_names):
    """
    Load multiple TSV files and add a 'Model' column to identify them.
    """
    dataframes = []
    for file, model in zip(file_paths, model_names):

        df = pd.read_csv(file, sep='\t')
        
        df.rename(columns={"Evaluation Metrics": "Metric"}, inplace=True)
        
        df['Model'] = model
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

file_paths = [
    "metrics.tsv", 
    "metrics_large-english.tsv", 
    "metrics_sst-2.tsv", 
    "metrics_tweetSent.tsv", 
    "metrics_tweetSent_finetuned.tsv"
]
model_names = [
    "NLTK", 
    "siebert/sentiment-roberta-large-english",  
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
    "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    "cardiffnlp/twitter-roberta-base-sentiment-latest finetuned on reddit"
]

data = load_and_process_tsv(file_paths, model_names)

plt.figure(figsize=(12, 8))
barplot = sns.barplot(data=data, x="Metric", y="Polarity", hue="Model")

for container in barplot.containers:
    barplot.bar_label(container, fmt="%.3f", label_type="edge", padding=3)

plt.legend([], [], frameon=False)  
plt.title("Model Evaluation Metrics (Polarity)", fontsize=16)
plt.ylabel("Polarity Score", fontsize=12)
plt.xlabel("Metrics", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.savefig("graph_without_legend.png")
plt.close() 


plt.figure(figsize=(6, 2))
handles, labels = barplot.get_legend_handles_labels()
plt.legend(handles, labels, title="Model", loc='center', frameon=False)
plt.axis('off')
plt.savefig("legend_only.png")
plt.close()