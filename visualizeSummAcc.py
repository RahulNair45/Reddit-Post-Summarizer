import matplotlib.pyplot as plt
import numpy as np

def plot_rouge_scores():
    models = [
        "Base Summarization with no K-means and Word2Vec",
        "distilbert",
        "cardiffnlp",
        "siebert",
        "Fine-Tuned RoBERTa Model"
    ]

    rouge_1_scores = [0.3839, 0.3742, 0.3718, 0.3742, 0.3711]
    rouge_2_scores = [0.2563, 0.1998, 0.1981, 0.1957, 0.1962]
    rouge_3_scores = [0.3213, 0.2833, 0.2807, 0.2827, 0.2815]

    x = np.arange(len(models)) 
    width = 0.2  

    fig, ax = plt.subplots(figsize=(11, 6)) 

    bar1 = ax.bar(x - width, rouge_1_scores, width, label="ROUGE-1", alpha=0.8)
    bar2 = ax.bar(x, rouge_2_scores, width, label="ROUGE-2", alpha=0.8)
    bar3 = ax.bar(x + width, rouge_3_scores, width, label="ROUGE-L", alpha=0.8)

    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("ROUGE Scores", fontsize=12)
    ax.set_title("ROUGE Score Comparison Between Models", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.legend()

    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontsize=8)

    output_file = "rouge_score_comparison_smaller_text2.png"
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Visualization saved to {output_file}")

plot_rouge_scores()