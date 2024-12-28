# Reddit-Post-Summarizer

This code is used to summarize the discussion found in the comment section of Reddit posts and get a general sentiment from all the users who typed in said comment section.

## Code Pipeline

1. **Scrape different multiplayer games from Steam**  
   To gather the subreddits of multiplayer games, I employed the following approach:
   - **Game Titles Extraction**: Used Selenium to automate the process of scraping game titles from Steam's website. This allowed us to efficiently retrieve a comprehensive list of multiplayer games.
   - **Subreddit Validation**: Checked if the scraped game titles had corresponding subreddits by querying the Reddit API. Only games with valid subreddits were included in the next step.

2. **Scrape posts and their comments from said subreddits**  
   Used the Reddit API to collect posts and their respective comments from the identified subreddits. This provided the raw textual data required for summarization and sentiment analysis.

3. **Summarize the posts using their comments**  
   Utilize `summarize.py` with a pipeline that incorporates the following tools:

   - **Named Entity Recognition (NER)**: The Tweebank-NLP NER model from the Twitter-Stanza pipeline was used to identify key entities such as people, organizations, and locations within Reddit comments. This model, trained on the Tweebank V2 dataset (English Twitter data), effectively handles the informal language often present in Reddit posts, allowing the pipeline to prioritize text containing these entities in the summary.

   - **K-Means Clustering**: Groups similar comments into clusters based on their semantic features. This ensures that the selected sentences for the summary represent a diverse range of viewpoints or topics discussed in the comments while avoiding redundancy.

   - **Word2Vec**: Provides word embeddings to capture relationships and similarities between words. This enhances the system’s ability to understand the contextual meaning of sentences, ensuring that relevant sentences with semantically similar phrasing are included in the summary.

   - **Sentiment Classification**: Assigns weights to comments based on their sentiment (e.g., negative comments are given higher priority as they often indicate issues or criticisms, followed by positive comments and then neutral ones). The weighted comments are then used to create an **extractive summary**.
  
   All these factors were used to assign weights to the comments and the top comments were chosen to create the extractive summary for their respective post.

4. **Assign a sentiment to the extractive summary**  
   Determine the overall sentiment (positive, negative, or neutral) of the generated summary, which provides insight into the general tone of the discussion.

   For the sentiment classification section of this project, I tested several models to determine the best fit for analyzing Reddit posts and comments. Below is a summary of the models used:

   ### 1. **TextBlob -- PatternAnalyzer**
   The first model tested was **PatternAnalyzer** from TextBlob, a lexicon-based sentiment classification model. It uses a predefined list of words and phrases labeled with specific emotions to classify text sentiment. This model served as a baseline for comparison with other, more advanced models. While it is lightweight and fast, its performance is limited when analyzing complex language, such as sarcasm or informal text, as it relies solely on word matching.
   
   ### 2. **DistilBERT -- distilbert-base-uncased-finetuned-sst-2-english**
   This transformer-based model is built on the DistilBERT architecture and fine-tuned on the SST-2 dataset, which contains movie reviews labeled as positive or negative. It provides a balance between simplicity and performance, achieving competitive accuracy (91.3%) while being smaller and faster than traditional BERT models.
   
   ### 3. **RoBERTa -- siebert/sentiment-roberta-large-english**
   This RoBERTa-large-based model is fine-tuned on 15 datasets for binary sentiment classification. It generalizes well across different text sources, including reviews and tweets, outperforming single-dataset models like DistilBERT, with an average accuracy of 93.2% compared to DistilBERT's 78.1%.
   
   ### 4. **Twitter-RoBERTa -- cardiffnlp/twitter-roberta-base-sentiment-latest**
   This RoBERTa-based model was pre-trained on 124 million tweets from 2018 to 2021. Using the TimeLMs framework, the model continuously updated its training data to capture evolving language trends. It was chosen for its ability to handle informal social media text and its similarity to Reddit language.
   
   ### 5. **Twitter-RoBERTa Fine-Tuned on Reddit Data**
   The final and most effective model was the **cardiffnlp/twitter-roberta-base-sentiment-latest** fine-tuned on a Reddit-specific dataset. This fine-tuning step enhanced the model's ability to classify sentiment in Reddit posts and comments by adapting it to Reddit-specific language, slang, and community-specific terms. This model outperformed all others in our evaluation, making it the best choice for our project.

---

This project integrates multiple tools and techniques to efficiently analyze Reddit discussions, extracting key points and sentiment trends while leveraging models like NER, K-Means, Word2Vec, and sentiment classifiers to deliver meaningful summaries.


