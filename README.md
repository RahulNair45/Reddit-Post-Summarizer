# Reddit-Post-Summarizer

This code is used to summarize the discussion found in the comment section of Reddit posts and get a general sentiment from all the users who typed in said comment section.

## Code Pipeline

1. **Scrape different multiplayer games from Steam**  
   Gather the respective subreddits of these games.

2. **Scrape posts and their comments from said subreddits**  
   Collect data from Reddit posts and their corresponding comment sections.

3. **Summarize the posts using their comments**  
   Utilize `summarize.py` with a pipeline that incorporates tools like:
   - **K-Means Clustering**: Groups similar comments together.
   - **Word2Vec**: Provides word embeddings for better context understanding.
   - **Sentiment Classification**: Assigns weights to comments based on their sentiment.
   This pipeline selects the best comments to create an **extractive summary** of the post.

4. **Assign a sentiment to the extractive summary**  
   Determine the general user sentiment of the post and its discussion based on the comment section.

---

This project provides an efficient way to analyze Reddit discussions and quickly extract key points and sentiment trends.

