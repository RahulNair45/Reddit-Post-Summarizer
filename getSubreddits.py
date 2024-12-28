import praw
import re
import time

reddit = praw.Reddit(
    client_id="vbFzRNrXG0BqdyPWHfImTw",
    client_secret="rQ-bxF-Lr4vIuHyQOxnO5w4B8KR29w",
    user_agent="script:reddit-game-search:v1.0 (by u/xxProReapsxx35777)"
)

# input_file = "multiplayer_games_limited.txt"
input_file = "multiplayer_games_max.txt"
output_file = "game_multi_max_reddits.txt"

with open(input_file, 'r', encoding='utf-8') as file:
    game_titles = [line.strip() for line in file.readlines()]

def preprocess_title(title):
    # Remove special characters, normalize spaces, and lowercase
    title = re.sub(r'[^a-zA-Z0-9 ]', '', title)
    return title.strip().lower()

# Step 3: Search for subreddits and save results
with open(output_file, 'w', encoding='utf-8') as out_file:
    for game_title in game_titles:
        preprocessed_title = preprocess_title(game_title)
        print(f"\nSearching for subreddit related to: {game_title}")

        search_results = list(reddit.subreddits.search(query=preprocessed_title, limit=1))
        
        if search_results:
            # Get the first match
            first_subreddit = search_results[0]
            out_file.write(f"{game_title}: r/{first_subreddit.display_name}\n")
        else:
            out_file.write(f"{game_title}: No subreddit found\n")
        
        time.sleep(1)