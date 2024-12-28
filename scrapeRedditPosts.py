import praw
import pandas as pd
from datetime import datetime
import json
from time import sleep
from collections import Counter
import re
from textblob import TextBlob
import spacy  # Add spaCy for NER
import numpy as np


class RedditGameScraper:
    def __init__(self, client_id, client_secret, user_agent):
        """Initialize the scraper with Reddit API credentials and spaCy NER model"""
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )

        self.nlp = spacy.load("en_core_web_sm")

        self.bug_keywords = [
            "bug",
            "glitch",
            "issue",
            "broken",
            "crash",
            "error",
            "problem",
            "not working",
            "buggy",
            "fix",
            "patch",
            "freeze",
            "stuck",
            "lag",
            "fps drop",
            "graphics issue",
            "performance issue",
            "low graphics",
            "low fps",
            "low performance",
            "low frame rate",
            "fix",
        ]

        # Platform-specific keywords
        self.platform_keywords = {
            "pc": [
                "pc",
                "steam",
                "epic",
                "desktop",
                "computer",
                "windows",
                "fps",
            ],
            "playstation": ["ps4", "ps5", "playstation", "psn"],
            "xbox": ["xbox", "series x", "series s", "xsx", "xss"],
            "switch": ["switch", "nintendo"],
        }

        self.severity_keywords = {
            "critical": [
                "crash",
                "unplayable",
                "broken",
                "game breaking",
                "save corrupted",
            ],
            "high": ["freeze", "stuck", "cant progress", "can't progress"],
            "medium": ["graphics", "audio", "visual", "ui"],
            "low": ["minor", "cosmetic", "typo", "text"],
        }

    def detect_platform(self, text):
        """Detect gaming platform from text"""
        if not text:
            return ["unspecified"]

        text = text.lower()
        detected_platforms = []

        for platform, keywords in self.platform_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_platforms.append(platform)

        return detected_platforms if detected_platforms else ["unspecified"]

    def detect_severity(self, text):
        """Detect bug severity from text"""
        if not text:
            return "unspecified"

        text = text.lower()

        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text for keyword in keywords):
                return severity

        return "unspecified"

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        if not text:
            return {"polarity": 0, "subjectivity": 0}

        analysis = TextBlob(text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity,
        }

    def extract_game_mentions(self, text):
        """Extract potential game titles using spaCy NER"""
        if not text:
            return []

        doc = self.nlp(text)
        game_titles = [
            ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "WORK_OF_ART"]
        ]
        return list(set(game_titles))

    def label_post(self, text):
        """Label posts based on content with predefined categories."""
        if not text:
            return "Uncategorized"

        text = text.lower()
        if any(keyword in text for keyword in self.bug_keywords):
            return "Bug"
        elif any(keyword in text for keyword in ["feature", "suggestion", "request"]):
            return "Feature Request"
        elif any(keyword in text for keyword in ["help", "question", "how", "what"]):
            return "Question"
        else:
            return "Other"

    def get_subreddit_posts(
        self,
        subreddit_name,
        post_limit=1000,
        time_filter="year",
        min_score=10,
        min_comments=5,
    ):
        """
        Scrape and label posts from a specified subreddit.
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []

            subreddit.id

            # Get posts from subreddit based on time filter
            for post in subreddit.top(time_filter=time_filter, limit=post_limit):
                try:
                    if post.score < min_score or post.num_comments < min_comments:
                        continue

                    # Process content and label it
                    full_text = f"{post.title} {post.selftext}"
                    post_label = self.label_post(full_text)

                    post_data = {
                        "post_id": post.id,
                        "title": post.title,
                        "text": post.selftext,
                        "label": post_label,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "author": str(post.author) if post.author else "[deleted]",
                        "subreddit": subreddit_name,
                        "timestamp": datetime.fromtimestamp(
                            post.created_utc
                        ).isoformat(),
                        "platform": self.detect_platform(full_text),
                        "severity": self.detect_severity(full_text),
                        "sentiment": self.analyze_sentiment(full_text),
                        "potential_games": self.extract_game_mentions(full_text),
                        "bug_keywords_found": [
                            kw for kw in self.bug_keywords if kw in full_text.lower()
                        ],
                        "is_resolved": any(
                            keyword in full_text.lower()
                            for keyword in ["resolved", "fixed", "solved"]
                        ),
                        "comments": self.process_comments(post),
                    }

                    posts_data.append(post_data)
                    sleep(1)

                except Exception as e:
                    print(f"Error processing post in r/{subreddit_name}: {str(e)}")
                    continue

            return posts_data

        except Exception as e:
            print(f"Error accessing r/{subreddit_name}: {str(e)}")
            return []

    def process_comments(self, post):
        """Process comments with advanced analysis"""
        processed_comments = []

        try:
            post.comments.replace_more(limit=0)
            for comment in post.comments[
                :100
            ]:  # need to prob limit to the most recent comments
                try:
                    # Skip deleted/removed comments
                    if not comment.author or comment.body in ["[deleted]", "[removed]"]:
                        continue

                    comment_data = {
                        "comment_id": comment.id,
                        "text": comment.body,
                        "score": comment.score,
                        "author": str(comment.author),
                        "timestamp": datetime.fromtimestamp(
                            comment.created_utc
                        ).isoformat(),
                        "sentiment": self.analyze_sentiment(comment.body),
                        "platform_mentioned": self.detect_platform(comment.body),
                        "has_solution": any(
                            keyword in comment.body.lower()
                            for keyword in ["solution", "fix", "solved", "resolved"]
                        ),
                        "is_op_response": str(comment.author) == str(post.author),
                    }

                    processed_comments.append(comment_data)
                except Exception as e:
                    print(f"Error processing comment: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error accessing comments: {str(e)}")

        return processed_comments

    def aggregate_sentiment_by_severity(self, posts_data):
        """Calculate average sentiment polarity by severity level"""
        severity_sentiments = {"critical": [], "high": [], "medium": [], "low": []}

        for post in posts_data:
            severity = post["severity"]
            polarity = post["sentiment"]["polarity"]
            if severity in severity_sentiments:
                severity_sentiments[severity].append(polarity)

        # Calculate average sentiment polarity per severity
        avg_sentiment = {
            severity: (sum(polarities) / len(polarities)) if polarities else 0
            for severity, polarities in severity_sentiments.items()
        }

        print("\nAverage Sentiment Polarity by Severity Level:")
        for severity, avg in avg_sentiment.items():
            print(f"{severity.capitalize()}: {avg:.2f}")

    def save_data(self, data, output_format="json", filename="game_bugs_data"):
        """Save scraped data in specified format with error handling"""
        try:
            if output_format.lower() == "json":
                with open(f"{filename}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

            elif output_format.lower() == "csv":
                flattened_data = []
                for post in data:
                    base_post = {
                        k: v
                        for k, v in post.items()
                        if k != "comments" and not isinstance(v, (dict, list))
                    }

                    # Handle lists and dicts
                    base_post["platforms"] = ",".join(post["platform"])
                    base_post["potential_games"] = ",".join(post["potential_games"])
                    base_post["bug_keywords"] = ",".join(post["bug_keywords_found"])
                    base_post["sentiment_polarity"] = post["sentiment"]["polarity"]
                    base_post["sentiment_subjectivity"] = post["sentiment"][
                        "subjectivity"
                    ]

                    flattened_data.append(base_post)

                df = pd.DataFrame(flattened_data)
                df.to_csv(f"{filename}.csv", index=False, encoding="utf-8")

            print(f"Data successfully saved as '{filename}.{output_format}'")

        except Exception as e:
            print(f"Error saving data: {str(e)}")
    
def create_subList(subreddsF, num_of_subs):
    import os  
    extracted_subreddits = []

    file_path = os.path.join(os.path.dirname(__file__), subreddsF)

    current_subs = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "r/" in line:
                    subreddit = line.split("r/")[-1].strip()
                    extracted_subreddits.append(subreddit)
                    current_subs += 1
                    if (current_subs >= num_of_subs):
                        break
    except FileNotFoundError:
        print(f"Error: File {subreddsF} not found in {file_path}")
        raise

    return extracted_subreddits


def main():
    # Initialize scraper
    scraper = RedditGameScraper(
        client_id="nst2HH1aa7Bbvc9igR3Vuw",
        client_secret="k8o-MuaK7vB2LbxtVeQ73uq8Qr92UA",
        user_agent="nlp final",
    )

    # List of gaming subreddits to scrape

    many_subs = create_subList("game_subreddits_clean.txt", 20)

    subreddits = [
        "gaming",
        "pcgaming",
        "GameBugs",
        "PS5",
        "XboxSeriesX",
        "NintendoSwitch",
        "PlayStation5",
        "PlayStation5Pro",
        "XboxSeriesXSeriesX",
        "XboxSeriesXSeriesS",
        "NintendoSwitch2",
        "Valorant",
        "VALORANT",
        "Diablo" "Gaming4Gamers",
        "blackops6",
        "CallOfDuty",
        "farmingsimulator25",
        "throneandliberty",
        "hoi4",
        "counterstrike2",
        "Warframe",
        "Enshrouded",
        "MicrosoftFlightSim",
        "apexlegends",
        "factorio",
        "BaldursGate3",
        "Astroneer",
        "helldivers2",
        "SpaceMarine_2",
        "EASportsFC",
        "DragonBallSparking",
        "rust",
        "deadbydaylight",
        "Grounded",
        "factorio",
        "RogueTraderCRPG",
        "destiny2",
        "MagicArena",
        "webfishing",
        "Eldenring",
        "PlanetCoaster2",
        "OnceHuman",
        "phasmophobia",
        "newworldgame",
        "NBA2K25",
        "FinalFantasyOnline",
        "GrandTheftAutoV",
        "TeamFortress2",
        "LostArk",
        "PUBATTLEGROUNDS",
        "blackdesert",
        "MonsterHunterWorld",
        "LiarsBar",
        "YuGiOhMasterDuel",
        "RainbowSixSiege",
        "TheElderScrollsOnline",
        "Vermintide",
        "DotA2",
        "AOW4",
        "overwatch2",
        "dayz",
        "MonsterHunterWilds",
        "TheFirstDescendant",
        "borderlands3",
        "satisfactory",
        "sixdaysinfallujah",
        "bodycam",
        "Stellaris",
        "CrusaderKings",
        "MaddenMobileForums",
        "StardewValley",
        "arma3",
        "WorldOfWarships",
        "ArkSurvivalAscended",
        "ReadyOrNotGame",
        "reddeadredemption2",
        "diabloiv",
        "RivalsOfAether",
        "americantruck",
        "TMNT",
        "LockdownProtocol",
        "ICARUS",
        "projectzomboid",
        "PlateUp",
        "GoldandBlack",
        "undisputedboxing",
        "MonsterHunter",
        "legoHorizonZeroDawn",
        "aoe2",
        "lethalcompany",
        "OvercookedGame",
        "ArmaReforger",
        "fo76",
        "SonsOfTheForest",
        "Breath_of_the_Wild",
        "PlayTemtem",
        "thefinals",
        "ghostoftsushima",
        "BlackMesa",
        "40kinquisitor",
        "HellLetLoose",
        "Palworld",
        "swtor",
        "Eldenring",
        "HalfLife",
        "RiskOfRain2",
        "Seaofthieves",
    ]

    subreddits += many_subs

    all_posts = []

    # Scrape each subreddit
    for subreddit in subreddits:
        print(f"Scraping r/{subreddit}...")
        posts = scraper.get_subreddit_posts(
            subreddit,
            post_limit=1000,
            time_filter="year",
            min_score=1,
            min_comments=1,
        )
        all_posts.extend(posts)
        print(f"Found {len(posts)} bug-related posts in r/{subreddit}")
        print(f"r/{subreddit} done")

    scraper.save_data(all_posts, "json")
    scraper.save_data(all_posts, "csv")

    total_posts = len(all_posts)
    total_comments = sum(len(post["comments"]) for post in all_posts)
    platforms_found = Counter(
        [platform for post in all_posts for platform in post["platform"]]
    )
    severity_counts = Counter(post["severity"] for post in all_posts)

    print("\nScraping Summary:")
    print(f"Total posts scraped: {total_posts}")
    print(f"Total comments processed: {total_comments}")
    print("\nPlatform distribution:")
    for platform, count in platforms_found.most_common():
        print(f"- {platform}: {count}")
    print("\nSeverity distribution:")
    for severity, count in severity_counts.most_common():
        print(f"- {severity}: {count}")

    scraper.aggregate_sentiment_by_severity(all_posts)


if __name__ == "__main__":
    main()