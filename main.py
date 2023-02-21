import datetime
from pathlib import Path

import pytz
import tweepy
import openai
from decouple import config


BASE_DIR = Path(__file__).resolve()
config.search_path = BASE_DIR


# UTILS
def get_current_time():
    return datetime.datetime.now(pytz.timezone("Europe/Warsaw"))


def convert_time_str(time: datetime.datetime) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


# TWITTER

username = config("TWEET_USERNAME", cast=str)

api_key = config("TWEET_API_KEY", cast=str)
secret_key = config("TWEET_SECRET_KEY", cast=str)
bearer_token = config("TWEET_BEARER_TOKEN", cast=str)

access_token = config("TWEET_ACCESS_TOKEN", cast=str)
access_token_secret = config("TWEET_ACCESS_TOKEN_SECRET", cast=str)


twitter_client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=secret_key,
    access_token=access_token,
    access_token_secret=access_token_secret,
)


def get_timeline():
    timeline = twitter_client.timeline()
    return timeline


def get_home_timeline(last_time: str, max_results=60):
    return twitter_client.get_home_timeline(
        user_auth=True, end_time=last_time, max_results=max_results, user_fields="username"
    )


def get_me():
    return twitter_client.get_user(username="{username}")


# OPENAI
openai.api_key = config("OPENAI_API_KEY", cast=str)

# Set up the model and prompt
model_engine = "text-davinci-003"


def _clean_results(results) -> list:
    results = results.split("\n")
    results = [result.strip().lower().split(".") for result in results]
    cleaned_results = []
    for result in results:
        if len(result) > 1:
            if result[0].isdigit():
                cleaned_results.insert(int(result[0]) - 1, result[1])
    return cleaned_results

def classify_tweets(tweet_texts: list) -> list:
    tweets = ""
    c = 1
    for tweet in tweet_texts:
        tweets += f"{c}. {tweet}\n"
        c += 1

    prompt = f"Classify the sentiment in these tweet: \n {tweets}. Return as python list"

    completion = openai.Completion.create(
        model=model_engine,
        prompt=prompt,
        temperature=0,
        max_tokens=512,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    results = completion.choices[0].text
    return _clean_results(results)



if __name__ == "__main__":
    last_time = get_current_time() - datetime.timedelta(minutes=60)
    tweets = get_home_timeline(last_time=convert_time_str(last_time))

    total_tweets = len(tweets.data)
    print(f"Total tweets: {total_tweets}")

    sentiments = {}

    tweet_texts = [tweet.text for tweet in tweets.data]

    response = classify_tweets(tweet_texts)

    print(response)

    for i, tweet in enumerate(tweet_texts):
        print(f"Tweet: {tweet}")
        sentiment = response[i]
        print(f"Sentiment: {sentiment}")
        sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

    print("=====================================")
    print("Summary of last hour:")
    for sentiment, count in sentiments.items():
        print(f"{sentiment}: {count}")
