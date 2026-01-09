import string
import json
import os

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read()
        stopwords = data.splitlines()
    return stopwords


def process_text(query: str) -> list[str]:
    query = query.lower()
    query = remove_punctuations(query)
    query_tokens = remove_stopwords(query)
    stemmed_query_tokens = [stem_token(token) for token in query_tokens]
    return stemmed_query_tokens


def remove_punctuations(query: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return query.translate(table)


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if token != "":
            cleaned_tokens.append(token)
    return cleaned_tokens


def remove_stopwords(query: str) -> list[str]:
    stopwords = load_stopwords()
    query_tokens = tokenize(query)
    new_query_tokens = []
    for token in query_tokens:
        if token not in stopwords:
            new_query_tokens.append(token)
    return new_query_tokens


def stem_token(token: str) -> str:
    stemmer = PorterStemmer()
    return stemmer.stem(token)
