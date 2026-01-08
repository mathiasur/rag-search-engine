import string
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        if find_token(query, movie["title"]):
            results.append(movie)
        if len(results) >= limit:
            break
    return results


def find_token(query: str, movie_title: str) -> bool:
    query_tokens = tokenize(query)
    movie_title = " ".join(tokenize(movie_title))
    for token in query_tokens:
        if token in movie_title:
            return True
    return False


def tokenize(query: str) -> list[str]:
    query = query.lower()
    query = remove_punctuations(query)
    query_tokens = remove_stopwords(query)
    stemmed_query_tokens = [stem_token(token) for token in query_tokens]
    return stemmed_query_tokens


def remove_punctuations(query: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return query.translate(table)


def remove_stopwords(query: str) -> list[str]:
    stopwords = load_stopwords()
    query_tokens = query.split()
    new_query_tokens = []
    for token in query_tokens:
        if token not in stopwords:
            new_query_tokens.append(token)
    return new_query_tokens


def stem_token(token: str) -> str:
    stemmer = PorterStemmer()
    return stemmer.stem(token)
