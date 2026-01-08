import string
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        if find_word(query, movie["title"]):
            results.append(movie)
        if len(results) >= limit:
            break
    return results


def find_word(query: str, movie_title: str) -> bool:
    query_words = process_text(query)
    movie_title = " ".join(process_text(movie_title))
    for word in query_words:
        if word in movie_title:
            return True
    return False


def process_text(query: str) -> list[str]:
    query = query.lower()
    query = remove_punctuations(query)
    query_words = remove_stopwords(query)
    stemmed_query_words = [stem_word(word) for word in query_words]
    return stemmed_query_words


def remove_punctuations(query: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return query.translate(table)


def remove_stopwords(query: str) -> list[str]:
    stopwords = load_stopwords()
    query_words = query.split()
    new_query_words = []
    for word in query_words:
        if word not in stopwords:
            new_query_words.append(word)
    return new_query_words


def stem_word(word: str) -> str:
    stemmer = PorterStemmer()
    return stemmer.stem(word)
