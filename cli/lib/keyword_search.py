import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        if process_text(query) in process_text(movie["title"]):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def process_text(query: str) -> str:
    query = query.lower()
    query = remove_punctuations(query)
    return query


def remove_punctuations(query: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return query.translate(table)
