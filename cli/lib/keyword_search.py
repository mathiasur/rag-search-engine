from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, process_text


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    print(process_text(query))
    for movie in movies:
        if find_token(query, movie["title"]):
            results.append(movie)
        if len(results) >= limit:
            break
    return results


def find_token(query: str, movie_title: str) -> bool:
    processed_query = process_text(query)
    processed_title = " ".join(process_text(movie_title))
    for token in processed_query:
        if token in processed_title:
            return True
    return False
