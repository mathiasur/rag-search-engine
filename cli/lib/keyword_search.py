from pathlib import Path

from .search_utils import DEFAULT_SEARCH_LIMIT, process_text


def search_command(query: str, index, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    if not Path("data/movies.json").exists():
        print("The movies.json file is not found")
        return
    
    if not Path("cache/index.pkl").exists():
        print("The index.pkl file is not found")
        return

    results = []
    index.load()
    query_tokens = process_text(query)
    
    for token in query_tokens:
        docs = index.get_documents(token)
        if docs is not None:
            for doc in docs:
                results.append(index.docmap[doc])

    return results
