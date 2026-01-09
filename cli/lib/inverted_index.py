import pickle

from pathlib import Path

from .search_utils import process_text
from .search_utils import load_movies


class InvertedIndex:
    index = {}  # map tokens to sets of documents IDs
    docmap = {}  # map document IDs to their full document objects

    def __add_document(self, doc_id, text):
        tokens = process_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        if term not in self.index:
            return None
        return sorted(self.index[term])

    def build(self):
        movies = load_movies()
        for movie in movies:
            input_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], input_text)
            self.docmap[movie["id"]] = movie

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        index_file_path = cache_dir / "index.pkl"
        with open(index_file_path, "wb") as f:
            pickle.dump(self.index, f)

        docmap_file_path = cache_dir / "docmap.pkl"
        with open(docmap_file_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        if not Path("cache/index.pkl").exists():
            print("The index.pkl file is not found")
            return
        
        if not Path("cache/docmap.pkl").exists():
            print("The docmap.pkl file is not found")
            return
        
        with open("index.pkl", "rb") as f:
            self.index = pickle.load(f)
            
        with open("docmap.pkl", "rb") as f:
            self.index = pickle.load(f)
