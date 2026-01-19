import os
import numpy as np
import json

from .semantic_search import (
    SemanticSearch, 
    semantic_chunk,
    cosine_similarity
)

from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    load_movies
)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        
        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            score_data = {
                "chunk_idx": self.chunk_metadata[idx]["chunk_idx"],
                "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                "score": similarity_score
            }
            chunk_scores.append(score_data)

        movie_scores: dict[int, float] = {}
        for score in chunk_scores:
            movie_scores[score["movie_idx"]] = score["score"]
            
        # continue here...

def embed_chunks_command() -> np.ndarray:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    return searcher.load_or_create_chunk_embeddings(movies)
