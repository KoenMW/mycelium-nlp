import os
import json
from typing import List, Tuple, Set
import numpy as np
from numpy.typing import NDArray
from src.embeddings import Embedding
from src.preprocessing import load_and_clean_documents

class VectorDB:
    def __init__(self, filepath: str = "./data/vector_store/vectordb.json") -> None:
        self.filepath: str = filepath
        self.data: List[Tuple[str, NDArray[np.float64]]] = []
        self.existing_texts: Set[str] = set()
        self.embedding: Embedding = Embedding()
        self._load()
        docs = load_and_clean_documents("./data/raw")
        self.add_documents(docs)

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                raw_data: List[dict] = json.load(f)
                self.data = [(item["text"], np.array(item["vector"], dtype=np.float64)) for item in raw_data]
                self.existing_texts = {item["text"] for item in raw_data}
                if self.data:
                    self.embedding.fit([text for text, _ in self.data])

    def _save(self) -> None:
        with open(self.filepath, "w") as f:
            json.dump([
                {"text": text, "vector": vector.tolist()} for text, vector in self.data
            ], f)

    def add_documents(self, texts: List[str]) -> None:
        new_texts = [text for text in texts if text not in self.existing_texts]

        if not new_texts:
            return

        if not self.embedding.fitted:
            self.embedding.fit(new_texts)

        for text in new_texts:
            vector: NDArray[np.float64] = self.embedding.transform(text)
            self.data.append((text, vector))
            self.existing_texts.add(text)
        self._save()

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vec: NDArray[np.float64] = self.embedding.transform(query)
        similarities: List[Tuple[str, float]] = [
            (text, self._cosine_similarity(query_vec, vec)) for text, vec in self.data
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> float:
        dot_product: float = float(np.dot(vec1, vec2))
        norm1: float = float(np.linalg.norm(vec1))
        norm2: float = float(np.linalg.norm(vec2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0  # No similarity if either vector is zero vector


        return dot_product / (norm1 * norm2)