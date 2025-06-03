from typing import Iterable
import numpy as np
import os
import pickle
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedding:
    def __init__(self, vectorizer_path: str = "./data/vector_store/vectorizer.pkl") -> None:
        self.vectorizer_path = vectorizer_path
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.fitted: bool = False

        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
                self.fitted = True

    def fit(self, documents: Iterable[str]) -> None:
        self.vectorizer.fit(documents)
        self.fitted = True
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def transform(self, text: str) -> NDArray[np.float64]:
        if not self.fitted:
            raise ValueError("The vectorizer has not been fitted.")
        sparse_vec = self.vectorizer.transform([text])
        return csr_matrix(sparse_vec).toarray()[0]