from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedding:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.fitted: bool = False

    def fit(self, documents: Iterable[str]) -> None:
        self.vectorizer.fit(documents)
        self.fitted = True

    def transform(self, text: str) -> NDArray[np.float64]:
        if not self.fitted:
            raise ValueError("The vectorizer has not been fitted.")
        sparse_vec = self.vectorizer.transform([text])
        return csr_matrix(sparse_vec).toarray()[0]