import numpy as np
from numpy.typing import NDArray
import re

class SimpleEmbedding:
    def __init__(self, dim: int=50):
        """Initialize a simple embedding model.

        Args:
            dim: Dimension of the embedding vectors
        """
        self.dim = dim
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.embeddings: NDArray[np.float64] | None = None
        self.vocab_size = 0

    def fit(self, sentences: list[list[str]], window_size: int=2):
        """Build a co-occurrence matrix and create simple embeddings.

        Args:
            sentences: List of tokenized sentences (lists of words)
            window_size: Context window size for co-occurrence
        """
        # Build vocabulary
        vocab: set[str] = set()
        for sentence in sentences:
            for word in sentence:
                vocab.add(word)

        # Assign IDs to words
        for i, word in enumerate(sorted(vocab)):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        self.vocab_size = len(vocab)

        # Build co-occurrence matrix
        cooc_matrix = np.zeros((self.vocab_size, self.vocab_size))

        for sentence in sentences:
            for i, center_word in enumerate(sentence):
                center_id = self.word_to_id[center_word]

                # Consider words within window_size
                context_start = max(0, i - window_size)
                context_end = min(len(sentence), i + window_size + 1)

                for j in range(context_start, context_end):
                    if i != j:  # Skip the center word itself
                        context_id = self.word_to_id[sentence[j]]
                        cooc_matrix[center_id, context_id] += 1

        # Apply SVD for dimensionality reduction
        U, _, _ = np.linalg.svd(cooc_matrix, full_matrices=False)
        self.embeddings = U[:, :self.dim]  # Take only the first 'dim' dimensions

    def get_vector(self, word: str) -> (None | NDArray[np.float64]):
        if self.embeddings is None:
            return None
        """Get the embedding vector for a word."""
        if word in self.word_to_id:
            return self.embeddings[self.word_to_id[word]]
        return None

    def similarity(self, word1: str, word2: str) -> (np.float64 | None):
        """Calculate cosine similarity between two words."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is None or vec2 is None:
            return None

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return dot_product / (norm1 * norm2)
    
    def find_similar_documents(self, query: str, nResults: int = 5) -> list[tuple[int, str ,np.float64]]: 
        similarities: list[tuple[int, str, np.float64]] = []
        for i, _ in enumerate(self.id_to_word):
            sim = self.similarity(query, self.id_to_word[i])
            if sim is not None:
                similarities.append((i,self.id_to_word[i], sim))
        return sorted(similarities, key=lambda x: float(x[2]), reverse=True)[:nResults]

    
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["a", "fox", "is", "an", "animal", "that", "lives", "in", "the", "forest"],
    ["the", "dog", "barks", "at", "the", "fox"],
    ["quick", "brown", "foxes", "are", "common", "in", "some", "forests"]
]

# Create and train our simple embedding model
simple_emb = SimpleEmbedding(dim=20)  # Using 10 dimensions for simplicity
simple_emb.fit(sentences)

# Get similarity between words
print(f"Similarity between 'fox' and 'dog': {simple_emb.similarity('fox', 'dog'):.4f}")
print(f"Similarity between 'fox' and 'forest': {simple_emb.similarity('fox', 'forest'):.4f}")
print(f"Similarity between 'quick' and 'brown': {simple_emb.similarity('quick', 'brown'):.4f}")

print(simple_emb.find_similar_documents("quick"))

def tokenize_document(text: str, max_tokens: int = 512) -> list[list[str]]:
    """
    Tokenize a document into a list of sentences, each with a limited number of tokens.

    Args:
        text: The input document as a single string.
        max_tokens: Maximum number of tokens per chunk

    Returns:
        A list of tokenized chunks (lists of words)
    """
    sentences: list[str] = re.split(r'[.!?]', text)
    tokens: list[list[str]] = []
    chunk: list[str] = []

    for sentence in sentences:
        words: list[str] = re.findall(r'\b\w+\b', sentence.lower())
        for word in words:
            chunk.append(word)
            if len(chunk) >= max_tokens:
                tokens.append(chunk)
                chunk = []

    if chunk:
        tokens.append(chunk)

    return tokens


# Example usage
text: str = """
The quick brown fox jumps over the lazy dog. A fox is an animal that lives in the forest.
The dog barks at the fox. Quick brown foxes are common in some forests.
"""

tokenized_chunks: list[list[str]] = tokenize_document(text, max_tokens=10)

print(tokenized_chunks)