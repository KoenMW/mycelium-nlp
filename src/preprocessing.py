import os
import re
import string
from typing import List, Optional
from docx import Document 

class DocumentProcessor:
    def __init__(self, stopwords: Optional[List[str]] = None) -> None:
        self.stopwords: List[str] = stopwords if stopwords else []

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return [word for word in text.split() if word not in self.stopwords]

    def preprocess(self, text: str) -> str:
        return " ".join(self.tokenize(self.clean_text(text)))

    def chunk_text(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        words: List[str] = text.split()
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap.")
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks

# Script to load Word documents and preprocess them
def load_and_clean_documents(doc_folder: str) -> List[str]:
    processor = DocumentProcessor()
    all_chunks: List[str] = []

    for filename in os.listdir(doc_folder):
        filepath = os.path.join(doc_folder, filename)
        if filename.endswith(".docx"):
            document = Document(filepath)
            full_text = " ".join(para.text for para in document.paragraphs)
        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            # Skip unsupported file types
            print("unsupported document: ", filename)
            continue

        cleaned = processor.preprocess(full_text)
        chunks = processor.chunk_text(cleaned, chunk_size=100, overlap=20)
        all_chunks.extend(chunks)

    return all_chunks