from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.docs = documents
        self.doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.index.add(np.array(self.doc_embeddings))

    def retrieve(self, query, k=3):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, k)
        return [self.docs[i] for i in I[0]]
