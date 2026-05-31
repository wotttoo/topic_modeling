import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Literal, Union

class EmbeddingVectorizer:
    def __init__(
        self,
        model_name: str = 'intfloat/multilingual-e5-base',
        normalize: bool = True
    ):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
    
    def _format_inputs(
        self, 
        texts: List[str],
        mode: Literal['query', 'passage']
    )->List[str]:
        if mode not in {'query', 'passage'}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f'{mode}: {text.strip()}' for text in texts]
    
    def transform(self, texts: List[str], mode: Literal['query', 'passage'] = 'query'):
        if mode == 'raw':
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)
        
        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return embeddings.tolist()
    
    def transform_numpy(self, texts, mode: Literal['query', 'passage'] = 'query')->np.ndarray:
        return np.array(self.transform(texts, mode=mode))
    
def build_feature_matrices(X_train, X_test) -> dict:
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    embedding_vectorizer = EmbeddingVectorizer()
    X_train_embedding = embedding_vectorizer.transform(X_train)
    X_test_embedding = embedding_vectorizer.transform(X_test)

    # convert all to numpy arrays for consistency
    X_train_bow, X_test_bow = np.array(X_train_bow.toarray()), np.array(X_test_bow.toarray())
    X_train_tfidf, X_test_tfidf = np.array(X_train_tfidf.toarray()), np.array(X_test_tfidf.toarray())
    X_train_embedding, X_test_embedding = np.array(X_train_embedding), np.array(X_test_embedding)

    return{
        'bow': (X_train_bow, X_test_bow),
        'tfidf': (X_train_tfidf, X_test_tfidf),
        'embedding': (X_train_embedding, X_test_embedding)
    }