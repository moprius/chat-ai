from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Indexer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def build_index(self, documents):
        """
        Constrói o índice TF-IDF para os documentos fornecidos.

        :param documents: Lista de documentos (strings) a serem indexados.
        """
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

    def search(self, query, top_n=5):
        """
        Realiza uma busca nos documentos indexados, encontrando os top_n documentos mais relevantes para a consulta.

        :param query: String de consulta para a qual a busca é realizada.
        :param top_n: Número de resultados mais relevantes para retornar.
        :return: Índices dos documentos mais relevantes e seus respectivos scores.
        """
        # Transforma a query em um vetor TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([query])

        # Calcula a similaridade de cosseno entre a query e os documentos
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        # Obtém os índices dos documentos mais relevantes
        relevant_indices = np.argsort(-cosine_similarities)[:top_n]

        # Retorna os índices e os scores dos documentos relevantes
        return [(index, cosine_similarities[index]) for index in relevant_indices]
