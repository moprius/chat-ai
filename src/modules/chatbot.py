import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from .text_processor import TextProcessor
from .indexer import Indexer

class Chatbot:
    def __init__(self, data_path='data'):
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path)
        self.text_processor = TextProcessor()
        self.indexer = Indexer()
        self.documents = []
        self._download_nltk_resources()
        self._load_and_process_texts()

    def _download_nltk_resources(self):

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def _load_and_process_texts(self):

        for filename in os.listdir(self.data_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    processed_text = self.text_processor.process(text)
                    self.documents.append(processed_text)
        self.indexer.build_index(self.documents)

    def extract_relevant_sentences(self, text, question, max_sentences=2):

        sentences = sent_tokenize(text)
        question_vector = self.indexer.tfidf_vectorizer.transform([question])
        sentence_vectors = self.indexer.tfidf_vectorizer.transform(sentences)

        relevances = cosine_similarity(question_vector, sentence_vectors).flatten()
        relevant_indices = relevances.argsort()[-max_sentences:]

        relevant_sentences = ' '.join([sentences[index] for index in sorted(relevant_indices)])
        return relevant_sentences

    def ask(self, question):

        processed_question = self.text_processor.process(question)
        results = self.indexer.search(processed_question)

        if results:
            best_result_index = results[0][0]
            best_result_text = self.documents[best_result_index]
            return self.extract_relevant_sentences(best_result_text, processed_question)
        else:
            return "Desculpe, não consigo encontrar uma resposta para isso."

    def run(self):

        print("Olá! Sou um chatbot treinado em seus textos. Criado por Brainium. Pergunte-me algo!")
        while True:
            question = input("Você: ")
            if question.lower() == 'sair':
                print("Até mais!")
                break
            answer = self.ask(question)
            print("Chatbot:", answer)
