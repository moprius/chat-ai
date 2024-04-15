import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from .chatbot import Chatbot

class ChatbotGUI(QMainWindow):
    def __init__(self, chatbot):
        super().__init__()
        self.chatbot = chatbot
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Chatbot")
        self.setGeometry(100, 100, 800, 600)

        self.textEdit = QTextEdit(self)
        self.textEdit.setReadOnly(True)

        self.inputEdit = QTextEdit(self)
        self.inputEdit.setFixedHeight(100)

        self.sendButton = QPushButton('Enviar', self)
        self.sendButton.clicked.connect(self.sendQuestion)

        layout = QVBoxLayout()
        layout.addWidget(self.textEdit)
        layout.addWidget(self.inputEdit)
        layout.addWidget(self.sendButton)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def sendQuestion(self):
        question = self.inputEdit.toPlainText().strip()
        self.inputEdit.clear()

        if question.lower() == 'sair':
            self.close()

        self.textEdit.append(f"Você: {question}")
        answer = self.chatbot.ask(question)
        self.textEdit.append(f"Chatbot: {answer}\n")

def run():
    chatbot = Chatbot()
    app = QApplication(sys.argv)
    gui = ChatbotGUI(chatbot)
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
