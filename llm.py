import openai
from dotenv import load_dotenv
import os
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
import uuid
import tiktoken
from PyPDF2 import PdfReader
from docx import Document
from tempfile import TemporaryDirectory

OPENAI_API_KEY = 'OPENAI_API_KEY'
OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'
OPENAI_MODEL = 'gpt-4o-mini'

def gen_client() -> openai.Client:
    # Load the API key from the .env file
    load_dotenv()
    api_key = os.getenv(OPENAI_API_KEY)

    # Create an instance of the OpenAI client
    client = openai.Client(api_key=api_key)
    return client

class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding(OPENAI_MODEL)

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i + chunk_size >= len(tokens):
                break
        return chunks

class EmbeddingModel(EmbeddingFunction):
    def __init__(self):
        self.client = gen_client()
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=input
        )
        return [embedding.embedding for embedding in response.data]

class VectorDB:
    def __init__(self):
        self.temp_dir = TemporaryDirectory()
        persist_directory = self.temp_dir.name
        self.client = Client(Settings(persist_directory=persist_directory))
        self.embedding_model = EmbeddingModel()
        self.collection = self.client.create_collection(name="text_collection", embedding_function=self.embedding_model)

    def add_texts(self, texts: list[str]):
        embeddings = self.embedding_model(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        for text, embedding, i in zip(texts, embeddings, ids):
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[i],
            )

    def query_texts(self, query: str, top_k=5) -> list[str]:
        embedding = self.embedding_model([query])[0]
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return [doc for doc in results['documents'][0]]


class LLM:
    def __init__(self):
        self.client = gen_client()
        self.history = []

    def __call__(self, messages: list[dict], vector_db: VectorDB, query: str, top_k: int = 5) -> str:
        # Perform RAG: Query the vector database to get relevant documents
        relevant_texts = vector_db.query_texts(query, top_k)
        context = "\n".join(relevant_texts)
        
        # Append the context to the input messages
        messages_with_context = messages + [{"role": "system", "content": context}]
        
        # Get the response from the model
        response = self.client.chat_completions.create(
            model=OPENAI_MODEL,
            messages=messages_with_context
        )
        bot_reply = response.choices[0].message['content']
        return bot_reply


def read_txt(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def read_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def process_and_add_files(vector_db: VectorDB, file_paths: list[str]):
    texts = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            text = read_txt(file_path)
        elif file_path.endswith('.pdf'):
            text = read_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = read_docx(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue
        texts.append(text)
    vector_db.add_texts(texts)


def main():
    vector_db = VectorDB()
    
    # Example texts to add
    texts = ["This is the first text.", "This is the second text.", "This is the third text."]
    vector_db.add_texts(texts)
    
    # Query the database
    query = "first text"
    results = vector_db.query_texts(query)
    print("Query results:", results)


if __name__ == "__main__":
    main()
