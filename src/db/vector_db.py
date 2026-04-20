
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'assets')

def loading_documents():
    """
    Carrega todos os PDFs da pasta assets e retorna 
    uma lista de documentos.

    Returns:
        list: Lista de documentos carregados.
    """    
    loader = PyPDFDirectoryLoader(ASSETS_DIR, glob="*.pdf")
    documents = loader.load()
    return documents

def split_documents_into_chunks(documents) -> list:
    """
    Divide os documentos em pedaços menores (chunks) 
    para facilitar o processamento.

    Args:
        documents (list): Lista de documentos a serem divididos.

    Returns:
        list: Lista de chunks resultantes da divisão dos documentos.
    """    
    documents_separator = RecursiveCharacterTextSplitter(
        chunk_size=2000, # Tamanho máximo de cada chunk (pedaço) em caracteres
        chunk_overlap=500, # Quantidade de caracteres que se sobrepõem entre os chunks para manter o contexto
        length_function=len, # Função para calcular o comprimento do texto (neste caso, usando a função len)
        add_start_index=True # Adiciona o índice inicial de cada chunk
    )
    chunks = documents_separator.split_documents(documents)
    print(f"Total de chunks criados: {len(chunks)}")
    return chunks

def vectorize_chunks(chunks):
    """
    Vetoriza os chunks usando um modelo de linguagem natural (LLM) 
    e armazena no banco de dados vetorial Chroma.

    Args:
        chunks (list): Lista de chunks a serem vetorizados e armazenados.
    """    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory="db"
    )
    print("Chunks vetorizados e armazenados no banco de dados vetorial.")

def create_vector_db():
    documents = loading_documents()
    
    chunks = split_documents_into_chunks(documents)

    vectorize_chunks(chunks)
    

create_vector_db()