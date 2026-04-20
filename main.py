import os
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich import print

from dotenv import load_dotenv

load_dotenv()

VECTOR_DB = os.path.join(os.path.dirname(__file__), 'db')

prompt_template = """
    Responda a pergunta do usuário: 
    {question}
    com base nessas informações:
    {context}
    Se voce não encontrar a resposta para a pergunta do usuário 
    nessas informações, diga que não sabe.
"""

def main():
    
    # Recebe a pergunta do usuário
    question = input("Digite sua pergunta: ")

    # Carrega o banco de dados vetorial Chroma e realiza a busca por similaridade
    db = Chroma(
        persist_directory=VECTOR_DB, 
        embedding_function=OpenAIEmbeddings()
    )

    # Realiza a busca por similaridade e obtém os resultados com suas respectivas pontuações de relevância
    results = db.similarity_search_with_relevance_scores(question, k=3)
    
    # Verifica se há resultados e se a relevância é suficiente para responder à pergunta do usuário
    if len(results) == 0 or results[0][1] < 0.7: 
        print("Nenhuma informação relevante encontrada para responder à pergunta.")
        return

    # Extrai os textos dos resultados e os concatena para formar o contexto a ser usado na resposta
    texts_results = []
    for result in results:
        text = result[0].page_content
        texts_results.append(text)

    # Concatena os textos dos resultados para formar o contexto a ser usado na resposta
    context = "\n\n----\n\n".join(texts_results)
    
    # Cria o prompt para o modelo de linguagem natural (LLM) usando o template definido e os dados da pergunta e do contexto
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"question": question, "context": context})
    model = ChatOpenAI()
    
    # Invoca o modelo de linguagem natural (LLM) com o prompt criado e obtém a resposta gerada
    response = model.invoke(prompt).content
    
    print(response)
    
    
if __name__ == "__main__":
    main()
