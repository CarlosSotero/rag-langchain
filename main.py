# Importa o banco vetorial Chroma para armazenar os embeddings
from langchain_chroma import Chroma
# Importa o modelo de embeddings da HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
#
from langchain.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
# Importa função para carregar variáveis do arquivo .env
from dotenv import load_dotenv

import os

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pegando a chave da API
chave_openrouter = os.getenv("OPENROUTER_API_KEY")

CAMINHO_DB = 'C:/Users/CarlosSotero/OneDrive/Documentos/Python Scripts/Curso Cientista de Dados - DNC/Desafio DNC -Desenvolvendo uma IA Generativa para a Oscar/LangChain Projeto RAG/db'

prompt_template = """Você é um assistente inteligente que ajuda os usuários a encontrar informações em documentos PDF.
Você receberá uma pergunta do usuário: {pergunta} 
Deve buscar a resposta com base nessa informações: {base_conhecimento}"""

def perguntar():
    pergunta = input("Digite sua pergunta: ")

    # Carregando o banco de dados de vetores
    funcao_embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        # Normaliza os embeddings para melhorar a busca
        encode_kwargs={'normalize_embeddings': True})
    # Carrega o banco de dados de vetores Chroma
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embeddings)

    # Compara a pergunta do usuário (embedding) com os documentos no banco de dados
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=5)

    # If para verificar se encontrou resultados relevantes
    if len(resultados) == 0 or resultados[0][1] < 0.5:
        print("Desculpe, não consegui encontrar uma resposta para sua pergunta.")
        return
    
    # Cria uma lista para armazenar os textos dos documentos retornados pela busca
    textos_resultados = []
    # Itera sobre os resultados da busca semântica
    for resultado in resultados:
        # Extrai o conteúdo textual de cada documento
        texto = resultado[0].page_content
        # Adiciona o texto à lista
        textos_resultados.append(texto)

    # Separa os textos com um delimitador para formar a base de conhecimento
    base_conhecimento = "\n\n----\n\n".join(textos_resultados)
    # Cria um template de prompt para o modelo de linguagem
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # Preenche o template com os dados reais: a pergunta do usuário e a base de conhecimento
    prompt = prompt.invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
        })
    
    # Modelo Mixtral via OpenRouter
    modelo = ChatOpenAI(
        api_key=chave_openrouter,
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/mixtral-8x7b-instruct"
        )
    texto_resposta = modelo.invoke(prompt).content
    print(f"Resposta: {texto_resposta}")

perguntar()