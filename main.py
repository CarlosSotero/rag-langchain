# Importa o banco vetorial Chroma para armazenar os embeddings
from langchain_chroma import Chroma
# Importa o modelo de embeddings da HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
# Importa função para carregar variáveis do arquivo .env
from dotenv import load_dotenv
from uploud_novos_arquivos import processar_pdfs

import os

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pegando a chave da API
chave_openrouter = os.getenv("OPENROUTER_API_KEY")

CAMINHO_DB = 'C:/Users/CarlosSotero/OneDrive/Documentos/Python Scripts/Curso Cientista de Dados - DNC/Desafio DNC -Desenvolvendo uma IA Generativa para a Oscar/LangChain Projeto RAG/db'

# Título e input do STREAMLIT
st.set_page_config(page_title="Assistente RAG", layout="wide")
st.title("🔍 Assistente de Perguntas sobre PDFs com RAG")
abas = st.tabs(["Consultar Base", "Atualizar Base"])

# Aba de consulta
with abas[0]:
    pergunta = st.text_input("Digite sua pergunta:")
    botao = st.button("Consultar")


    # Função para consulta
    def responder(pergunta):
        # Criação do objeto de embeddings (paraphrase-multilingual-mpnet-base-v2)
        funcao_embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            encode_kwargs={'normalize_embeddings': True}
        )

        # Carrega o banco de vetores
        db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embeddings)

        # Busca semântica no vetor
        resultados = db.similarity_search_with_relevance_scores(pergunta, k=5)

        # Verificação de resultados relevantes
        if len(resultados) == 0 or resultados[0][1] < 0.5:
            return "Desculpe, não encontrei resposta para essa pergunta."

        # Junta os textos retornados
        textos_resultados = [r[0].page_content for r in resultados]
        base_conhecimento = "\n\n----\n\n".join(textos_resultados)

        # Prompt para o modelo
        prompt_template = """
        Você é um assistente inteligente que ajuda os usuários a encontrar informações em documentos PDF.
        Você receberá uma pergunta do usuário: {pergunta} 
        Deve buscar a resposta com base nessas informações: {base_conhecimento}

        Se não souber, diga que não sabe responder.
        """
        # Cria o template do prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt.invoke({
            "pergunta": pergunta,
            "base_conhecimento": base_conhecimento
        })

        # Chama o modelo Mixtral via OpenRouter
        modelo = ChatOpenAI(
            api_key=chave_openrouter,
            base_url="https://openrouter.ai/api/v1",
            model="mistralai/mixtral-8x7b-instruct"
        )
        resposta = modelo.invoke(prompt).content
        return resposta

    #  Resposta no Streamlit
    if botao and pergunta:
        with st.spinner("Consultando documentos..."):
            resposta = responder(pergunta)
            st.markdown("### 💬 Resposta da IA")
            st.write(resposta)

# Aba de atualização
with abas[1]:
    arquivos = st.file_uploader("Faça upload de novos PDFs", type="pdf", accept_multiple_files=True)
    if arquivos:
        pasta_temp = "uploads_temp"
        os.makedirs(pasta_temp, exist_ok=True)

        for file in arquivos:
            caminho_arquivo = os.path.join(pasta_temp, file.name)
            with open(caminho_arquivo, "wb") as f:
                f.write(file.read())

        if st.button("Adicionar ao Banco de Dados"):
            qtd_chunks = processar_pdfs(pasta_temp, CAMINHO_DB)
            st.success(f"{qtd_chunks} chunks adicionados ao banco com sucesso!")
