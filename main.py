from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from uploud_novos_arquivos import processar_pdfs
import os

# Carrega variáveis de ambiente
load_dotenv()

chave_openrouter = os.getenv("OPENROUTER_API_KEY")
CAMINHO_DB = "faiss_index"

# Configuração da interface
st.set_page_config(page_title="Assistente RAG", layout="wide")
st.title("🔍 Assistente de Perguntas sobre PDFs com RAG")
abas = st.tabs(["Consultar Base", "Criar / Atualizar Base"])

# Aba 1 - Consulta
with abas[0]:
    pergunta = st.text_input("Digite sua pergunta:")
    botao = st.button("Consultar")

    def responder(pergunta):
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            encode_kwargs={'normalize_embeddings': True}
        )

        if not os.path.exists(CAMINHO_DB):
            return "O banco de dados ainda não foi criado."

        db = FAISS.load_local(CAMINHO_DB, embeddings, allow_dangerous_deserialization=True)
        resultados = db.similarity_search_with_relevance_scores(pergunta, k=5)

        if len(resultados) == 0 or resultados[0][1] < 0.5:
            return "Desculpe, não encontrei resposta para essa pergunta."

        textos_resultados = [r[0].page_content for r in resultados]
        base_conhecimento = "\n\n----\n\n".join(textos_resultados)

        prompt_template = """
        Você é um assistente inteligente que ajuda os usuários a encontrar informações em documentos PDF.
        Você receberá uma pergunta do usuário: {pergunta} 
        Deve buscar a resposta com base nessas informações: {base_conhecimento}

        Se não souber, diga que não sabe responder.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        prompt = prompt.invoke({
            "pergunta": pergunta,
            "base_conhecimento": base_conhecimento
        })

        modelo = ChatOpenAI(
            api_key=chave_openrouter,
            base_url="https://openrouter.ai/api/v1",
            model="mistralai/mixtral-8x7b-instruct"
        )
        return modelo.invoke(prompt).content

    if botao and pergunta:
        with st.spinner("Consultando documentos..."):
            resposta = responder(pergunta)
            st.markdown("### 💬 Resposta da IA")
            st.write(resposta)

# Aba 2 - Atualização da base
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
