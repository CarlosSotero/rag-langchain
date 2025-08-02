# Importa o loader de PDFs em diretório
from langchain_community.document_loaders import PyPDFDirectoryLoader
# Importa o splitter para dividir os documentos em pedaços menores
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importa o banco vetorial Chroma para armazenar os embeddings
from langchain_community.vectorstores import Chroma
# Importa o modelo de embeddings da HuggingFace
from langchain.embeddings import HuggingFaceEmbeddings
# Importa função para carregar variáveis do arquivo .env
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

PASTA_BASE = 'C:/Users/CarlosSotero/OneDrive/Documentos/Python Scripts/Curso Cientista de Dados - DNC/Desafio DNC -Desenvolvendo uma IA Generativa para a Oscar/LangChain Projeto RAG/base'

# Função para criar o banco de dados de vetores
def criar_db():
  documentos = carregar_documentos()
  print(f'Esse é o ducumento: {documentos}')
  chunks = dividir_documentos_em_chunks(documentos)
  vetorizar_chunks(chunks)

# Função para carregar documentos
def carregar_documentos():
  # Não precisa do argumento 'glob' se você quiser carregar todos os PDFs na pasta
  # .pdf já é o padrão
  carregador = PyPDFDirectoryLoader(PASTA_BASE)
  # Carrega os documentos da pasta base
  documentos = carregador.load()
  return documentos

# Divide os documentos em chunks
def dividir_documentos_em_chunks(documentos):
  separador_documentos = RecursiveCharacterTextSplitter(
      chunk_size=2000,  # Tamanho do chunk em caracteres
      chunk_overlap=500,  # Sobreposição entre chunks para preservar contexto entre partes
    # A sobreposição ajuda a manter frases ou ideias que estão no final de um chunk
    # e que podem ser importantes para o início do próximo. Isso melhora a qualidade
    # da busca semântica e evita perda de informação.
      length_function=len,  # Função para calcular o tamanho do chunk 
      add_start_index=True  # Adiciona o índice de início do chunk
)
  # Divide os documentos em chunks
  chunks = separador_documentos.split_documents(documentos)
  print(len(chunks), 'chunks criados')
  return chunks

def vetorizar_chunks(chunks):
       
    # Cria o vetor de embeddings usando o modelo 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Cria o banco de dados de vetores usando Chroma
    db = Chroma.from_documents(chunks, embeddings, persist_directory='C:/Users/CarlosSotero/OneDrive/Documentos/Python Scripts/Curso Cientista de Dados - DNC/Desafio DNC -Desenvolvendo uma IA Generativa para a Oscar/LangChain Projeto RAG/db')
    
    # Persistindo o banco de dados
    db.persist()
    print('Banco de dados criado com sucesso!')


criar_db()