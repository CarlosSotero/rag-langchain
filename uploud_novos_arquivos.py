from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Função para processar os novos PDFs
def processar_pdfs(pasta_pdf, pasta_base):
    documentos = []
    for arquivo in os.listdir(pasta_pdf):
        if arquivo.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pasta_pdf, arquivo))
            documentos.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, add_start_index=True)
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        encode_kwargs={'normalize_embeddings': True}
    )

    # Carrega o banco existente e adiciona os novos documentos
    db = Chroma(persist_directory=pasta_base, embedding_function=embeddings)
    db.add_documents(chunks)
    db.persist()

    return len(chunks)
