from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def processar_pdfs(pasta_pdf, caminho_faiss):
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

    if os.path.exists(caminho_faiss):
        db = FAISS.load_local(caminho_faiss, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(caminho_faiss)
    return len(chunks)
