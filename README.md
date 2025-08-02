# rag-langchain
## RAG em Python com LangChain, Streamlit e Mistral

Este é um projeto em desenvolvimento que implementa um sistema de Recuperação Aumentada por Geração (RAG) usando Python. A proposta é integrar documentos PDF a um modelo de linguagem para responder perguntas de forma inteligente.

## 🚧 Status do Projeto

🔨 Concluído  
✅ Banco vetorial com FAISS (compatível com o Streamlit Cloud)  
✅ Integração com modelo LLM (Mistral via OpenRouter)  
✅ Interface gráfica criada com Streamlit  
✅ Upload dinâmico de novos PDFs  
✅ Deploy funcionando no **Streamlit Cloud** 

## 📁 Estrutura atual

- `criar_db.py`: script original para criação do banco vetorial a partir dos PDFs
- `main.py`: aplicação principal com interface Streamlit (consultas e upload de novos arquivos)
- `uploud_novos_arquivos.py`: lógica modular para processar novos PDFs e atualizar o banco
- `requirements.txt`: dependências do projeto
- `uploads_temp/`: pasta temporária usada no deploy

## 📌 Objetivo

Criar uma aplicação de perguntas e respostas baseada em documentos locais (PDFs), utilizando RAG com LangChain, modelo Mistral e interface simples via Streamlit.

## 🧠 Tecnologias Utilizadas

- **Python**
- **LangChain**
- **FAISS** (banco vetorial)
- **Mistral** (via OpenRouter)
- **Sentence Transformers** (modelo `paraphrase-multilingual-mpnet-base-v2`)
- **Streamlit** (interface gráfica)
- **dotenv** (gerenciamento de chaves)

## ▶️ Como Executar (localmente)

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/rag-langchain.git
   cd rag-langchain

2. Crie um ambiente virtual (opcional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   
4. Crie um arquivo .env com sua chave da OpenRouter:
   ```bash
   OPENROUTER_API_KEY=sua_chave_aqui

5. Rode o app:
   ```bash
   streamlit run main.py
   
☁️ Como Acessar Online  
🔗 [Clique aqui para abrir a aplicação no Streamlit](https://rag-pdf-carlos-sotero.streamlit.app/)


🤝 Contribuições  
Atualmente o projeto não está aberto para contribuições externas, mas feedbacks e sugestões são sempre bem-vindos!  

📄 Licença
Distribuído sob a MIT License. Consulte o arquivo LICENSE para mais detalhes.
