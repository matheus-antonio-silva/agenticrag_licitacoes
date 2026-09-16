# Libs

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai.types import embedding

from agenticrag_licitacoes.config import settings
from agenticrag_licitacoes.ingestion.loader import load_documents
from agenticrag_licitacoes.ingestion.splitter import split_documents

# Definindo o modelo de embeddings
def get_embeddings() -> OpenAIEmbeddings:

    embeddings = OpenAIEmbeddings(
        model = settings.embedding_model,
        api_key = settings.openai_api_key
    )

    return embeddings

# Criando banco de dados vetorial
def create_vector_store() -> Chroma:

    # Definindo os documentos
    documents = load_documents()

    # Criando os chunks
    chunks = split_documents(documents)

    # Gerando o modelo de embeddings
    embeddings = get_embeddings()

    # Criando a instancia do Chroma

    vector_store = Chroma.from_documents(
        documents=chunks, # Definindo os chunks para ser armazenado,
        embedding=embeddings, # Definindo o modelo de embedding
        collection_name = settings.collection_name, # Definido o nome da coleção que armazenará os chunks
        persist_directory=settings.vector_store_path # Definindo o local de armazenamento do banco de dados vetorial
    )

    # Retornando o banco de dados
    return vector_store

if __name__ == "__main__":

    vector_store = create_vector_store()

    print("Vector store criado com sucesso.")