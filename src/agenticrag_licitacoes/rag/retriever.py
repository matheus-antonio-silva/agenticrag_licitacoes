# Libs

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from agenticrag_licitacoes.config import settings

# Instanciando Chroma Vector Store
def get_vector_store() -> Chroma:

    # Definindo embeddings
    embeddings = OpenAIEmbeddings(
        model = settings.embedding_model,
        api_key=settings.openai_api_key
    )

    # Instaciando Vector Store
    vector_store = Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
        embedding_function=embeddings
    )

    # Retornando instancia da VectorStore
    return vector_store


# Funcao de Retriever

def search_documents(query : str,k : int = 3):
    
    # Definindo VectorStore
    vector_store = get_vector_store()

    # retornando os 3 chunks mais simlares com a query
    results = vector_store.similarity_search_with_relevance_scores(
        query = query,
        k = k
    )

    # Retornando resultado do Retriever
    return results

if __name__ == "__main__":

    query = "Quais são as penalidades em uma licitação?"

    results = search_documents(
        query=query,
        k=3
    )

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Resultado {index} ---")
        print(f"Score: {score:.4f}")
        print(f"Fonte: {document.metadata['source']}")
        print(document.page_content)

