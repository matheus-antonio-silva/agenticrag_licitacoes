# Libs
import chunk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Lib para splitar os documentos carregado
def split_documents(documents : list[Document]) -> list[Document]:

    # Definindo o splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, # Definindo o tamanho do chunk
        chunk_overlap = 100, # Definindo o tamanho da sobreposicao do chunk
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ] # Definindo os separadores do documento
    )

    # Utilizando o splitter dentro dos documentos
    chunks = splitter.split_documents(documents)

    # Retornando os chunk
    return chunks