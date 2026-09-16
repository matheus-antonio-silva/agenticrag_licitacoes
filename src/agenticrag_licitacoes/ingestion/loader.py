# Libs
import json
from pathlib import Path
from langchain_core.documents import Document


# Definicao do Path do Arquivo
DATA_DIR = Path('data/raw')

# Lib que faz a leitura do documento Json
def load_documents(data_dir : Path = DATA_DIR) -> list[Document]:

    # Define lista vazia de documentos
    documents = []

    # Realizando for loop em cada documento
    for file_path in sorted(data_dir.glob("*.json")):

        # Abrindo e Formatando os arquivos Json
        with file_path.open(
            "r",
            encoding="utf-8-sig"
        ) as file:

            data = json.load(file)

        content = "\n\n".join(
            f"{key}\n{value}"
            for key, value in data.items()
        )

        # Lendo os Documentos Json
        document = Document(
            page_content=content,
            metadata={
                "source": file_path.name
            }
        )

        documents.append(document)

    return documents

if __name__ == "__main__":
    from agenticrag_licitacoes.ingestion.splitter import split_documents

    documents = load_documents()

    chunks = split_documents(documents)

    print(f"Documentos carregados: {len(documents)}")
    print(f"Chunks criados: {len(chunks)}")

    print("\nPrimeiro chunk:")
    print(chunks[0].page_content)

    print("\nMetadata:")
    print(chunks[0].metadata)