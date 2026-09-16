# Libs
from openai import OpenAI
import math

from agenticrag_licitacoes.config import settings

# Definindo cliente OpenAI
client = OpenAI(
    api_key= settings.openai_api_key
)

# Definindo funcao para gerar embeddings dos chunks
def generate_embedding(text : str) -> list[float]:

    # Definindo o embeddings pelo input de texto
    response = client.embeddings.create(
        model = settings.embedding_model,
        input = text
    )

    return response.data[0].embedding

# Definindo funcao de Similariedade consena
def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":

    frase_a = "Quais são as penalidades em uma licitação?"

    frase_b = (
        "As sanções incluem advertências, multas e suspensão."
    )

    frase_c = "Qual é a previsão do tempo para amanhã?"

    embedding_a = generate_embedding(frase_a)
    embedding_b = generate_embedding(frase_b)
    embedding_c = generate_embedding(frase_c)

    similarity_ab = cosine_similarity(
        embedding_a,
        embedding_b
    )

    similarity_ac = cosine_similarity(
        embedding_a,
        embedding_c
    )

    print(f"Similaridade A x B: {similarity_ab}")
    print(f"Similaridade A x C: {similarity_ac}")