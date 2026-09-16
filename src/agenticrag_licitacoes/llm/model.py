# Libs do projeto
from openai import OpenAI

from agenticrag_licitacoes.config import settings

# Defininco o Cliente da OpenAI
client = OpenAI(
    api_key = settings.openai_api_key
)

# Criando funcao para gerar a resposta
def generate_response(prompt : str) -> str:

    # Definindo o Response de acordo com o Prompt
    response = client.responses.create(
        model = settings.llm_model,
        input = prompt
    )

    # Retornando a resposta do modelode LLM
    return response.output_text


if __name__ == '__main__':

    resposta = generate_response(
        'Explique o que é RAG em uma frase.'
    )

    print(resposta)