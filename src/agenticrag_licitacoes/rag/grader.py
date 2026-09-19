from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from agenticrag_licitacoes.config import settings


# Estrutura que queremos receber do LLM
class RelevanceGrade(BaseModel):

    relevant : bool = Field(
        description=(
            'True se os documentos recuperados possuem '
            'Informações suficientes para responder à pergunta.'
        )
    )

    reason : str = Field(
        description='Motivo curto da decisão.'
    )


# Modelo responsável pela avaliação

llm = ChatOpenAI(
    model = settings.llm_model,
    api_key = settings.openai_api_key,
    use_responses_api=True
)

# Força o modelo a responder no formato definido acima
grader = llm.with_structured_output(
    RelevanceGrade,
    method = 'json_schema',
    strict = True
)

def grade_documents(query: str,
                    results: list[tuple[Document,float]]
                    ) -> RelevanceGrade:

    # Lista vazia de parts
    context_parts = []

    # Iterando sobre os documentos gerados
    for document, distance in results:

        context_parts.append(
            f"""
Distância vetorial: {distance:.4f}

{document.page_content}
"""
        )

    # Concatenado os documentos gerados
    context = "\n\n".join(context_parts)

    # Definicao do Prompt do Modelo de avaliacao
    prompt = f"""
Você é um avaliador de relevância para um sistema RAG.

Sua tarefa NÃO é responder à pergunta.

Avalie apenas se os documentos recuperados possuem
informações suficientes para responder corretamente à pergunta.

PERGUNTA:
{query}

DOCUMENTOS RECUPERADOS:
{context}
"""
    # Realizar analise de conformidade entre pergunta e documentos recuperados
    result = grader.invoke(prompt)

    # Retorna o resultado final
    return result


if __name__ == "__main__":

    from agenticrag_licitacoes.rag.retriever import search_documents

    question = "Quais são as penalidades em uma licitação?"

    documents = search_documents(
        query=question,
        k=3
    )

    result = grade_documents(
        query=question,
        results=documents
    )

    print(f"Relevante: {result.relevant}")
    print(f"Motivo: {result.reason}")