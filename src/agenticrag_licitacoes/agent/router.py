from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agenticrag_licitacoes.config import settings


class RouteDecision(BaseModel):

    route: Literal["rag", "web"] = Field(
        description=(
            "Use 'rag' para perguntas relacionadas a licitações "
            "e aos documentos internos. "
            "Use 'web' para assuntos externos."
        )
    )

    reason: str = Field(
        description="Motivo curto da decisão."
    )


llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
    use_responses_api=True
)


router = llm.with_structured_output(
    RouteDecision,
    method="json_schema",
    strict=True
)


def route_query(query: str) -> RouteDecision:

    prompt = f"""
Você é o roteador de um sistema Agentic RAG.

Decida qual fonte deve ser usada para responder à pergunta.

Use:

rag:
quando a pergunta estiver relacionada a licitações,
contratações públicas, modalidades, fases, sanções,
critérios de julgamento ou assuntos presentes
na base documental interna.

web:
quando a pergunta for sobre qualquer outro assunto
ou depender de informação externa.

PERGUNTA:
{query}
"""

    return router.invoke(prompt)


if __name__ == "__main__":

    result = route_query(
        "Quais são as penalidades em uma licitação?"
    )

    print("Rota:", result.route)
    print("Motivo:", result.reason)