# Libs

from typing import NotRequired, TypedDict, Literal
from langchain_core.documents import Document
from numpy._core.numerictypes import typeDict

# Definicao da classe do agente

class AgentState(TypedDict):


    # Define a query de entrada do usuário
    query : str

    # Definicao da rota do agente
    route :NotRequired[Literal['rag','web']]

    # Define a lista de documentos recuperados
    documents : NotRequired[list[tuple[Document,float]]]

    # Define se os documentos recuperados são relevantes
    is_relevant: NotRequired[bool]

    # Define a resposta do Web Search
    web_results : NotRequired[list[dict]]

    # Define a resposta do agente
    answer: NotRequired[str]



