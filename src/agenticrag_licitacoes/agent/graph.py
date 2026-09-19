# libs
from typing import Literal

from langgraph.graph import END,START,StateGraph

from agenticrag_licitacoes.agent.nodes import (
    generate_node,
    generate_web_node,
    grade_node,
    retrieve_node,
    router_node,
    web_search_node,
)
from agenticrag_licitacoes.agent.state import AgentState

# Funcao para roteador

def route_after_grade(
    state : AgentState
) -> Literal['generate','web_search']:
    if state['is_relevant']:
        return 'generate'
    return 'web_search'

# Funcao de roteador inciial
def route_after_router(
    state: AgentState
) -> Literal["retrieve", "web_search"]:

    if state["route"] == "rag":
        return "retrieve"

    return "web_search"

# Definindo agente
builder = StateGraph(AgentState)

# Adicionando nós no agente
builder.add_node(
    'retrieve',
    retrieve_node
)

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    'grade',
    grade_node
)

builder.add_node(
    'generate',
    generate_node
)

builder.add_node(
    "web_search",
    web_search_node
)

builder.add_node(
    "generate_web",
    generate_web_node
)

# Adicionando arestas do Agente
builder.add_edge(
    START,
    "router"
)

builder.add_edge(
    'retrieve',
    'grade'
)

builder.add_conditional_edges(
    source="router",
    path=route_after_router,
    path_map={
        "retrieve": "retrieve",
        "web_search": "web_search",
    },
)

builder.add_conditional_edges(
    source = 'grade',
    path = route_after_grade,
    path_map=
    {
        'generate' : 'generate',
        'web_search' : 'web_search'
    }
)



builder.add_edge(
    "web_search",
    "generate_web"
)


builder.add_edge(
    'generate',
    END
)

builder.add_edge(
    'generate_web',
    END
)


# Compilando agente para se tornar um executável
graph = builder.compile()

if __name__ == "__main__":

    result = graph.invoke(
        {
            "query": "Quais são as penalidades em uma licitação?"
            
        }
    )

    print("Rota:", result["route"])

    if "is_relevant" in result:
        print("Relevante:", result["is_relevant"])

    print("\nResposta:")
    print(result["answer"])