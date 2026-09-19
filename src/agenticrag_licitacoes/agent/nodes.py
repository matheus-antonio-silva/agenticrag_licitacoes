# Libs

from agenticrag_licitacoes.agent.state import AgentState
from agenticrag_licitacoes.rag.retriever import search_documents
from agenticrag_licitacoes.rag.grader import grade_documents
from agenticrag_licitacoes.rag.pipeline import (generate_answer,generate_web_answer)
from agenticrag_licitacoes.tools.web_search import search_web
from agenticrag_licitacoes.agent.router import route_query



# Definicao do nó de Retriever

def retrieve_node(state : AgentState) -> dict:

    # Pergunta do usuário

    query = state['query']

    # Gerando documentos com base na pergunta do usuário
    documents = search_documents(
        query = query,
        k = 3
    )

    # Retornando os documentos gerados pelo Retriever
    return {
        'documents' : documents
    }


# Definicao do nó de router
def router_node(state: AgentState) -> dict:

    query = state["query"]

    decision = route_query(query)

    return {
        "route": decision.route
    }

# Definicao do nó de Grader (Avaliacao do resultado do Retrieve)

def grade_node(state: AgentState) -> dict:

    # Pergunta do usuário
    query = state['query']

    # Documentos retornados pela etapa de Retrieve
    documents = state['documents']

    # Análise de relevãncia dos documentos retornados
    result = grade_documents(
        query = query,
        results = documents
    )

    # Retornando resultado de relevancia do Retriever
    return {
        'is_relevant' : result.relevant
    }

# Nó de geracao de resposta do agente
def generate_node(state: AgentState) -> dict:

    # Pergunta do usuário
    query = state['query']
    # Documentos retornados do Retriever
    documents = state['documents']

    # Gerando resposta do agente
    answer = generate_answer(
        query = query,
        results = documents
    )

    # Retornando resposta
    return {
        "answer" : answer
    }

# Nó de resposta do web search
def web_search_node(state : AgentState) -> dict:

    # Pergunta do usuário
    query = state['query']

    # Buscando resultados
    results = search_web(
        query = query,
        max_results=5
    )

    # Retorno dos resultados
    return {
        'web_results' : results
    }

# Nó que gera resposta web

def generate_web_node(state : AgentState) -> dict:

    # Pergunta do usuário
    query = state['query']
    # Buscando informacoes na web
    web_results = state['web_results']
    # Retornando respostas
    answer = generate_web_answer(
        query = query,
        web_results = web_results
    )
    # Retornando respota fginal
    return {
        "answer" : answer
    }