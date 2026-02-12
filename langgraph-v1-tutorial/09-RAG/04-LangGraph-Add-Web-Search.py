# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 웹 검색 모듈 추가
#
# 이번 튜토리얼에서는 RAG 파이프라인에 **웹 검색(Web Search)** 기능을 추가합니다.
#
# **학습 목표**
#
# - Tavily Search를 활용한 웹 검색 방법을 학습합니다.
# - 관련성 체크 실패 시 웹 검색으로 폴백하는 흐름을 구현합니다.
# - 재귀 상태 없이 안정적인 RAG 파이프라인을 구축합니다.
#
# **참고**
#
# - 이전 튜토리얼에서 확장된 내용이므로, 겹치는 부분은 간략히 설명합니다.
#
# ![langgraph-web-search](assets/langgraph-web-search.png)

# %% [markdown]
# ## 환경 설정

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-RAG")

# %% [markdown]
# ## PDF 기반 Retrieval Chain 생성

# %%
from rag.pdf import PDFRetrievalChain

# PDF 문서를 로드합니다.
pdf = PDFRetrievalChain(["data/SPRI_AI_Brief_2023년12월호_F.pdf"]).create_chain()

# retriever와 chain을 생성합니다.
pdf_retriever = pdf.retriever
pdf_chain = pdf.chain

# %% [markdown]
# ## State 정의

# %%
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


# GraphState 상태 정의 (TypedDict 기반 - LangGraph v1 호환)
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 사용자 질문
    context: Annotated[str, "Context"]  # 검색된 문서
    answer: Annotated[str, "Answer"]  # 생성된 답변
    messages: Annotated[list, add_messages]  # 대화 히스토리 (누적)
    relevance: Annotated[str, "Relevance"]  # 관련성 체크 결과 (yes/no)


# %% [markdown]
# ## 노드(Node) 정의
#
# 기존 노드에 **웹 검색 노드**를 추가합니다.

# %%
from langchain_openai import ChatOpenAI
from langchain_teddynote.evaluator import GroundednessChecker
from langchain_teddynote.messages import messages_to_history
from rag.utils import format_docs


def retrieve_document(state: GraphState) -> GraphState:
    """문서를 검색하는 노드입니다."""
    latest_question = state["question"]
    retrieved_docs = pdf_retriever.invoke(latest_question)
    retrieved_docs = format_docs(retrieved_docs)
    return {"context": retrieved_docs}


def llm_answer(state: GraphState) -> GraphState:
    """답변을 생성하는 노드입니다."""
    latest_question = state["question"]
    context = state["context"]

    response = pdf_chain.invoke(
        {
            "question": latest_question,
            "context": context,
            "chat_history": messages_to_history(state["messages"]),
        }
    )
    
    return {
        "answer": response,
        "messages": [("user", latest_question), ("assistant", response)],
    }


def relevance_check(state: GraphState) -> GraphState:
    """관련성을 체크하는 노드입니다."""
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0), 
        target="question-retrieval"
    ).create()

    response = question_answer_relevant.invoke(
        {"question": state["question"], "context": state["context"]}
    )

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    return {"relevance": response.score}


def is_relevant(state: GraphState) -> str:
    """조건부 엣지에서 사용되는 라우팅 함수입니다."""
    if state["relevance"] == "yes":
        return "relevant"
    else:
        return "not relevant"


# %% [markdown]
# ## 웹 검색 노드 추가
#
# `TavilySearch` 도구를 사용하여 웹 검색을 수행합니다.
#
# 먼저 웹 검색 도구의 사용법을 확인합니다.

# %%
from langchain_teddynote.tools.tavily import TavilySearch

# 검색 도구 생성
tavily_tool = TavilySearch()

search_query = "2024년 노벨 문학상 수상자는?"

# 다양한 파라미터를 사용한 검색 예제
search_result = tavily_tool.search(
    query=search_query,  # 검색 쿼리
    max_results=3,  # 최대 검색 결과
    format_output=True,  # 결과 포맷팅
)

# 검색 결과 출력
print(search_result)


# %% [markdown]
# ### 웹 검색 노드 함수 정의
#
# 웹 검색을 수행하는 노드 함수를 정의합니다.

# %%
def web_search(state: GraphState) -> GraphState:
    """웹 검색을 수행하는 노드입니다.
    
    관련성 체크 실패 시 웹에서 추가 정보를 검색합니다.
    """
    # 검색 도구 생성
    tavily_tool = TavilySearch()

    search_query = state["question"]

    # 웹 검색 수행
    search_result = tavily_tool.search(
        query=search_query,  # 검색 쿼리
        topic="general",  # 일반 주제
        max_results=6,  # 최대 검색 결과
        format_output=True,  # 결과 포맷팅
    )

    return {"context": search_result}


# %% [markdown]
# ## 그래프 생성
#
# 관련성이 없을 경우 **웹 검색**으로 폴백하는 흐름을 구현합니다.

# %%
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 그래프 정의
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("web_search", web_search)  # 웹 검색 노드 추가

# 엣지 추가
workflow.add_edge("retrieve", "relevance_check")  # 검색 -> 관련성 체크

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "relevance_check",
    is_relevant,
    {
        "relevant": "llm_answer",  # 관련성 있음 -> 답변 생성
        "not relevant": "web_search",  # 관련성 없음 -> 웹 검색
    },
)

workflow.add_edge("web_search", "llm_answer")  # 웹 검색 -> 답변 생성
workflow.add_edge("llm_answer", END)  # 답변 -> 종료

# 그래프 진입점 설정
workflow.set_entry_point("retrieve")

# 체크포인터 설정
memory = MemorySaver()

# 그래프 컴파일
app = workflow.compile(checkpointer=memory)

# %% [markdown]
# ### 그래프 시각화

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(app)

# %% [markdown]
# ## 그래프 실행
#
# 관련성 체크가 실패하면 웹 검색을 통해 답변을 생성합니다.

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid

# config 설정
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

# 질문 입력 (PDF에 없는 최신 정보 - 웹 검색 필요)
inputs = GraphState(question="2024년 노벨 문학상 수상자는?")

# 그래프 실행
invoke_graph(app, inputs, config, ["relevance_check", "llm_answer"])


# %%
# 콜백 함수 예시 (특정 노드 출력만 처리)
def callback_function(args):
    if args["node"] == "llm_answer":
        print(args["content"], end="", flush=True)


# 그래프 스트리밍 출력 (콜백 적용)
stream_graph(
    app, inputs, config, ["relevance_check", "llm_answer"], callback=callback_function
)

# %% [markdown]
# ### 결과 확인

# %%
# 최종 출력 확인
outputs = app.get_state(config).values

print(f'Question: {outputs["question"]}')
print("===" * 20)
print(f'Answer:\n{outputs["answer"]}')

# %% [markdown]
# ## 정리
#
# 이 튜토리얼에서는 RAG 파이프라인에 **웹 검색** 기능을 추가했습니다.
#
# ### 핵심 개념
#
# 1. **TavilySearch**: 웹에서 최신 정보를 검색하는 도구입니다.
# 2. **폴백 전략**: 관련성 체크 실패 시 웹 검색으로 보완합니다.
# 3. **재귀 상태 해결**: 웹 검색으로 폴백하여 무한 루프를 방지합니다.
#
# ### 다음 단계
#
# 다음 튜토리얼에서는 **쿼리 재작성(Query Rewrite)**을 추가하여 검색 품질을 더욱 향상시키는 방법을 학습합니다.
