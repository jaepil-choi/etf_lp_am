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
# # 기본 그래프 생성
#
# 이번 튜토리얼에서는 LangGraph를 사용하여 그래프를 생성하는 방법을 학습합니다.
#
# LangGraph의 그래프를 정의하기 위해서는 다음 단계를 거칩니다.
#
# 1. **State 정의**: 그래프 전체에서 공유되는 상태를 정의합니다.
# 2. **노드 정의**: 각 단계에서 수행할 작업을 함수로 정의합니다.
# 3. **그래프 정의**: 노드와 엣지를 연결하여 워크플로우를 구성합니다.
# 4. **그래프 컴파일**: 실행 가능한 형태로 그래프를 컴파일합니다.
# 5. **그래프 시각화**: 구성된 그래프를 시각적으로 확인합니다.
#
# 그래프 생성 시 조건부 엣지를 사용하는 방법과 다양한 흐름 변경 방법을 알아봅니다.
#
# ![langgraph-building-graphs](assets/langgraph-building-graphs.png)

# %% [markdown]
# ## 환경 설정
#
# 먼저 필요한 환경 변수를 로드하고 LangSmith 추적을 설정합니다.

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
# ## State 정의
#
# State는 그래프의 노드 간에 공유되는 상태 정보를 정의합니다.
#
# LangGraph v1에서는 `TypedDict`만 지원합니다. `Annotated` 타입을 사용하여 각 필드에 대한 리듀서(reducer) 함수를 지정할 수 있습니다.

# %%
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator


# State 정의 (TypedDict 기반 - LangGraph v1 호환)
class GraphState(TypedDict):
    # operator.add를 사용하여 리스트가 누적되도록 설정
    context: Annotated[List[Document], operator.add]
    answer: Annotated[List[Document], operator.add]
    # 문자열 필드는 덮어쓰기 방식
    question: Annotated[str, "user question"]
    sql_query: Annotated[str, "sql query"]
    binary_score: Annotated[str, "binary score yes or no"]


# %% [markdown]
# ## 노드 정의
#
# 노드(Node)는 그래프의 각 단계에서 수행할 작업을 정의합니다.
#
# 각 노드는 State를 입력으로 받아 처리 후 업데이트된 State를 반환합니다.

# %%
def retrieve(state: GraphState) -> GraphState:
    """문서를 검색하는 노드입니다."""
    documents = "검색된 문서"
    return {"context": documents}


def rewrite_query(state: GraphState) -> GraphState:
    """쿼리를 재작성하는 노드입니다."""
    documents = "검색된 문서"
    return {"context": documents}


def llm_gpt_execute(state: GraphState) -> GraphState:
    """GPT 모델로 답변을 생성하는 노드입니다."""
    answer = "GPT 생성된 답변"
    return {"answer": answer}


def llm_claude_execute(state: GraphState) -> GraphState:
    """Claude 모델로 답변을 생성하는 노드입니다."""
    answer = "Claude의 생성된 답변"
    return {"answer": answer}


def relevance_check(state: GraphState) -> GraphState:
    """검색된 문서의 관련성을 체크하는 노드입니다."""
    binary_score = "Relevance Score"
    return {"binary_score": binary_score}


def sum_up(state: GraphState) -> GraphState:
    """결과를 종합하는 노드입니다."""
    answer = "종합된 답변"
    return {"answer": answer}


def search_on_web(state: GraphState) -> GraphState:
    """웹에서 추가 검색을 수행하는 노드입니다."""
    documents = state["context"]
    searched_documents = "검색된 문서"
    documents += searched_documents
    return {"context": documents}


def get_table_info(state: GraphState) -> GraphState:
    """테이블 정보를 가져오는 노드입니다."""
    table_info = "테이블 정보"
    return {"context": table_info}


def generate_sql_query(state: GraphState) -> GraphState:
    """SQL 쿼리를 생성하는 노드입니다."""
    sql_query = "SQL 쿼리"
    return {"sql_query": sql_query}


def execute_sql_query(state: GraphState) -> GraphState:
    """SQL 쿼리를 실행하는 노드입니다."""
    sql_result = "SQL 결과"
    return {"context": sql_result}


def validate_sql_query(state: GraphState) -> GraphState:
    """SQL 쿼리를 검증하는 노드입니다."""
    binary_score = "SQL 쿼리 검증 결과"
    return {"binary_score": binary_score}


def handle_error(state: GraphState) -> GraphState:
    """에러를 처리하는 노드입니다."""
    error = "에러 발생"
    return {"context": error}


def decision(state: GraphState) -> str:
    """조건부 엣지에서 사용되는 라우팅 함수입니다.
    
    binary_score 값에 따라 다음 노드를 결정합니다.
    """
    if state["binary_score"] == "yes":
        return "종료"
    else:
        return "재검색"


# %% [markdown]
# ## 그래프 정의
#
# 노드를 추가하고 엣지로 연결하여 그래프를 정의합니다.
#
# 다양한 흐름을 구성할 수 있습니다:
# - **(1)** Conventional RAG: 기본적인 검색-생성 흐름
# - **(2)** 재검색: 관련성이 낮을 경우 다시 검색
# - **(3)** 멀티 LLM: 여러 LLM을 동시에 활용
# - **(4)** 쿼리 재작성: 검색 전 쿼리를 최적화

# %%
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.graphs import visualize_graph

# StateGraph 초기화
workflow = StateGraph(GraphState)

# 노드를 추가합니다.
workflow.add_node("retrieve", retrieve)
# workflow.add_node("rewrite_query", rewrite_query)  # (4) 쿼리 재작성 옵션

workflow.add_node("GPT 요청", llm_gpt_execute)
# workflow.add_node("Claude 요청", llm_claude_execute)  # (3) 멀티 LLM 옵션
workflow.add_node("GPT_relevance_check", relevance_check)
# workflow.add_node("Claude_relevance_check", relevance_check)  # (3) 멀티 LLM 옵션
workflow.add_node("결과 종합", sum_up)

# 엣지를 연결합니다.
workflow.add_edge("retrieve", "GPT 요청")
# workflow.add_edge("retrieve", "Claude 요청")  # (3) 멀티 LLM 옵션
# workflow.add_edge("rewrite_query", "retrieve")  # (4) 쿼리 재작성 옵션
workflow.add_edge("GPT 요청", "GPT_relevance_check")
workflow.add_edge("GPT_relevance_check", "결과 종합")
# workflow.add_edge("Claude 요청", "Claude_relevance_check")  # (3) 멀티 LLM 옵션
# workflow.add_edge("Claude_relevance_check", "결과 종합")  # (3) 멀티 LLM 옵션

workflow.add_edge("결과 종합", END)

# 조건부 엣지 예시 (주석 처리됨)
# (2) 재검색 옵션
# workflow.add_conditional_edges(
#     "결과 종합",
#     decision,
#     {
#         "재검색": "retrieve",
#         "종료": END,
#     },
# )

# (4) 쿼리 재작성 + 재검색 옵션
# workflow.add_conditional_edges(
#     "결과 종합",
#     decision,
#     {
#         "재검색": "rewrite_query",
#         "종료": END,
#     },
# )

# 시작점을 설정합니다.
workflow.set_entry_point("retrieve")

# 체크포인터를 설정합니다. (대화 기록 저장용)
memory = MemorySaver()

# 그래프를 컴파일합니다.
app = workflow.compile(checkpointer=memory)

# 그래프 시각화
visualize_graph(app)

# %% [markdown]
# ## 정리
#
# 이 튜토리얼에서는 LangGraph의 기본 그래프 구조를 학습했습니다.
#
# ### 핵심 개념
#
# 1. **State**: `TypedDict` 기반으로 그래프 전체에서 공유되는 상태를 정의합니다.
# 2. **Node**: 각 단계에서 수행할 작업을 함수로 정의합니다.
# 3. **Edge**: 노드 간의 연결을 정의합니다.
# 4. **Conditional Edge**: 조건에 따라 다음 노드를 결정하는 분기점입니다.
# 5. **Checkpointer**: 대화 기록을 저장하여 이전 상태로 복원할 수 있습니다.
#
# 다음 튜토리얼에서는 실제 RAG 파이프라인을 구축해보겠습니다.
