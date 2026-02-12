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
# # Naive RAG
#
# 이번 튜토리얼에서는 LangGraph를 사용하여 가장 기본적인 RAG(Retrieval-Augmented Generation) 파이프라인을 구축합니다.
#
# **학습 목표**
#
# - PDF 문서를 기반으로 Retriever와 Chain을 생성하는 방법을 학습합니다.
# - LangGraph의 State, Node, Edge를 활용하여 RAG 워크플로우를 구성합니다.
# - 그래프를 실행하고 스트리밍 출력하는 방법을 익힙니다.
#
# ![langgraph-naive-rag](assets/langgraph-naive-rag.png)

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
# ## PDF 기반 Retrieval Chain 생성
#
# PDF 문서를 기반으로 Retrieval Chain을 생성합니다. LangGraph에서는 Retriever와 Chain을 분리하여 각 노드에서 세부 처리를 수행할 수 있습니다.
#
# **사용 문서**
#
# - 소프트웨어정책연구소(SPRi) - 2023년 12월호
# - 파일명: `SPRI_AI_Brief_2023년12월호_F.pdf`

# %%
from rag.pdf import PDFRetrievalChain

# PDF 문서를 로드합니다.
pdf = PDFRetrievalChain(["data/SPRI_AI_Brief_2023년12월호_F.pdf"]).create_chain()

# retriever와 chain을 생성합니다.
pdf_retriever = pdf.retriever
pdf_chain = pdf.chain

# %% [markdown]
# ### Retriever 테스트
#
# `pdf_retriever`를 사용하여 검색 결과를 가져옵니다.

# %%
# 검색 테스트
search_result = pdf_retriever.invoke("앤스로픽에 투자한 기업과 투자금액을 알려주세요.")
search_result

# %% [markdown]
# ### Chain 테스트
#
# 검색된 문서를 Chain의 context로 전달하여 답변을 생성합니다.

# %%
# 검색 결과를 기반으로 답변을 생성합니다.
answer = pdf_chain.invoke(
    {
        "question": "앤스로픽에 투자한 기업과 투자금액을 알려주세요.",
        "context": search_result
    }
)
print(answer)

# %% [markdown]
# ## State 정의
#
# `State`는 그래프의 노드 간에 공유되는 상태를 정의합니다.
#
# LangGraph v1에서는 `TypedDict` 형식만 지원합니다. `Annotated` 타입을 사용하여 리듀서 함수를 지정할 수 있습니다.

# %%
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


# GraphState 상태 정의 (TypedDict 기반 - LangGraph v1 호환)
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 사용자 질문
    context: Annotated[str, "Context"]  # 검색된 문서
    answer: Annotated[str, "Answer"]  # 생성된 답변
    messages: Annotated[list, add_messages]  # 대화 히스토리 (누적)


# %% [markdown]
# ## 노드(Node) 정의
#
# 노드는 각 단계를 처리하는 함수입니다. State를 입력으로 받아 처리 후 업데이트된 State를 반환합니다.
#
# **노드 목록**
#
# - `retrieve_document`: 문서를 검색합니다.
# - `llm_answer`: 검색된 문서를 기반으로 답변을 생성합니다.

# %%
from langchain_teddynote.messages import messages_to_history
from rag.utils import format_docs


def retrieve_document(state: GraphState) -> GraphState:
    """문서를 검색하는 노드입니다.
    
    사용자 질문을 기반으로 관련 문서를 검색하고 포맷팅합니다.
    """
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = pdf_retriever.invoke(latest_question)

    # 검색된 문서를 형식화합니다. (프롬프트 입력용)
    retrieved_docs = format_docs(retrieved_docs)

    # 검색된 문서를 context 키에 저장합니다.
    return {"context": retrieved_docs}


def llm_answer(state: GraphState) -> GraphState:
    """답변을 생성하는 노드입니다.
    
    검색된 문서와 대화 기록을 기반으로 답변을 생성합니다.
    """
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 검색된 문서를 상태에서 가져옵니다.
    context = state["context"]

    # 체인을 호출하여 답변을 생성합니다.
    response = pdf_chain.invoke(
        {
            "question": latest_question,
            "context": context,
            "chat_history": messages_to_history(state["messages"]),
        }
    )
    
    # 생성된 답변과 메시지를 상태에 저장합니다.
    return {
        "answer": response,
        "messages": [("user", latest_question), ("assistant", response)],
    }


# %% [markdown]
# ## 그래프 생성
#
# 노드를 추가하고 엣지로 연결하여 그래프를 생성합니다.
#
# **엣지 종류**
#
# - **일반 엣지**: 항상 다음 노드로 이동합니다.
# - **조건부 엣지**: 조건에 따라 다음 노드를 결정합니다.

# %%
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 그래프 생성
workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("llm_answer", llm_answer)

# 엣지 정의
workflow.add_edge("retrieve", "llm_answer")  # 검색 -> 답변
workflow.add_edge("llm_answer", END)  # 답변 -> 종료

# 그래프 진입점 설정
workflow.set_entry_point("retrieve")

# 체크포인터 설정 (대화 기록 저장)
memory = MemorySaver()

# 그래프 컴파일
app = workflow.compile(checkpointer=memory)

# %% [markdown]
# ### 그래프 시각화
#
# 컴파일한 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(app)

# %% [markdown]
# ## 그래프 실행
#
# 그래프를 실행하여 RAG 파이프라인을 동작시킵니다.
#
# **주요 파라미터**
#
# - `config`: 그래프 실행 시 필요한 설정 정보를 전달합니다.
# - `recursion_limit`: 그래프 실행 시 재귀 최대 횟수를 설정합니다.
# - `inputs`: 그래프 실행에 필요한 입력 데이터입니다.
#
# **참고**
#
# - 스트리밍 출력에 대한 자세한 내용은 [LangGraph 스트리밍 모드](https://wikidocs.net/265770)를 참고해주세요.

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

# config 설정 (재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

# 질문 입력
inputs = GraphState(question="앤스로픽에 투자한 기업과 투자금액을 알려주세요.")

# 그래프 실행 (일반 모드)
invoke_graph(app, inputs, config)

# %%
# 그래프 실행 (스트리밍 모드)
stream_graph(app, inputs, config)

# %% [markdown]
# ### 결과 확인
#
# 그래프 실행 후 최종 상태를 확인합니다.

# %%
# 최종 상태 조회
outputs = app.get_state(config).values

print(f'Question: {outputs["question"]}')
print("===" * 20)
print(f'Answer:\n{outputs["answer"]}')

# %% [markdown]
# ## 정리
#
# 이 튜토리얼에서는 LangGraph를 사용하여 기본적인 Naive RAG 파이프라인을 구축했습니다.
#
# ### 핵심 개념
#
# 1. **State**: `TypedDict` 기반으로 노드 간 공유 상태를 정의합니다.
# 2. **Node**: 각 단계의 처리 로직을 함수로 구현합니다.
# 3. **Edge**: 노드 간의 흐름을 정의합니다.
# 4. **Checkpointer**: 대화 기록을 저장하여 멀티턴 대화를 지원합니다.
#
# ### 다음 단계
#
# 다음 튜토리얼에서는 검색된 문서에 대한 **관련성 체크(Groundedness Check)**를 추가하여 RAG 파이프라인의 품질을 향상시키는 방법을 학습합니다.
