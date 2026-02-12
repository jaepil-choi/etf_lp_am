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
# # Agentic RAG
#
# 이번 튜토리얼에서는 **에이전트(Agent)** 기반의 RAG 시스템을 구축합니다.
#
# 에이전트는 검색 도구를 사용할지 여부를 스스로 결정합니다. 에이전트에 대한 자세한 내용은 [Agent 튜토리얼](https://wikidocs.net/233782)을 참고하세요.
#
# **학습 목표**
#
# - LLM에 검색 도구를 바인딩하여 에이전트를 구성하는 방법을 학습합니다.
# - `ToolNode`와 `tools_condition`을 활용한 에이전트 그래프를 구축합니다.
# - 문서 관련성 평가 및 쿼리 재작성 흐름을 구현합니다.
#
# ![langgraph-agentic-rag](assets/langgraph-agentic-rag.png)

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
# ## 검색 도구 생성
#
# `create_retriever_tool`을 사용하여 Retriever를 도구로 변환합니다.
#
# **`document_prompt` 사용 가능한 키**
#
# - `page_content`: 문서 내용
# - `source`: 문서 출처
# - `page`: 페이지 번호

# %%
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate

# PDF 문서 검색 도구 생성
retriever_tool = create_retriever_tool(
    pdf_retriever,
    "pdf_retriever",
    "Search and return information about SPRI AI Brief PDF file. It contains useful information on recent AI trends. The document is published on Dec 2023.",
    document_prompt=PromptTemplate.from_template(
        "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
    ),
)

# 도구 리스트에 추가
tools = [retriever_tool]

# %% [markdown]
# ## Agent State 정의
#
# 에이전트의 상태는 `messages` 목록으로 구성됩니다. 각 노드는 이 목록에 메시지를 추가합니다.

# %%
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# 에이전트 상태 정의 (TypedDict 기반 - LangGraph v1 호환)
class AgentState(TypedDict):
    # add_messages 리듀서를 사용하여 메시지 시퀀스 관리
    messages: Annotated[Sequence[BaseMessage], add_messages]


# %% [markdown]
# ## 노드와 엣지
#
# 에이전트 기반 RAG 그래프는 다음과 같이 구성됩니다.
#
# - **상태**: 메시지들의 집합
# - **노드**: 상태를 업데이트(추가)
# - **조건부 엣지**: 다음에 방문할 노드를 결정

# %% [markdown]
# ### 문서 평가기(Grader) 및 노드 정의
#
# 검색된 문서의 관련성을 평가하는 Grader와 각 노드를 정의합니다.

# %%
from typing import Literal
from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from langchain_teddynote.models import get_model_name, LLMs

# 최신 모델 이름 가져오기
MODEL_NAME = get_model_name(LLMs.GPT4o)


# 관련성 평가를 위한 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성을 평가하는 이진 점수입니다."""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'"
    )


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """검색된 문서의 관련성을 평가하는 함수입니다."""
    # LLM 초기화
    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)

    # 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(GradeDocuments)

    # 프롬프트 템플릿 정의
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # 체인 생성
    chain = prompt | llm_with_tool

    # 현재 상태에서 메시지 추출
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    retrieved_docs = last_message.content

    # 관련성 평가 실행
    scored_result = chain.invoke({"question": question, "context": retrieved_docs})
    score = scored_result.binary_score

    # 관련성 여부에 따른 결정
    if score == "yes":
        print("==== [DECISION: DOCS RELEVANT] ====")
        return "generate"
    else:
        print("==== [DECISION: DOCS NOT RELEVANT] ====")
        return "rewrite"


def agent(state):
    """에이전트 노드입니다. 도구 사용 여부를 결정합니다."""
    messages = state["messages"]

    # LLM 초기화 및 도구 바인딩
    model = ChatOpenAI(temperature=0, streaming=True, model=MODEL_NAME)
    model = model.bind_tools(tools)

    # 에이전트 응답 생성
    response = model.invoke(messages)

    return {"messages": [response]}


def rewrite(state):
    """질문을 재작성하는 노드입니다."""
    print("==== [QUERY REWRITE] ====")
    messages = state["messages"]
    question = messages[0].content

    # 질문 개선을 위한 프롬프트 구성
    msg = [
        HumanMessage(
            content=f"""\n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # LLM으로 질문 개선
    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    response = model.invoke(msg)

    return {"messages": [response]}


def generate(state):
    """답변을 생성하는 노드입니다."""
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content

    # RAG 프롬프트 가져오기
    prompt = hub.pull("teddynote/rag-prompt")

    # LLM 초기화 및 체인 구성
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    # 답변 생성
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# %% [markdown]
# ## 그래프 생성
#
# 에이전트 그래프를 구성합니다.
#
# - `agent`로 시작하여 함수 호출 여부를 결정합니다.
# - 함수 호출 시 `retrieve` 노드에서 도구를 실행합니다.
# - 도구 출력을 상태에 추가하여 에이전트를 다시 호출합니다.

# %%
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# AgentState 기반 워크플로우 초기화
workflow = StateGraph(AgentState)

# 노드 정의
workflow.add_node("agent", agent)  # 에이전트 노드
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # 검색 노드
workflow.add_node("rewrite", rewrite)  # 질문 재작성 노드
workflow.add_node("generate", generate)  # 답변 생성 노드

# 엣지 연결
workflow.add_edge(START, "agent")

# 검색 여부 결정을 위한 조건부 엣지
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # 에이전트 결정 평가
    {
        "tools": "retrieve",  # 도구 호출 -> 검색
        END: END,  # 도구 미호출 -> 종료
    },
)

# 검색 후 문서 평가를 위한 조건부 엣지
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # 문서 품질 평가
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# 그래프 컴파일
graph = workflow.compile(checkpointer=MemorySaver())

# %% [markdown]
# ### 그래프 시각화

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(graph)

# %% [markdown]
# ## 그래프 실행

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid

# config 설정
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

# 문서 검색이 필요한 질문
inputs = {
    "messages": [
        ("user", "삼성전자가 개발한 생성형 AI 의 이름은?"),
    ]
}

# 그래프 실행
invoke_graph(graph, inputs, config)

# %%
# 그래프 스트리밍 출력
stream_graph(graph, inputs, config, ["agent", "rewrite", "generate"])

# %% [markdown]
# ### 문서 검색이 불필요한 질문 예시
#
# 에이전트가 검색 도구를 사용하지 않고 바로 답변하는 경우입니다.

# %%
# 문서 검색이 불필요한 질문 예시
inputs = {
    "messages": [
        ("user", "대한민국의 수도는?"),
    ]
}

# 그래프 실행
stream_graph(graph, inputs, config, ["agent", "rewrite", "generate"])

# %% [markdown]
# ### 재귀 상태 처리
#
# 문서 검색이 불가능한 질문의 경우, 지속적인 검색 시도로 인해 `GraphRecursionError`가 발생할 수 있습니다.

# %%
from langgraph.errors import GraphRecursionError

# 문서 검색이 불가능한 질문
inputs = {
    "messages": [
        ("user", "테디노트의 랭체인 튜토리얼에 대해서 알려줘"),
    ]
}

try:
    # 그래프 실행
    stream_graph(graph, inputs, config, ["agent", "rewrite", "generate"])
except GraphRecursionError as recursion_error:
    print(f"GraphRecursionError: {recursion_error}")

# %% [markdown]
# ## 정리
#
# 이 튜토리얼에서는 **Agentic RAG** 시스템을 구축했습니다.
#
# ### 핵심 개념
#
# 1. **Agent**: 도구 사용 여부를 스스로 결정하는 LLM 기반 에이전트입니다.
# 2. **ToolNode**: 도구를 노드로 래핑하여 그래프에서 사용합니다.
# 3. **tools_condition**: 에이전트의 도구 호출 결정을 기반으로 라우팅합니다.
# 4. **문서 평가**: 검색된 문서의 관련성을 평가하여 답변 생성 또는 재검색을 결정합니다.
#
# ### 다음 단계
#
# 다음 튜토리얼에서는 **CRAG(Corrective RAG)**를 구현하여 검색 결과를 정제하고 웹 검색으로 보강하는 방법을 학습합니다.
