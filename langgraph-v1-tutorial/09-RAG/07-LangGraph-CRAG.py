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
# # CRAG: Corrective RAG
#
# 이번 튜토리얼은 **Corrective RAG (CRAG)** 전략을 사용하여 RAG 기반 시스템을 개선하는 방법을 다룹니다.
#
# CRAG는 검색된 문서들에 대한 자기 반성(self-reflection) 및 자기 평가(self-evaluation) 단계를 포함하여, 검색-생성 파이프라인을 정교하게 다루는 접근법입니다.
#
# ![crag](./assets/langgraph-crag.png)
#
# ---
#
# **CRAG란?**
#
# **Corrective-RAG (CRAG)**는 RAG 전략에서 **검색 과정에서 찾아온 문서를 평가하고, 지식을 정제(refine) 하는 단계를 추가한 방법론**입니다.
#
# CRAG의 핵심 아이디어 ([논문 링크](https://arxiv.org/pdf/2401.15884.pdf)):
#
# 1. 검색된 문서 중 하나 이상이 관련성 임계값을 초과하면 생성 단계로 진행
# 2. 생성 전에 지식 정제 단계를 수행
# 3. 문서를 "knowledge strips"로 세분화하여 평가
# 4. 모든 문서가 관련성 임계값 이하이면 웹 검색으로 보강
# 5. 웹 검색 시 쿼리 재작성(Query-Rewrite)을 통해 검색 결과 최적화
#
# ---
#
# **참고**
#
# - [LangGraph CRAG 튜토리얼 (공식 문서)](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/)

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
# ## 검색된 문서의 관련성 평가 (Retrieval Grader)
#
# 검색된 문서가 질문과 관련이 있는지 평가하는 평가기를 생성합니다.

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import get_model_name, LLMs
from pydantic import BaseModel, Field

# 모델 이름 가져오기
MODEL_NAME = get_model_name(LLMs.GPT4o)


# 관련성 평가를 위한 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성을 평가하는 이진 점수입니다."""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'"
    )


# LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# 구조화된 출력을 생성하는 LLM
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 시스템 프롬프트 정의
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Retrieval 평가기 생성
retrieval_grader = grade_prompt | structured_llm_grader

# %% [markdown]
# ### 평가기 테스트

# %%
# 질문 정의
question = "삼성전자가 개발한 생성AI 에 대해 설명하세요."

# 문서 검색
docs = pdf_retriever.invoke(question)

# 검색된 문서 중 1번 index 문서의 관련성 평가
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# %% [markdown]
# ## 답변 생성 체인 (RAG Chain)

# %%
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# RAG 프롬프트 가져오기
prompt = hub.pull("teddynote/rag-prompt")

# LLM 초기화
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)


# 문서 포맷팅 함수
def format_docs(docs):
    """문서 리스트를 포맷팅된 문자열로 변환합니다."""
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata["source"]}</source><page>{doc.metadata["page"]+1}</page></document>'
            for doc in docs
        ]
    )


# RAG 체인 생성
rag_chain = prompt | llm | StrOutputParser()

# 체인 실행 테스트
generation = rag_chain.invoke({"context": format_docs(docs), "question": question})
print(generation)

# %% [markdown]
# ## 쿼리 재작성 (Question Re-writer)

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM 설정
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# Query Rewrite 시스템 프롬프트
system = """You a question re-writer that converts an input question to a better version that is optimized 
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

# 프롬프트 정의
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# Question Re-writer 체인 생성
question_rewriter = re_write_prompt | llm | StrOutputParser()

# %%
# 쿼리 재작성 테스트
print(f'[원본 질문]: "{question}"')
print("[쿼리 재작성]:", question_rewriter.invoke({"question": question}))

# %% [markdown]
# ## 웹 검색 도구

# %%
from langchain_teddynote.tools.tavily import TavilySearch

# 웹 검색 도구 생성 (최대 3개 결과)
web_search_tool = TavilySearch(max_results=3)

# %%
# 웹 검색 테스트
results = web_search_tool.invoke({"query": question})
print(results)

# %% [markdown]
# ## State 정의

# %%
from typing import Annotated, List
from typing_extensions import TypedDict


# 상태 정의 (TypedDict 기반 - LangGraph v1 호환)
class GraphState(TypedDict):
    question: Annotated[str, "질문"]
    generation: Annotated[str, "생성된 답변"]
    web_search: Annotated[str, "웹 검색 필요 여부 (yes/no)"]
    documents: Annotated[List[str], "검색된 문서 리스트"]


# %% [markdown]
# ## 노드 정의

# %%
from langchain.schema import Document


def retrieve(state: GraphState):
    """문서를 검색하는 노드입니다."""
    print("\n==== RETRIEVE ====\n")
    question = state["question"]

    # 문서 검색 수행
    documents = pdf_retriever.invoke(question)
    return {"documents": documents}


def generate(state: GraphState):
    """답변을 생성하는 노드입니다."""
    print("\n==== GENERATE ====\n")
    question = state["question"]
    documents = state["documents"]

    # RAG를 사용한 답변 생성
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}


def grade_documents(state: GraphState):
    """검색된 문서의 관련성을 평가하는 노드입니다."""
    print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
    question = state["question"]
    documents = state["documents"]

    # 필터링된 문서 리스트
    filtered_docs = []
    relevant_doc_count = 0

    for d in documents:
        # Question-Document 관련성 평가
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score

        if grade == "yes":
            print("==== [GRADE: DOCUMENT RELEVANT] ====")
            filtered_docs.append(d)
            relevant_doc_count += 1
        else:
            print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")
            continue

    # 관련 문서가 없으면 웹 검색 수행
    web_search = "Yes" if relevant_doc_count == 0 else "No"
    return {"documents": filtered_docs, "web_search": web_search}


def query_rewrite(state: GraphState):
    """쿼리를 재작성하는 노드입니다."""
    print("\n==== [REWRITE QUERY] ====\n")
    question = state["question"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}


def web_search(state: GraphState):
    """웹 검색을 수행하는 노드입니다."""
    print("\n==== [WEB SEARCH] ====\n")
    question = state["question"]
    documents = state["documents"]

    # 웹 검색 수행
    docs = web_search_tool.invoke({"query": question})
    
    # 검색 결과를 문서 형식으로 변환
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents}


# %% [markdown]
# ## 조건부 엣지 함수

# %%
def decide_to_generate(state: GraphState):
    """평가 결과에 따라 다음 단계를 결정하는 함수입니다."""
    print("==== [ASSESS GRADED DOCUMENTS] ====")
    web_search = state["web_search"]

    if web_search == "Yes":
        # 웹 검색이 필요한 경우
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, QUERY REWRITE] ===="
        )
        return "query_rewrite"
    else:
        # 관련 문서가 있으면 답변 생성
        print("==== [DECISION: GENERATE] ====")
        return "generate"


# %% [markdown]
# ## 그래프 생성

# %%
from langgraph.graph import END, StateGraph, START

# 그래프 상태 초기화
workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("query_rewrite", query_rewrite)
workflow.add_node("web_search_node", web_search)

# 엣지 연결
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 문서 평가 후 조건부 엣지
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "query_rewrite": "query_rewrite",
        "generate": "generate",
    },
)

workflow.add_edge("query_rewrite", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# 그래프 컴파일
app = workflow.compile()

# %% [markdown]
# ### 그래프 시각화

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(app)

# %% [markdown]
# ## 그래프 실행

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid

# config 설정
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

# 질문 입력 (문서에서 검색 가능한 질문)
inputs = {
    "question": "삼성전자가 개발한 생성형 AI 의 이름은?",
}

# 스트리밍 실행
stream_graph(
    app,
    inputs,
    config,
    ["retrieve", "grade_documents", "query_rewrite", "web_search_node", "generate"],
)

# %%
# 질문 입력 (문서에 없는 정보 - 웹 검색 필요)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

inputs = {
    "question": "2024년 노벨문학상 수상자의 이름은?",
}

# %%
# 그래프 실행
invoke_graph(app, inputs, config)

# %%
# 스트리밍 실행
stream_graph(
    app,
    inputs,
    config,
    ["retrieve", "grade_documents", "query_rewrite", "generate"],
)

# %% [markdown]
# ## 정리
#
# 이 튜토리얼에서는 **CRAG(Corrective RAG)** 전략을 구현했습니다.
#
# ### 핵심 개념
#
# 1. **문서 관련성 평가**: 검색된 각 문서의 질문에 대한 관련성을 평가합니다.
# 2. **조건부 웹 검색**: 관련 문서가 없으면 쿼리를 재작성하고 웹 검색을 수행합니다.
# 3. **자기 수정**: 검색 결과를 평가하고 필요시 보완하는 자기 수정 메커니즘입니다.
#
# ### 다음 단계
#
# 다음 튜토리얼에서는 **Self-RAG**를 구현하여 답변의 환각(hallucination)을 검증하는 방법을 학습합니다.
