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
# # 쿼리 재작성 모듈 추가
#
# 이번 튜토리얼에서는 RAG 파이프라인에 **쿼리 재작성(Query Rewrite)** 기능을 추가합니다.
#
# **학습 목표**
#
# - 검색 품질 향상을 위한 쿼리 재작성 방법을 학습합니다.
# - 재작성된 쿼리를 사용하여 더 관련성 높은 문서를 검색합니다.
# - 쿼리 히스토리를 관리하는 방법을 익힙니다.
#
# **참고**
#
# - 이전 튜토리얼에서 확장된 내용이므로, 겹치는 부분은 간략히 설명합니다.
#
# ![langgraph-query-rewrite](assets/langgraph-query-rewrite.png)

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
#
# 이번에는 `question`을 **리스트 형식**으로 정의합니다. 재작성된 쿼리를 누적하여 저장하기 위함입니다.

# %%
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages


# GraphState 상태 정의 (TypedDict 기반 - LangGraph v1 호환)
class GraphState(TypedDict):
    question: Annotated[List[str], add_messages]  # 질문 리스트 (원본 + 재작성)
    context: Annotated[str, "Context"]  # 검색된 문서
    answer: Annotated[str, "Answer"]  # 생성된 답변
    messages: Annotated[list, add_messages]  # 대화 히스토리 (누적)
    relevance: Annotated[str, "Relevance"]  # 관련성 체크 결과


# %% [markdown]
# ## 노드(Node) 정의
#
# 기존 노드에 **쿼리 재작성 노드**를 추가합니다.

# %%
from langchain_openai import ChatOpenAI
from langchain_teddynote.evaluator import GroundednessChecker
from langchain_teddynote.messages import messages_to_history
from langchain_teddynote.tools.tavily import TavilySearch
from rag.utils import format_docs


def retrieve_document(state: GraphState) -> GraphState:
    """문서를 검색하는 노드입니다."""
    # 가장 최근 질문 사용 (재작성된 질문이 있으면 그것을 사용)
    latest_question = state["question"][-1].content

    retrieved_docs = pdf_retriever.invoke(latest_question)
    retrieved_docs = format_docs(retrieved_docs)

    return {"context": retrieved_docs}


def llm_answer(state: GraphState) -> GraphState:
    """답변을 생성하는 노드입니다."""
    latest_question = state["question"][-1].content
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
        {"question": state["question"][-1].content, "context": state["context"]}
    )

    return {"relevance": response.score}


def is_relevant(state: GraphState) -> str:
    """조건부 엣지에서 사용되는 라우팅 함수입니다."""
    if state["relevance"] == "yes":
        return "relevant"
    else:
        return "not relevant"


def web_search(state: GraphState) -> GraphState:
    """웹 검색을 수행하는 노드입니다."""
    tavily_tool = TavilySearch()

    search_query = state["question"]

    search_result = tavily_tool.search(
        query=search_query,
        topic="general",
        max_results=6,
        format_output=True,
    )

    return {"context": search_result}


# %% [markdown]
# ## 쿼리 재작성 노드 추가
#
# 쿼리를 벡터 검색에 최적화된 형태로 재작성하는 노드를 추가합니다.

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Query Rewrite 프롬프트 정의
re_write_prompt = PromptTemplate(
    template="""Reformulate the given question to enhance its effectiveness for vectorstore retrieval.

- Analyze the initial question to identify areas for improvement such as specificity, clarity, and relevance.
- Consider the context and potential keywords that would optimize retrieval.
- Maintain the intent of the original question while enhancing its structure and vocabulary.

# Steps

1. **Understand the Original Question**: Identify the core intent and any keywords.
2. **Enhance Clarity**: Simplify language and ensure the question is direct and to the point.
3. **Optimize for Retrieval**: Add or rearrange keywords for better alignment with vectorstore indexing.
4. **Review**: Ensure the improved question accurately reflects the original intent and is free of ambiguity.

# Output Format

- Provide a single, improved question.
- Do not include any introductory or explanatory text; only the reformulated question.

# Examples

**Input**:
"What are the benefits of using renewable energy sources over fossil fuels?"

**Output**:
"How do renewable energy sources compare to fossil fuels in terms of benefits?"

**Input**:
"How does climate change impact polar bear populations?"

**Output**:
"What effects does climate change have on polar bear populations?"

# Notes

- Ensure the improved question is concise and contextually relevant.
- Avoid altering the fundamental intent or meaning of the original question.


[REMEMBER] Re-written question should be in the same language as the original question.

# Here is the original question that needs to be rewritten:
{question}
""",
    input_variables=["question"],
)

# 쿼리 재작성 체인 생성
question_rewriter = (
    re_write_prompt
    | ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    | StrOutputParser()
)

# %% [markdown]
# ### 쿼리 재작성 테스트

# %%
# 질문 재작성 테스트
question = "앤스로픽에 투자한 미국기업"

rewritten = question_rewriter.invoke({"question": question})
print(f"Original: {question}")
print(f"Rewritten: {rewritten}")


# %%
def query_rewrite(state: GraphState) -> GraphState:
    """쿼리를 재작성하는 노드입니다.
    
    검색 품질 향상을 위해 사용자 질문을 최적화된 형태로 재작성합니다.
    """
    latest_question = state["question"][-1].content
    question_rewritten = question_rewriter.invoke({"question": latest_question})
    return {"question": question_rewritten}


# %% [markdown]
# ## 그래프 생성
#
# 쿼리 재작성을 그래프의 시작 노드로 추가합니다.

# %%
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 그래프 정의
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("llm_answer", llm_answer)
workflow.add_node("web_search", web_search)
workflow.add_node("query_rewrite", query_rewrite)  # 쿼리 재작성 노드 추가

# 엣지 추가
workflow.add_edge("query_rewrite", "retrieve")  # 쿼리 재작성 -> 검색
workflow.add_edge("retrieve", "relevance_check")  # 검색 -> 관련성 체크

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "relevance_check",
    is_relevant,
    {
        "relevant": "llm_answer",
        "not relevant": "web_search",
    },
)

workflow.add_edge("web_search", "llm_answer")
workflow.add_edge("llm_answer", END)

# 그래프 진입점 설정 (쿼리 재작성부터 시작)
workflow.set_entry_point("query_rewrite")

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

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid

# config 설정
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

# 질문 입력 (간략한 질문 -> 쿼리 재작성으로 개선)
inputs = GraphState(question="앤스로픽 투자 금액")

# 그래프 실행
invoke_graph(app, inputs, config)

# %%
# 그래프 스트리밍 실행
stream_graph(app, inputs, config, ["query_rewrite", "llm_answer"])

# %% [markdown]
# ### 결과 확인

# %%
# 최종 출력 확인
outputs = app.get_state(config).values

print(f'Original Question: {outputs["question"][0].content}')
print(f'Rewritten Question: {outputs["question"][-1].content}')
print("===" * 20)
print(f'Answer:\n{outputs["answer"]}')

# %% [markdown]
# ## 정리
#
# 이 튜토리얼에서는 RAG 파이프라인에 **쿼리 재작성** 기능을 추가했습니다.
#
# ### 핵심 개념
#
# 1. **Query Rewrite**: 사용자 질문을 벡터 검색에 최적화된 형태로 재작성합니다.
# 2. **쿼리 히스토리**: 리스트 형식으로 원본 및 재작성된 쿼리를 관리합니다.
# 3. **검색 품질 향상**: 재작성된 쿼리로 더 관련성 높은 문서를 검색합니다.
#
# ### 다음 단계
#
# 다음 튜토리얼에서는 **Agentic RAG**를 구현하여 에이전트가 검색 도구 사용 여부를 스스로 결정하는 방법을 학습합니다.
