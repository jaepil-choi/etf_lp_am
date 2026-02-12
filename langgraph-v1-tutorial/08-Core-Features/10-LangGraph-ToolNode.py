# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: langchain-kr-lwwSZlnu-py3.11
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ToolNode 를 사용하여 도구를 호출하는 방법
#
# 이번 튜토리얼에서는 도구 호출을 위한 LangGraph의 사전 구축된 `pre-built`의 `ToolNode` 사용 방법을 다룹니다.
#
# `ToolNode`는 메시지 목록이 포함된 그래프 상태를 입력으로 받아 도구 호출 결과로 상태를 업데이트하는 LangChain Runnable입니다. 
#
# 이는 LangGraph의 사전 구축된 Agent 와 즉시 사용할 수 있도록 설계되었으며, 상태에 적절한 리듀서가 있는 `messages` 키가 포함된 경우 모든 `StateGraph` 와 함께 작동할 수 있습니다.

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# # !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-V1-Tutorial")

# %% [markdown]
# ## 도구 정의
#
# `ToolNode`에서 사용할 도구들을 정의합니다. LangChain의 `@tool` 데코레이터를 사용하면 일반 Python 함수를 도구로 변환할 수 있습니다. 각 도구에는 docstring으로 설명을 작성하여 LLM이 언제 해당 도구를 사용할지 판단할 수 있도록 합니다.
#
# 아래 코드에서는 Google 뉴스 검색 도구와 Python 코드 실행 도구를 정의합니다.

# %%
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_teddynote.tools import GoogleNews
from typing import List, Dict


# 도구 생성
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


@tool
def python_code_interpreter(code: str):
    """Call to execute python code."""
    return PythonAstREPLTool().invoke(code)


# %% [markdown]
# ### ToolNode 초기화
#
# 정의한 도구들을 리스트로 묶어 `ToolNode`에 전달합니다. `ToolNode`는 이 도구 리스트를 받아 메시지의 `tool_calls` 정보에 따라 적절한 도구를 실행하는 역할을 합니다.
#
# 아래 코드에서는 도구 리스트를 생성하고 `ToolNode`를 초기화합니다.

# %%
from langgraph.prebuilt import ToolNode, tools_condition

# 도구 리스트 생성
tools = [search_news, python_code_interpreter]

# ToolNode 초기화
tool_node = ToolNode(tools)

# %% [markdown]
# ## ToolNode 수동 호출
#
# `ToolNode`를 직접 호출하여 도구 실행을 테스트할 수 있습니다. 이 방식은 그래프 없이 도구의 동작을 빠르게 확인하거나 디버깅할 때 유용합니다.

# %% [markdown]
# `ToolNode`는 메시지 목록과 함께 그래프 상태에서 작동합니다. 
#
# - **중요**: 이때 목록의 마지막 메시지는 `tool_calls` 속성을 포함하는 `AIMessage`여야 합니다.
#
# 먼저 도구 노드를 수동으로 호출하는 방법을 살펴보겠습니다.

# %%
from langchain_core.messages import AIMessage

# 단일 도구 호출을 포함하는 AI 메시지 객체 생성
# AIMessage 객체이어야 함
message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "search_news",  # 도구 이름
            "args": {"query": "AI"},  # 도구 인자
            "id": "tool_call_id",  # 도구 호출 ID
            "type": "tool_call",  # 도구 호출 유형
        }
    ],
)

# 도구 노드를 통한 메시지 처리 및 날씨 정보 요청 실행
tool_node.invoke({"messages": [message_with_single_tool_call]})

# %% [markdown]
# 일반적으로 `AIMessage`를 수동으로 생성할 필요가 없으며, 도구 호출을 지원하는 모든 LangChain 채팅 모델에서 자동으로 생성됩니다.
#
# 또한 `AIMessage`의 `tool_calls` 매개변수에 여러 도구 호출을 전달하면 `ToolNode`를 사용하여 병렬 도구 호출을 수행할 수 있습니다.

# %%
# 다중 도구 호출을 포함하는 AI 메시지 객체 생성 및 초기화
message_with_multiple_tool_calls = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "search_news",
            "args": {"query": "AI"},
            "id": "tool_call_id",
            "type": "tool_call",
        },
        {
            "name": "python_code_interpreter",
            "args": {"code": "print(1+2+3+4)"},
            "id": "tool_call_id",
            "type": "tool_call",
        },
    ],
)

# 생성된 메시지를 도구 노드에 전달하여 다중 도구 호출 실행
tool_node.invoke({"messages": [message_with_multiple_tool_calls]})

# %% [markdown]
# ## LLM과 함께 사용하기
#
# 실제 애플리케이션에서는 `AIMessage`를 수동으로 생성하지 않고, LLM이 자동으로 생성한 `tool_calls`를 사용합니다. LLM이 도구를 호출할 수 있도록 하려면 먼저 `bind_tools()` 메서드로 사용 가능한 도구들을 알려주어야 합니다.

# %% [markdown]
# 도구 호출 기능이 있는 채팅 모델을 사용하기 위해서는 먼저 모델이 사용 가능한 도구들을 인식하도록 해야 합니다. 
#
# 이는 `ChatOpenAI` 모델에서 `.bind_tools` 메서드를 호출하여 수행합니다.

# %%
from langchain_openai import ChatOpenAI

# LLM 모델 초기화 및 도구 바인딩
model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# %%
# 도구 호출 확인
model_with_tools.invoke("처음 5개의 소수를 출력하는 python code 를 작성해줘").tool_calls

# %% [markdown]
# 보시다시피 채팅 모델이 생성한 AI 메시지에는 이미 `tool_calls`가 채워져 있으므로, 이를 `ToolNode`에 직접 전달할 수 있습니다.

# %%
# 도구 노드를 통한 메시지 처리 및 LLM 모델의 도구 기반 응답 생성
tool_node.invoke(
    {
        "messages": [
            model_with_tools.invoke(
                "처음 5개의 소수를 출력하는 python code 를 작성해줘"
            )
        ]
    }
)

# %% [markdown]
# ## Agent 그래프와 함께 사용하기
#
# `ToolNode`의 가장 일반적인 사용 사례는 LangGraph 에이전트 그래프 내에서 도구 실행 노드로 활용하는 것입니다. 에이전트는 사용자 질문을 받아 필요에 따라 도구를 호출하고, 그 결과를 바탕으로 최종 응답을 생성합니다.

# %% [markdown]
# ### 에이전트 그래프 구현
#
# 다음으로, StateGraph를 사용하여 에이전트 그래프를 구현합니다. 이 에이전트는 쿼리를 입력으로 받아, 필요한 정보를 얻을 때까지 반복적으로 도구들을 호출합니다. `tools_condition`은 LLM 응답에 도구 호출이 있는지 확인하여 다음 노드를 결정하는 사전 구축된 조건 함수입니다.
#
# 아래 코드에서는 에이전트 그래프를 정의하고 컴파일합니다.

# %%
# LangGraph 워크플로우 상태 및 메시지 처리를 위한 타입 임포트
from langgraph.graph import StateGraph, MessagesState, START, END


# LLM 모델을 사용하여 메시지 처리 및 응답 생성, 도구 호출이 포함된 응답 반환
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# 메시지 상태 기반 워크플로우 그래프 초기화
workflow = StateGraph(MessagesState)

# 에이전트와 도구 노드 정의 및 워크플로우 그래프에 추가
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 워크플로우 시작점에서 에이전트 노드로 연결
workflow.add_edge(START, "agent")

# 에이전트 노드에서 조건부 분기 설정, 도구 노드 또는 종료 지점으로 연결
workflow.add_conditional_edges("agent", tools_condition)

# 도구 노드에서 에이전트 노드로 순환 연결
workflow.add_edge("tools", "agent")

# 에이전트 노드에서 종료 지점으로 연결
workflow.add_edge("agent", END)


# 정의된 워크플로우 그래프 컴파일 및 실행 가능한 애플리케이션 생성
app = workflow.compile()

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(app)

# %% [markdown]
# ### 에이전트 실행 테스트
#
# 구현한 에이전트를 다양한 질문으로 테스트해봅니다. Python 코드 작성 요청, 뉴스 검색 요청, 그리고 도구 호출이 필요 없는 일반 대화까지 에이전트가 적절히 처리하는지 확인합니다.
#
# 아래 코드에서는 에이전트를 실행하고 결과를 출력합니다.

# %%
# 실행 및 결과 확인
for chunk in app.stream(
    {"messages": [("human", "처음 5개의 소수를 출력하는 python code 를 작성해줘")]},
    stream_mode="values",
):
    # 마지막 메시지 출력
    chunk["messages"][-1].pretty_print()

# %%
# 검색 질문 수행
for chunk in app.stream(
    {"messages": [("human", "search google news about AI")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

# %%
# 도구 호출이 필요 없는 질문 수행
for chunk in app.stream(
    {"messages": [("human", "안녕? 반가워")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

# %% [markdown]
# ## 도구 오류 처리
#
# `ToolNode`는 도구 실행 중 발생하는 오류도 자동으로 처리합니다. 기본적으로 `handle_tool_errors=True`가 설정되어 있어, 도구에서 예외가 발생하면 오류 메시지를 `ToolMessage`로 변환하여 LLM에게 전달합니다. 이를 통해 LLM이 오류 상황을 인식하고 다른 접근 방식을 시도할 수 있습니다.
#
# 오류 처리를 비활성화하려면 `ToolNode(tools, handle_tool_errors=False)`로 설정하면 됩니다.
