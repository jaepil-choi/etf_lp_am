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
# # LangGraph 챗봇 구축
#
# 먼저 `LangGraph`를 사용하여 간단한 챗봇을 만들어 보겠습니다. 이 챗봇은 사용자 메시지에 직접 응답할 것입니다. 비록 간단하지만, `LangGraph`로 구축하는 핵심 개념을 설명할 것입니다. 이 섹션이 끝나면 기본적인 챗봇을 구축하게 될 것입니다.
#
# `StateGraph`를 생성하는 것으로 시작하십시오. `StateGraph` 객체는 챗봇의 구조를 "상태 기계(State Machine)"로 정의합니다. 
#
# `nodes`를 추가하여 챗봇이 호출할 수 있는 `llm`과 함수들을 나타내고, `edges`를 추가하여 봇이 이러한 함수들 간에 어떻게 전환해야 하는지를 지정합니다.

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
# ## Step-by-Step 개념 이해하기!

# %% [markdown]
# ### STEP 1. 상태(State) 정의
#
# LangGraph에서 **State(상태)**는 그래프의 모든 노드가 공유하는 데이터 구조입니다. `TypedDict`를 사용하여 정의하며, 각 노드는 이 상태를 읽고 업데이트합니다.
#
# `messages` 필드는 `Annotated` 타입과 `add_messages` 리듀서를 사용하여 정의합니다. 이렇게 하면 새로운 메시지가 기존 메시지 목록에 자동으로 추가됩니다.
#
# > 📖 **참고 문서**: [LangGraph Graph API](https://langchain-ai.github.io/langgraph/)
#
# 아래 코드에서는 챗봇의 상태를 정의합니다.

# %%
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    messages: Annotated[list, add_messages]


# %% [markdown]
# ### STEP 2. 노드(Node) 정의

# %% [markdown]
# 노드(Node)는 그래프에서 실제 작업을 수행하는 단위입니다. 일반적으로 Python 함수로 정의되며, 현재 상태(State)를 입력으로 받아 처리한 후 업데이트된 상태를 반환합니다.
#
# 챗봇 노드 함수는 상태에서 메시지를 읽어 LLM에 전달하고, LLM의 응답을 새 메시지로 반환합니다. 반환된 메시지는 `add_messages` 리듀서에 의해 기존 메시지 목록에 자동으로 추가됩니다.
#
# 아래 코드에서는 LLM을 정의하고 챗봇 노드 함수를 구현합니다.

# %%
from langchain_openai import ChatOpenAI

# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 챗봇 함수 정의
def chatbot(state: State):
    # 메시지 호출 및 반환
    return {"messages": [llm.invoke(state["messages"])]}


# %% [markdown]
# ### STEP 3. 그래프(Graph) 정의, 노드 추가
#
# `StateGraph`는 LangGraph의 핵심 클래스로, 상태 기반 워크플로우를 정의합니다. 정의한 `State` 타입을 인자로 전달하여 그래프 빌더를 생성합니다.
#
# `add_node()` 메서드를 사용하여 그래프에 노드를 추가합니다. 첫 번째 인자는 노드의 이름(문자열)이고, 두 번째 인자는 해당 노드에서 실행될 함수입니다.
#
# 아래 코드에서는 StateGraph를 생성하고 "chatbot" 노드를 추가합니다.

# %%
# 그래프 생성
graph_builder = StateGraph(State)

# 노드 이름, 함수 혹은 callable 객체를 인자로 받아 노드를 추가
graph_builder.add_node("chatbot", chatbot)

# %% [markdown]
# **참고**
#
# - `chatbot` 노드 함수는 현재 `State`를 입력으로 받아 "messages"라는 키 아래에 업데이트된 `messages` 목록을 포함하는 사전(TypedDict) 을 반환합니다. 
#
# - `State`의 `add_messages` 함수는 이미 상태에 있는 메시지에 llm의 응답 메시지를 추가합니다. 

# %% [markdown]
# ### STEP 4. 그래프 엣지(Edge) 추가
#
# 다음으로, `START` 지점을 추가하세요. `START`는 그래프가 실행될 때마다 **작업을 시작할 위치** 입니다.

# %%
# 시작 노드에서 챗봇 노드로의 엣지 추가
graph_builder.add_edge(START, "chatbot")

# %% [markdown]
#
# 마찬가지로, `END` 지점을 설정하십시오. 이는 그래프 흐름의 종료(끝지점) 를 나타냅니다.

# %%
# 그래프에 엣지 추가
graph_builder.add_edge("chatbot", END)

# %% [markdown]
# ### STEP 5. 그래프 컴파일(compile)
#
# `StateGraph`를 정의한 후에는 반드시 `compile()` 메서드를 호출하여 실행 가능한 형태로 변환해야 합니다. 컴파일 과정에서 노드 간 연결이 검증되고, 실행 순서가 결정됩니다.
#
# 컴파일된 그래프는 `invoke()` 또는 `stream()` 메서드를 통해 실행할 수 있습니다.
#
# 아래 코드는 정의한 그래프를 컴파일하고, 실행 가능한 `CompiledGraph` 객체를 반환합니다.

# %%
# 그래프 컴파일
graph = graph_builder.compile()

# %% [markdown]
# ### STEP 6. 그래프 시각화
#
# `langchain_teddynote.graphs` 모듈의 `visualize_graph()` 함수를 사용하면 컴파일된 그래프의 구조를 시각적으로 확인할 수 있습니다. 
#
# 시각화를 통해 노드 간의 연결 관계와 실행 흐름을 직관적으로 파악할 수 있습니다. 복잡한 그래프를 디버깅하거나 구조를 검토할 때 유용합니다.
#
# 아래 코드에서는 컴파일된 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(graph)

# %% [markdown]
# ### STEP 7. 그래프 실행
#
# 컴파일된 그래프는 `stream()` 메서드를 사용하여 실행할 수 있습니다. `stream()`은 각 노드의 실행 결과를 순차적으로 반환하여 실시간으로 처리할 수 있게 합니다.
#
# 입력은 딕셔너리 형태로 전달하며, `messages` 키에 사용자 메시지 튜플 `("user", 질문내용)`을 리스트로 감싸서 전달합니다.
#
# > **Tip**: 일회성 실행이 필요한 경우 `invoke()` 메서드를 사용할 수 있으며, 스트리밍이 필요한 경우 `stream()` 메서드를 사용합니다.
#
# 아래 코드에서는 챗봇 그래프를 실행하고 응답을 출력합니다.

# %%
question = "서울의 유명한 맛집 TOP 10 추천해줘"

# 그래프 이벤트 스트리밍
for event in graph.stream({"messages": [("user", question)]}):
    # 이벤트 값 출력
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)

# %% [markdown]
# 자! 여기까지가 가장 기본적인 챗봇 구축이었습니다. 
#
# 아래는 이전 과정을 정리한 전체 코드입니다.

# %% [markdown]
# ## 전체 코드

# %%
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_teddynote.graphs import visualize_graph


###### STEP 1. 상태(State) 정의 ######
class State(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    messages: Annotated[list, add_messages]


###### STEP 2. 노드(Node) 정의 ######
# LLM 정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 챗봇 함수 정의
def chatbot(state: State):
    # 메시지 호출 및 반환
    return {"messages": [llm.invoke(state["messages"])]}


###### STEP 3. 그래프(Graph) 정의, 노드 추가 ######
# 그래프 생성
graph_builder = StateGraph(State)

# 노드 이름, 함수 혹은 callable 객체를 인자로 받아 노드를 추가
graph_builder.add_node("chatbot", chatbot)

###### STEP 4. 그래프 엣지(Edge) 추가 ######
# 시작 노드에서 챗봇 노드로의 엣지 추가
graph_builder.add_edge(START, "chatbot")

# 그래프에 엣지 추가
graph_builder.add_edge("chatbot", END)

###### STEP 5. 그래프 컴파일(compile) ######
# 그래프 컴파일
graph = graph_builder.compile()

###### STEP 6. 그래프 시각화 ######
# 그래프 시각화
visualize_graph(graph)

###### STEP 7. 그래프 실행 ######
question = "서울의 유명한 맛집 TOP 10 추천해줘"

# 그래프 이벤트 스트리밍
for event in graph.stream({"messages": [("user", question)]}):
    # 이벤트 값 출력
    for value in event.values():
        print(value["messages"][-1].content)
