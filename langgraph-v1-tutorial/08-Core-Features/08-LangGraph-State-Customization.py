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
# # 사람에게 물어보는 노드
#
# 지금까지는 메시지들의 상태(State) 에 의존해 왔습니다. 
#
# 이런 상태 값들의 수정으로도 많은 것을 할 수 있지만, 메시지 목록에만 의존하지 않고 복잡한 동작을 정의하고자 한다면 상태에 추가 필드를 더할 수 있습니다. 이 튜토리얼에서는 새로운 노드를 추가하여 챗봇을 확장하는 방법을 설명합니다.
#
# 위의 예에서는 도구가 호출될 때마다 **interrupt 를 통해 그래프가 항상 중단** 되도록 Human-in-the-loop 을 구현하였습니다. 
#
# 이번에는, 챗봇이 인간에 의존할지 선택할 수 있도록 하고 싶다고 가정해 봅시다.
#
# 이를 수행하는 한 가지 방법은 그래프가 항상 멈추는 **"human" 노드** 를 생성하는 것입니다. 이 노드는 LLM이 "human" 도구를 호출할 때만 실행됩니다. 편의를 위해, 그래프 상태에 "ask_human" 플래그를 포함시켜 LLM이 이 도구를 호출하면 플래그를 전환하도록 할 것입니다.

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
# ## 사람에게 의견을 묻는 노드 설정
#
# 이 섹션에서는 상태에 `ask_human` 필드를 추가하고, LLM이 `HumanRequest` 도구를 호출할 때 이 플래그를 활성화하는 방법을 구현합니다. 필요한 모듈들을 먼저 임포트합니다.

# %%
from typing import Annotated
from typing_extensions import TypedDict
from langchain_teddynote.tools.tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# %% [markdown]
# ### State 정의
#
# 이번에는 기존 `messages` 필드 외에 `ask_human`이라는 boolean 필드를 상태에 추가합니다. 이 필드는 LLM이 사람에게 도움을 요청하는 도구를 호출했는지 여부를 추적합니다.
#
# 아래 코드에서는 확장된 State 클래스를 정의합니다.

# %%
class State(TypedDict):
    # 메시지 목록
    messages: Annotated[list, add_messages]
    # 사람에게 질문할지 여부를 묻는 상태 추가
    ask_human: bool


# %% [markdown]
# ### HumanRequest 스키마 정의
#
# `HumanRequest`는 Pydantic 모델로 정의되며, LLM이 사람에게 도움을 요청할 때 사용하는 도구입니다. `request` 필드에는 사용자의 요청 내용이 담기며, 이를 통해 전문가가 적절한 안내를 제공할 수 있습니다.
#
# 아래 코드에서는 `HumanRequest` 스키마를 정의합니다.

# %%
from pydantic import BaseModel


class HumanRequest(BaseModel):
    """Forward the conversation to an expert. Use when you can't assist directly or the user needs assistance that exceeds your authority.
    To use this function, pass the user's 'request' so that an expert can provide appropriate guidance.
    """

    request: str


# %% [markdown]
# 다음으로, 챗봇 노드를 정의합니다. 
#
# 여기서 주요 수정 사항은 챗봇이 `HumanRequest` 도구를 호출한 경우 `ask_human` 플래그를 `True`로 설정하는 것입니다. 이를 통해 조건부 라우팅에서 human 노드로 분기할 수 있습니다.
#
# 아래 코드에서는 도구를 바인딩한 LLM과 `ask_human` 플래그를 관리하는 chatbot 노드 함수를 정의합니다.

# %%
from langchain_openai import ChatOpenAI

# 도구 추가
tool = TavilySearch(max_results=3)

# 도구 목록 추가(HumanRequest 도구)
tools = [tool, HumanRequest]

# LLM 추가
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 도구 바인딩
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # LLM 도구 호출을 통한 응답 생성
    response = llm_with_tools.invoke(state["messages"])

    # 사람에게 질문할지 여부 초기화
    ask_human = False

    # 도구 호출이 있고 이름이 'HumanRequest' 인 경우
    if response.tool_calls and response.tool_calls[0]["name"] == HumanRequest.__name__:
        ask_human = True

    # 메시지와 ask_human 상태 반환
    return {"messages": [response], "ask_human": ask_human}


# %% [markdown]
# ### 그래프 빌더 및 노드 추가
#
# 이제 StateGraph를 생성하고 노드들을 추가합니다. `chatbot` 노드는 사용자 입력을 처리하고 도구 호출 여부를 결정하며, `tools` 노드는 ToolNode를 사용하여 실제 도구(TavilySearch)를 실행합니다. 주의할 점은 `HumanRequest`는 실제 실행되는 도구가 아니라 플래그 전환용이므로 ToolNode에는 포함하지 않습니다.
#
# 아래 코드에서는 그래프 빌더를 생성하고 노드들을 추가합니다.

# %%
# 상태 그래프 초기화
graph_builder = StateGraph(State)

# 챗봇 노드 추가
graph_builder.add_node("chatbot", chatbot)

# 도구 노드 추가
graph_builder.add_node("tools", ToolNode(tools=[tool]))

# %% [markdown]
# ## Human 노드 설정
#
# 이 섹션에서는 사람의 개입이 필요할 때 실행되는 `human` 노드를 구현합니다. 이 노드는 그래프에서 인터럽트를 트리거하고, 사람의 응답을 기다린 후 대화를 계속 진행하는 역할을 합니다.

# %% [markdown]
# 다음으로 `human` 노드를 생성합니다. 
#
# 이 노드는 주로 그래프에서 인터럽트를 트리거하는 자리 표시자 역할을 합니다. 사용자가 `interrupt` 동안 수동으로 상태를 업데이트하지 않으면, LLM이 사용자가 요청을 받았지만 응답하지 않았음을 알 수 있도록 도구 메시지를 삽입합니다. 
#
# 이 노드는 또한 `ask_human` 플래그를 해제하여 추가 요청이 없는 한 그래프가 노드를 다시 방문하지 않도록 합니다.

# %% [markdown]
# > 참고 이미지
#
# ![](image/langgraph-07.jpeg)

# %%
from langchain_core.messages import AIMessage, ToolMessage


# 응답 메시지 생성(ToolMessage 생성을 위한 함수)
def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


# 인간 노드 처리
def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # 사람으로부터 응답이 없는 경우
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # 새 메시지 추가
        "messages": new_messages,
        # 플래그 해제
        "ask_human": False,
    }


# 그래프에 인간 노드 추가
graph_builder.add_node("human", human_node)

# %% [markdown]
# 다음으로, 조건부 논리를 정의합니다. 
#
# `select_next_node`는 플래그가 설정된 경우 `human` 노드로 경로를 지정합니다. 그렇지 않으면, 사전 구축된 `tools_condition` 함수가 다음 노드를 선택하도록 합니다.
#
# `tools_condition` 함수는 단순히 `chatbot`이 응답 메시지에서 `tool_calls`을 사용했는지 확인합니다. 
#
# 사용한 경우, `action` 노드로 경로를 지정합니다. 그렇지 않으면, 그래프를 종료합니다.

# %%
from langgraph.graph import END


# 다음 노드 선택
def select_next_node(state: State):
    # 인간에게 질문 여부 확인
    if state["ask_human"]:
        return "human"
    # 이전과 동일한 경로 설정
    return tools_condition(state)


# 조건부 엣지 추가
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)

# %% [markdown]
# ### 그래프 컴파일
#
# 마지막으로 모든 노드를 연결하는 엣지를 추가하고 그래프를 컴파일합니다. `interrupt_before=["human"]`을 설정하여 human 노드 실행 전에 그래프가 중단되도록 합니다. 이를 통해 전문가가 상태를 검토하고 수동으로 응답을 추가할 수 있습니다.
#
# 아래 코드에서는 엣지를 연결하고 체크포인터와 인터럽트 설정을 포함하여 그래프를 컴파일합니다.

# %%
# 엣지 추가: 'tools'에서 'chatbot'으로
graph_builder.add_edge("tools", "chatbot")

# 엣지 추가: 'human'에서 'chatbot'으로
graph_builder.add_edge("human", "chatbot")

# 엣지 추가: START에서 'chatbot'으로
graph_builder.add_edge(START, "chatbot")

# 메모리 저장소 초기화
memory = MemorySaver()

# 그래프 컴파일: 메모리 체크포인터 사용
graph = graph_builder.compile(
    checkpointer=memory,
    # 'human' 이전에 인터럽트 설정
    interrupt_before=["human"],
)

# %% [markdown]
# ### 그래프 시각화
#
# 컴파일된 그래프의 구조를 시각화하여 노드 간 연결과 조건부 분기를 확인합니다. `chatbot`에서 `human`, `tools`, `END`로 향하는 세 가지 경로가 있음을 볼 수 있습니다.
#
# 아래 코드는 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(graph)

# %% [markdown]
# `chatbot` 노드는 다음과 같은 동작을 합니다.
#
# - 챗봇은 인간에게 도움을 요청할 수 있으며 (chatbot->select->human)
# - 검색 엔진 도구를 호출하거나 (chatbot->select->action)
# - 직접 응답할 수 있습니다 (chatbot->select->__end__). 
#
# 일단 행동이나 요청이 이루어지면, 그래프는 `chatbot` 노드로 다시 전환되어 작업을 계속합니다. 

# %%
# user_input = "이 AI 에이전트를 구축하기 위해 전문가의 도움이 필요합니다. 검색해서 답변하세요" (Human 이 아닌 웹검색을 수행하는 경우)
user_input = "이 AI 에이전트를 구축하기 위해 전문가의 도움이 필요합니다. 도움을 요청할 수 있나요?"

# config 설정
config = {"configurable": {"thread_id": "1"}}

# 스트림 또는 호출의 두 번째 위치 인수로서의 구성
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        # 마지막 메시지의 예쁜 출력
        event["messages"][-1].pretty_print()

# %% [markdown]
# **Notice:** `LLM`이 제공된 "`HumanRequest`" 도구를 호출했으며, 인터럽트가 설정되었습니다. 그래프 상태를 확인해 봅시다.

# %%
# 그래프 상태 스냅샷 생성
snapshot = graph.get_state(config)

# 다음 스냅샷 상태 접근
snapshot.next

# %% [markdown]
# 그래프 상태는 실제로 `'human'` 노드 이전에 **중단**됩니다. 
#
# 이 시나리오에서 "전문가"로서 행동하고 입력을 사용하여 새로운 `ToolMessage`를 추가하여 상태를 수동으로 업데이트할 수 있습니다.
#
# 다음으로, 챗봇의 요청에 응답하려면 다음을 수행합니다.
#
# 1. 응답을 포함한 `ToolMessage`를 생성합니다. 이는 `chatbot`에 다시 전달됩니다.
# 2. `update_state`를 호출하여 그래프 상태를 수동으로 업데이트합니다.

# %%
# AI 메시지 추출
ai_message = snapshot.values["messages"][-1]

# 인간 응답 생성
human_response = (
    "전문가들이 도와드리겠습니다! 에이전트 구축을 위해 LangGraph를 확인해 보시기를 적극 추천드립니다. "
    "단순한 자율 에이전트보다 훨씬 더 안정적이고 확장성이 뛰어납니다. "
    "https://wikidocs.net/233785 에서 더 많은 정보를 확인할 수 있습니다."
)

# 도구 메시지 생성
tool_message = create_response(human_response, ai_message)

# 그래프 상태 업데이트
graph.update_state(config, {"messages": [tool_message]})

# %% [markdown]
# 상태를 확인하여 응답이 추가되었는지 확인할 수 있습니다.

# %%
# 그래프 상태에서 메시지 값 가져오기
graph.get_state(config).values["messages"]

# %% [markdown]
# 다음으로, 입력값으로 `None`을 사용하여 그래프를 **resume**합니다.

# %%
# 그래프에서 이벤트 스트림 생성
events = graph.stream(None, config, stream_mode="values")

# 각 이벤트에 대한 처리
for event in events:
    # 메시지가 있는 경우 마지막 메시지 출력
    if "messages" in event:
        event["messages"][-1].pretty_print()

# %% [markdown]
# 최종 결과를 확인합니다.

# %%
# 최종 상태 확인
state = graph.get_state(config)

# 단계별 메시지 출력
for message in state.values["messages"]:
    message.pretty_print()
