# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 에이전트 대화 시뮬레이션(고객 응대 시나리오)
#
# 챗봇을 구축할 때, 예를 들어 고객 지원 어시스턴트와 같은 경우, 챗봇의 성능을 제대로 평가하는 것이 어려울 수 있습니다. 코드 변경마다 집중적으로 수동으로 상호 작용하는 것은 시간이 많이 소요됩니다.
#
# 평가 과정을 더 쉽고 재현 가능하게 만드는 한 가지 방법은 **사용자 상호 작용을 시뮬레이션하는 것** 입니다.
#
# LangGraph를 사용하면 이를 설정하는 것이 쉽습니다. 
#
# 아래는 대화를 시뮬레이션하기 위해 "가상 사용자(Simulated User)"를 생성하는 방법의 예시입니다.
#
# ![agent-simulations.png](assets/agent-simulations.png)

# %% [markdown]
# ## 환경 설정
#
# LangGraph를 사용하여 에이전트 시뮬레이션을 구현하기 위해 필요한 환경을 설정합니다. API 키를 환경변수로 관리하고, LangSmith를 통해 실행 과정을 추적할 수 있도록 설정합니다.
#
# 아래 코드는 `.env` 파일에서 API 키를 로드하고 LangSmith 추적을 활성화합니다.

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-Use-Cases")

# %% [markdown]
# ## 상태(State) 정의
#
# LangGraph에서 그래프의 상태(State)는 노드 간에 전달되는 데이터를 정의합니다. 에이전트 시뮬레이션에서는 사용자와 상담사 간의 대화 메시지를 상태로 관리합니다.
#
# `add_messages` 리듀서를 사용하면 새로운 메시지가 기존 메시지 리스트에 자동으로 추가됩니다. 아래 코드는 시뮬레이션에서 사용할 State 클래스를 정의합니다.

# %%
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


# State 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 사용자 - 상담사 간의 대화 메시지


# %% [markdown]
# ## 상담사, 고객 역할 정의
#
# ### 상담사 역할 정의
#
# 시뮬레이션에서 **상담사** 역할을 하는 챗봇을 정의합니다.
#
# **참고**
#
# - `call_chatbot` 내의 구현은 설정 가능하며, 내부에서 사용한 모델을 Agent 로 변경하는 것도 가능합니다.
# - `call_chatbot` 은 사용자로부터 메시지를 입력으로 받아, 고객을 상담하는 역할을 부여하겠습니다. 
#
# *고객 지원 시나리오에서의 대화 응답 생성에 활용될 수 있습니다.*

# %%
from typing import List
from langchain_teddynote.models import LLMs, get_model_name
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

# 모델 이름 설정: GPT-4o 사용 권장
MODEL_NAME = get_model_name(LLMs.GPT4o)


def call_chatbot(messages: List[BaseMessage]) -> dict:
    """상담사 역할의 챗봇 호출 함수
    
    사용자 메시지를 받아 항공사 고객 지원 상담사로서 응답을 생성합니다.
    LangChain ChatOpenAI 모델을 Agent로 변경하여 확장할 수 있습니다.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer support agent for an airline. Answer in Korean.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=0.6)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"messages": messages})


# %% [markdown]
# `call_chatbot` 함수는 사용자의 입력을 받아 챗봇의 응답을 처리합니다. 아래 코드는 간단한 인사말을 입력으로 전달하여 챗봇의 응답을 확인합니다.

# %%
call_chatbot([("user", "안녕하세요?")])

# %% [markdown]
# ### 고객 역할(Simulated User) 정의
#
# 이제 시뮬레이션된 고객의 역할을 정의합니다. 고객 지원 시나리오에서의 대화를 시뮬레이션합니다. 
#
# 시스템 프롬프트는 고객과 고객 지원 담당자 간의 상호작용을 설정하며, 사용자 지시사항을 통해 시나리오의 세부 사항을 제공합니다. 
#
# 이 구성은 특정 사용자 요구(예: 환불 요청)에 대한 모델의 반응을 시뮬레이션하는 데 사용됩니다.

# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def create_scenario(name: str, instructions: str):
    # 시스템 프롬프트를 정의: 필요에 따라 변경
    system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

Your name is {name}.

# Instructions:
{instructions}

[IMPORTANT] 
- When you are finished with the conversation, respond with a single word 'FINISHED'
- You must speak in Korean."""

    # 대화 메시지와 시스템 프롬프트를 결합하여 채팅 프롬프트 템플릿을 생성합니다.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # 특정 사용자 이름과 지시사항을 사용하여 프롬프트를 부분적으로 채웁니다.
    prompt = prompt.partial(name=name, instructions=instructions)
    return prompt


# %% [markdown]
# 가상의 시나리오를 생성합니다. 이 가상의 시나리오는 고객의 입장에서의 시나리오입니다.
#
# 여기서는 환불을 요청하는 시나리오를 정의합니다.
#

# %%
# 사용자 지시사항을 정의합니다.
instructions = """You are trying to get a refund for the trip you took to Jeju Island. \
You want them to give you ALL the money back. This trip happened last year."""

# 사용자 이름을 정의합니다.
name = "Teddy"

create_scenario(name, instructions).pretty_print()

# %%
# OpenAI 챗봇 모델을 초기화합니다.
model = ChatOpenAI(model=MODEL_NAME, temperature=0.6)

# 시뮬레이션된 사용자 대화를 생성합니다.
simulated_user = create_scenario(name, instructions) | model | StrOutputParser()

# %% [markdown]
# 생성된 `simulated_user` 를 호출하여 시뮬레이션된 사용자에게 메시지를 전달합니다.

# %%
from langchain_core.messages import HumanMessage

# 시뮬레이션된 사용자에게 메시지를 전달(상담사 -> 고객)
messages = [HumanMessage(content="안녕하세요? 어떻게 도와 드릴까요?")]
simulated_user.invoke({"messages": messages})

# %% [markdown]
# ## 에이전트 시뮬레이션 정의하기
#
# 아래의 코드는 시뮬레이션을 실행하기 위한 LangGraph 워크플로우를 생성합니다. 
#
# 주요 구성 요소는 다음과 같습니다:
#
# 1. 시뮬레이션된 사용자와 챗봇을 위한 두 개의 노드입니다.
# 2. 조건부 정지 기준을 가진 그래프 자체입니다.

# %% [markdown]
# ### 노드 정의
#
# 먼저, 그래프에서 노드를 정의합니다. 이들은 메시지 목록을 입력으로 받아 상태에 추가할 메시지 목록을 반환해야 합니다.
# 이것들은 위에 있는 챗봇과 시뮬레이션된 사용자를 둘러싼 것 래퍼들입니다.
#
# **참고:** 여기서 까다로운 점은 어떤 메시지가 어떤 것인지 구분하는 것입니다. 
#
# 챗봇과 시뮬레이션된 사용자 모두 LLMs이기 때문에, 둘 다 AI 메시지로 응답할 것입니다. 우리의 상태는 인간과 AI 메시지가 번갈아 가며 나열된 목록이 될 것입니다. 이는 노드 중 하나에서 AI와 인간 역할을 바꾸는 논리가 필요함을 의미합니다. 
#
# 이 예제에서는, HumanMessages가 시뮬레이션된 사용자로부터 온 메시지라고 가정할 것입니다. 이는 시뮬레이션된 사용자 노드에 AI와 Human 메시지를 교환하는 논리가 필요함을 의미합니다.

# %%
from langchain_core.messages import AIMessage


# 상담사 역할
def ai_assistant_node(messages):
    # 상담사 응답 호출
    ai_response = call_chatbot(messages)

    # AI 상담사의 응답을 반환
    return {"messages": [("assistant", ai_response)]}


# %% [markdown]
# 상담사 역할의 노드를 호출합니다.

# %%
ai_assistant_node(
    [
        ("user", "안녕하세요?"),
        ("assistant", "안녕하세요! 어떻게 도와드릴까요?"),
        ("user", "환불 어떻게 하나요?"),
    ]
)


# %% [markdown]
# 다음으로, 우리의 시뮬레이션된 사용자를 위한 노드를 정의해 보겠습니다.
#
# **참고**
#
# - 이 과정에서는 메시지의 역할을 교체하는 작은 로직이 포함될 것입니다.

# %%
def _swap_roles(messages):
    # 메시지의 역할을 교환: 시뮬레이션 사용자 단계에서 메시지 타입을 AI -> Human, Human -> AI 로 교환합니다.
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            # AIMessage 인 경우, HumanMessage 로 변환합니다.
            new_messages.append(HumanMessage(content=m.content))
        else:
            # HumanMessage 인 경우, AIMessage 로 변환합니다.
            new_messages.append(AIMessage(content=m.content))
    return new_messages


# 상담사 역할(AI Assistant) 노드 정의
def ai_assistant_node(state: State):
    # 상담사 응답 호출
    ai_response = call_chatbot(state["messages"])

    # AI 상담사의 응답을 반환
    return {"messages": [("assistant", ai_response)]}


# 시뮬레이션된 사용자(Simulated User) 노드 정의
def simulated_user_node(state: State):
    # 메시지 타입을 교환: AI -> Human, Human -> AI
    new_messages = _swap_roles(state["messages"])

    # 시뮬레이션된 사용자를 호출
    response = simulated_user.invoke({"messages": new_messages})
    return {"messages": [("user", response)]}


# %% [markdown]
# ### 엣지 정의
#
# 이제 엣지에 대한 로직을 정의할 필요가 있습니다. 주된 로직은 시뮬레이션된 사용자가 작업을 마친 후 발생하며, 두 가지 결과 중 하나로 이어져야 합니다:
#
# - 고객 지원 봇을 호출하여 계속 진행("continue")
# - 대화를 마치고 종료("end")
#
# 그렇다면 대화가 종료되는 로직은 무엇일까요? 우리는 이를 인간 챗봇이 `FINISHED`로 응답하거나(시스템 프롬프트 참조) 대화가 6개 메시지를 초과하는 경우로 정의할 것입니다 (이는 이 예제를 짧게 유지하기 위한 임의의 숫자입니다).

# %% [markdown]
# `should_continue` 함수는 메시지 리스트를 인자로 받아, 리스트의 길이가 6을 초과하거나 마지막 메시지의 내용이 'FINISHED'일 경우 'end'를 반환합니다. 
#
# 그렇지 않으면 'continue'를 반환하여 처리를 계속하도록 합니다.

# %%
def should_continue(state: State):
    # 메시지 리스트의 길이가 6보다 크면 'end'를 반환합니다.
    if len(state["messages"]) > 6:
        return "end"
    # 마지막 메시지의 내용이 'FINISHED'라면 'end'를 반환합니다.
    elif state["messages"][-1].content == "FINISHED":
        return "end"
    # 위의 조건에 해당하지 않으면 'continue'를 반환합니다.
    else:
        return "continue"


# %% [markdown]
# ## 그래프 정의
#
# 이제 시뮬레이션을 설정하는 그래프를 정의합니다. `StateGraph` 클래스를 사용하여 챗봇과 시뮬레이션된 사용자 간의 상호작용을 구성합니다.
#
# 그래프는 다음과 같은 구조로 구성됩니다:
# - **ai_assistant 노드**: 상담사 역할을 수행하며 고객의 질문에 응답합니다.
# - **simulated_user 노드**: 고객 역할을 수행하며 상담사에게 질문을 합니다.
# - **조건부 엣지**: 대화 종료 조건을 확인하여 시뮬레이션을 계속하거나 종료합니다.
#
# 아래 코드는 StateGraph를 생성하고 노드와 엣지를 정의합니다.

# %%
from langgraph.graph import END, StateGraph

# StateGraph 인스턴스 생성
graph_builder = StateGraph(State)

# 노드 정의
graph_builder.add_node("simulated_user", simulated_user_node)
graph_builder.add_node("ai_assistant", ai_assistant_node)

# 엣지 정의 (챗봇 -> 시뮬레이션된 사용자)
graph_builder.add_edge("ai_assistant", "simulated_user")

# 조건부 엣지 정의
graph_builder.add_conditional_edges(
    "simulated_user",
    should_continue,
    {
        "end": END,  # 종료 조건이 충족되면 시뮬레이션을 중단
        "continue": "ai_assistant",  # 종료 조건이 충족되지 않으면 상담사 역할 노드로 메시지를 전달
    },
)

# 시작점 설정
graph_builder.set_entry_point("ai_assistant")

# 그래프 컴파일
simulation = graph_builder.compile()

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(simulation)

# %% [markdown]
# ## 시뮬레이션 시작
#
# 이제 구축한 시뮬레이션 그래프를 실행하여 챗봇의 성능을 평가합니다. 빈 메시지로 호출하면 챗봇이 초기 대화를 시작하게 됩니다.
#
# `stream_graph` 함수를 사용하여 시뮬레이션 과정을 실시간으로 확인할 수 있습니다. 아래 코드는 고객이 화가 난 상태로 대화를 시작하는 시나리오를 시뮬레이션합니다.

# %% [markdown]
# `stream_graph` 함수는 시뮬레이션에서 스트리밍되는 데이터 청크를 순회하며, 각 노드의 실행 결과를 실시간으로 출력합니다. `recursion_limit`을 설정하여 무한 루프를 방지합니다.

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid


# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

# 입력 메시지 설정
inputs = {
    "messages": [HumanMessage(content="안녕하세요? 저 지금 좀 화가 많이 났습니다^^")]
}

# 그래프 스트리밍
stream_graph(simulation, inputs, config, node_names=["simulated_user", "ai_assistant"])
