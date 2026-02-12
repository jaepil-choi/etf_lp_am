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
# # 멀티 에이전트 감독자 (Multi-Agent Supervisor)
#
# 이 튜토리얼에서는 **LangGraph**를 활용하여 다중 에이전트 시스템을 구축하고, 에이전트 간 작업을 효율적으로 조정하고 감독자(Supervisor)를 통해 관리하는 방법을 살펴봅니다. 여러 에이전트를 동시에 다루며, 각 에이전트가 자신의 역할을 수행하도록 관리하고, 작업 완료 시 이를 적절히 처리하는 과정을 다룹니다.
#
# > **참고 문서**: [LangGraph Multi-Agent Supervisor](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor)
#
# ---
#
# ## 개요
#
# 이전 튜토리얼에서는 초기 연구자(Researcher) 에이전트의 출력에 따라 메시지를 자동으로 라우팅하는 방식을 보여주었습니다. 그러나 에이전트가 여러 개로 늘어나고, 이들을 조정해야 할 경우, 단순한 분기 로직만으로는 한계가 있습니다.
#
# 여기서는 LLM을 활용한 Supervisor를 통해 에이전트들을 관리하고, 각 에이전트 노드의 결과를 바탕으로 팀 전체를 조율하는 방법을 소개합니다.
#
# **중점 사항:**
#
# - Supervisor는 다양한 전문 에이전트를 한 데 모아, 하나의 팀(team)으로 운영하는 역할을 합니다.
# - Supervisor 에이전트는 팀의 진행 상황을 관찰하고, 각 단계별로 적절한 에이전트를 호출하거나 작업을 종료하는 등의 로직을 수행합니다.
#
# ---
#
# ## 이 튜토리얼에서 다룰 내용
#
# - **설정(Setup)**: 필요한 패키지 설치 및 API 키 설정 방법
# - **도구 생성(Tool Creation)**: 웹 검색 및 플롯(plot) 생성 등, 에이전트가 사용할 도구 정의
# - **에이전트 감독자 생성(Creating the Supervisor)**: 작업자(Worker) 노드의 선택 및 작업 완료 시 처리 로직을 담은 Supervisor 생성
# - **그래프 구성(Constructing the Graph)**: 상태(State) 및 작업자(Worker) 노드를 정의하여 전체 그래프 구성
# - **팀 호출(Invoking the Team)**: 그래프를 호출하여 실제로 다중 에이전트 시스템이 어떻게 작동하는지 확인

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
logging.langsmith("LangGraph-Tutorial")

# %% [markdown]
# ## 모델 설정
#
# 본 튜토리얼에 사용할 모델명을 설정합니다. `langchain_teddynote` 패키지의 `get_model_name` 함수를 사용하여 최신 모델명을 가져옵니다. 아래 코드는 GPT-4 모델을 사용하도록 설정합니다.

# %%
from langchain_teddynote.models import get_model_name, LLMs

# 최신 버전의 모델명을 가져옵니다.
MODEL_NAME = get_model_name(LLMs.GPT4)
print(f"사용 모델: {MODEL_NAME}")

# %% [markdown]
# ## 상태 정의
#
# 멀티 에이전트 시스템에서 활용할 상태(State)를 정의합니다. `messages`는 에이전트 간 공유하는 메시지 목록이며, `next`는 다음으로 라우팅할 에이전트를 나타냅니다.
#
# 아래 코드는 에이전트 상태 스키마를 정의합니다.

# %%
import operator
from typing import Sequence, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


# 상태 정의
class AgentState(TypedDict):
    # 에이전트 간 공유하는 메시지 목록
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 다음으로 라우팅할 에이전트
    next: str


# %% [markdown]
# ## 에이전트 생성
#
# ### 도구(Tool) 생성
#
# 이 예제에서는 검색 엔진을 사용하여 웹 조사를 수행하는 에이전트와 플롯을 생성하는 에이전트를 만듭니다. 
#
# - **Research Agent**: `TavilySearch` 도구를 사용하여 웹 조사를 수행합니다.
# - **Coder Agent**: `PythonREPLTool` 도구를 사용하여 코드를 실행합니다.
#
# 아래 코드는 두 도구를 정의합니다.

# %%
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool

# 최대 5개의 검색 결과를 반환하는 Tavily 검색 도구 초기화
tavily_tool = TavilySearch(max_results=5)

# 로컬에서 코드를 실행하는 Python REPL 도구 초기화 (안전하지 않을 수 있음)
python_repl_tool = PythonREPLTool()

# %% [markdown]
# ## Agent 노드 생성 유틸리티 구현
#
# LangGraph를 사용하여 다중 에이전트 시스템을 구축할 때, **도우미 함수(Utility Function)**는 에이전트 노드를 생성하고 관리하는 데 중요한 역할을 합니다. 이러한 함수는 코드의 재사용성을 높이고, 에이전트 간의 상호작용을 간소화합니다.
#
# 이 섹션에서 구현할 유틸리티는 다음과 같은 역할을 수행합니다.
#
# - **에이전트 노드 생성**: 각 에이전트의 역할에 맞는 노드를 생성하기 위한 함수 정의
# - **작업 흐름 관리**: 에이전트 간의 작업 흐름을 조정하고 최적화하는 유틸리티 제공
# - **결과 변환**: 에이전트의 응답을 표준화된 형식으로 변환하여 다른 에이전트와 공유
#
# `functools.partial`을 활용하면 같은 함수를 여러 에이전트에 재사용할 수 있어 코드 중복을 크게 줄일 수 있습니다.

# %% [markdown]
# ### agent_node 함수 정의
#
# 다음은 `agent_node`라는 함수를 정의하는 예시입니다. 이 함수는 주어진 상태(state)와 에이전트를 사용하여 에이전트를 실행하고, 결과를 `HumanMessage` 형식으로 변환하여 반환합니다.
#
# 이 함수를 나중에 `functools.partial`을 사용하여 특정 에이전트와 이름이 바인딩된 노드 함수로 만들 것입니다.
#
# 아래 코드는 에이전트 노드 생성 함수를 정의합니다.

# %%
from langchain_core.messages import HumanMessage


def agent_node(state, agent, name):
    """에이전트 노드 함수

    지정한 agent를 실행하고, 결과를 HumanMessage로 변환하여 반환합니다.
    name 파라미터는 메시지의 출처를 식별하는 데 사용됩니다.
    """
    # 에이전트를 현재 상태로 호출
    agent_response = agent.invoke(state)
    # 에이전트의 마지막 응답을 HumanMessage로 변환하여 반환
    return {
        "messages": [
            HumanMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }


# %% [markdown]
# ### functools.partial의 역할
#
# `functools.partial`은 기존 함수의 일부 인자 또는 키워드 인자를 미리 고정하여 새 함수를 생성하는 Python 내장 유틸리티입니다. 자주 사용하는 함수 호출 패턴을 간소화할 수 있습니다.
#
# **주요 역할**
#
# 1. **미리 정의된 값으로 새 함수 생성**: 기존 함수의 일부 인자를 미리 지정해서 새 함수를 반환합니다.
# 2. **코드 간결화**: 자주 사용하는 함수 호출 패턴을 단순화하여 코드 중복을 줄입니다.
# 3. **가독성 향상**: 특정 작업에 맞춰 함수의 동작을 맞춤화해 더 직관적으로 사용 가능하게 만듭니다.
#
# **예시 코드**
#
# ```python
# research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
# ```
#
# 위 코드에서 `functools.partial`은 `agent_node` 함수에 `agent=research_agent`와 `name="Researcher"`라는 값을 고정합니다. 이제 `research_node`는 `state`만 전달하면 됩니다.
#
# ```python
# # 기존 방식: 매번 모든 인자 전달
# agent_node(state, agent=research_agent, name="Researcher")
#
# # partial 사용 후: state만 전달
# research_node(state)
# ```

# %% [markdown]
# ### Research Agent 생성
#
# 이제 `functools.partial`을 사용하여 `research_node`를 생성합니다. `create_react_agent`로 Research Agent를 만들고, 이를 `agent_node` 함수와 결합하여 노드 함수로 변환합니다.
#
# 아래 코드는 Research Agent와 해당 노드를 생성합니다.

# %%
import functools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Research Agent 생성: Tavily 검색 도구를 사용하는 ReAct 에이전트
research_agent = create_react_agent(
    ChatOpenAI(model="gpt-4.1-mini"), 
    tools=[tavily_tool]
)

# research_node 생성: agent_node 함수에 agent와 name을 바인딩
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# %% [markdown]
# ### Research Agent 테스트
#
# 생성된 `research_node`가 정상적으로 동작하는지 확인합니다. 테스트 메시지를 전달하여 에이전트가 응답을 생성하는지 확인할 수 있습니다.
#
# 아래 코드는 Research Agent 노드를 테스트합니다.

# %%
# Research Agent 테스트: 간단한 메시지로 동작 확인
research_node(
    {
        "messages": [
            HumanMessage(content="Code hello world and print it to the terminal")
        ]
    }
)

# %% [markdown]
# ## Supervisor Agent 생성
#
# Supervisor는 여러 에이전트를 관리하고, 현재 상태를 기반으로 다음에 실행할 에이전트를 결정하는 역할을 합니다. 이 섹션에서는 Supervisor Agent를 생성합니다.
#
# Supervisor의 핵심 역할은 다음과 같습니다.
#
# - **작업 분배**: 사용자 요청을 분석하여 적합한 에이전트에게 작업을 할당합니다.
# - **진행 상황 모니터링**: 각 에이전트의 결과를 확인하고 다음 단계를 결정합니다.
# - **종료 판단**: 모든 작업이 완료되면 `FINISH`를 반환하여 워크플로우를 종료합니다.
#
# 아래 코드는 Supervisor가 반환할 응답 모델과 팀 멤버를 정의합니다.

# %%
from pydantic import BaseModel
from typing import Literal

# 팀 멤버 에이전트 목록 정의
members = ["Researcher", "Coder"]

# 다음 작업자 선택 옵션: FINISH + 멤버 목록
options_for_next = ["FINISH"] + members


class RouteResponse(BaseModel):
    """Supervisor의 라우팅 응답 모델

    다음으로 실행할 에이전트를 선택하거나 FINISH를 반환합니다.
    """
    next: Literal[*options_for_next]


# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# 시스템 프롬프트: 작업자 간의 대화를 관리하는 감독자 역할을 정의
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# ChatPromptTemplate 생성: 시스템 프롬프트와 메시지 플레이스홀더 결합
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? "
            "Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options_for_next), members=", ".join(members))

# LLM 초기화: Supervisor에 사용할 모델 설정
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


def supervisor_agent(state):
    """Supervisor Agent 함수

    현재 상태를 분석하여 다음으로 실행할 에이전트를 결정합니다.
    RouteResponse 형식으로 구조화된 출력을 반환합니다.
    """
    # 프롬프트와 LLM을 결합하여 체인 구성
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    # 체인 호출 및 결과 반환
    return supervisor_chain.invoke(state)


# %% [markdown]
# ## 그래프 구성
#
# 이제 그래프를 구축할 준비가 되었습니다. 앞서 정의한 에이전트 노드와 Supervisor를 `StateGraph`에 등록하고, 엣지를 연결하여 워크플로우를 완성합니다.
#
# 그래프 구성의 핵심 요소는 다음과 같습니다.
#
# - **노드 추가**: Research Agent, Coder Agent, Supervisor Agent를 그래프에 등록합니다.
# - **엣지 연결**: 각 멤버 에이전트는 작업 완료 후 Supervisor로 돌아갑니다.
# - **조건부 라우팅**: Supervisor는 다음 에이전트를 동적으로 결정합니다.
#
# 아래 코드는 Research Agent와 Coder Agent 노드를 생성합니다.

# %%
import functools
from langgraph.prebuilt import create_react_agent

# Research Agent 생성: Tavily 검색 도구를 사용
research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# Coder Agent 시스템 프롬프트: 한글 폰트 설정 포함
code_system_prompt = """
Be sure to use the following font in your code for visualization.

##### 폰트 설정 #####
import platform

# OS 판단
current_os = platform.system()

if current_os == "Windows":
    # Windows 환경 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
    fontprop = fm.FontProperties(fname=font_path, size=12)
    plt.rc("font", family=fontprop.get_name())
elif current_os == "Darwin":  # macOS
    # Mac 환경 폰트 설정
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux 등 기타 OS
    # 기본 한글 폰트 설정 시도
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except:
        print("한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")

##### 마이너스 폰트 깨짐 방지 #####
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 폰트 깨짐 방지
"""

# Coder Agent 생성: Python REPL 도구를 사용
coder_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    prompt=code_system_prompt,
)
coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

# %%
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# StateGraph 생성: AgentState를 상태 스키마로 사용
workflow = StateGraph(AgentState)

# 그래프에 노드 추가
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Supervisor", supervisor_agent)

# 멤버 노드 -> Supervisor 노드로 엣지 추가 (작업 완료 후 Supervisor로 복귀)
for member in members:
    workflow.add_edge(member, "Supervisor")

# 조건부 라우팅 맵 정의: 멤버 이름 -> 노드, FINISH -> END
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END


def get_next(state):
    """상태에서 다음 노드를 반환하는 함수"""
    return state["next"]


# Supervisor 노드에서 조건부 엣지 추가
workflow.add_conditional_edges("Supervisor", get_next, conditional_map)

# 시작점: START -> Supervisor
workflow.add_edge(START, "Supervisor")

# 그래프 컴파일: MemorySaver를 체크포인터로 사용
graph = workflow.compile(checkpointer=MemorySaver())

# %% [markdown]
# ### 그래프 시각화
#
# 완성된 그래프 구조를 시각화하여 노드 간의 연결 관계를 확인합니다. `visualize_graph` 함수를 사용하면 그래프의 전체 구조를 한눈에 파악할 수 있습니다.
#
# 아래 코드는 컴파일된 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(graph)

# %% [markdown]
# ## 팀 호출
#
# 이제 생성된 그래프를 실행하여 Multi-Agent Supervisor 시스템의 동작을 확인합니다. 사용자 요청을 입력으로 전달하면, Supervisor가 적절한 에이전트를 선택하여 작업을 수행하고, 최종 결과를 반환합니다.
#
# 아래 예시에서는 대한민국의 1인당 GDP 추이를 시각화하는 작업을 요청합니다. Supervisor는 먼저 Researcher에게 데이터 수집을 지시하고, 이후 Coder에게 시각화를 지시합니다.
#
# 아래 코드는 그래프를 호출하여 결과를 확인합니다.

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid, invoke_graph

# config 설정: 재귀 최대 횟수와 thread_id 지정
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})

# 입력 메시지 정의
inputs = {
    "messages": [
        HumanMessage(
            content="2010년 ~ 2024년까지의 대한민국의 1인당 GDP 추이를 그래프로 시각화 해주세요."
        )
    ],
}

# 그래프 실행 및 결과 출력
invoke_graph(graph, inputs, config)
