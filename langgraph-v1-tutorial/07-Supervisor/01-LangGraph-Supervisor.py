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
# # LangGraph 멀티에이전트 협업 네트워크 (Supervisor Pattern)
#
# ## 개요
#
# 이 튜토리얼에서는 **Supervisor Pattern** 을 사용하여 여러 전문 에이전트가 협업하는 시스템을 구축합니다. Supervisor Pattern은 한 명의 관리자(supervisor)가 여러 전문가(agent)들에게 작업을 분배하고 조정하는 실무와 유사한 아키텍처입니다.
#
# > **참고 문서**: [LangChain Multi-Agent](https://docs.langchain.com/oss/python/langchain/multi-agent/)
#
# ### 실무 비유로 이해하기
#
# 일반적인 회사에서 **프로젝트 매니저** 가 팀을 관리하는 방식과 매우 유사합니다. 프로젝트 매니저(Supervisor)는 고객 요청을 분석하고 적절한 팀원에게 업무를 배정합니다. 개발팀(Research Agent)은 기술 조사와 정보 수집을 담당하고, 재무팀(Math Agent)은 예산 계산과 수치 분석을 담당합니다.
#
# 각 팀원은 자신의 전문 분야에만 집중하고, 매니저가 전체 프로젝트를 조율하는 구조입니다.
#
# ## 학습 목표
#
# 이 튜토리얼을 완료하면 다음을 수행할 수 있습니다.
#
# 1. **Supervisor Pattern 이해** - 멀티에이전트 시스템의 핵심 아키텍처
# 2. **전문 에이전트 구축** - 연구(Research)와 수학(Math) 에이전트 생성
# 3. **Supervisor 구현** - langgraph-supervisor 라이브러리 활용
# 4. **커스텀 Supervisor** - 스크래치부터 직접 구현
# 5. **고급 작업 위임** - 동적 태스크 할당과 핸드오프
#
# ## 아키텍처 다이어그램
#
# Supervisor Pattern의 전체 구조는 다음과 같습니다. 사용자의 요청이 Supervisor에게 전달되면, Supervisor는 적절한 전문 에이전트에게 작업을 위임하고 결과를 수집하여 최종 응답을 생성합니다.
#
# ```
#         ┌─────────────┐
#         │   사용자    │
#         └──────┬──────┘
#                │
#                ▼
#      ┌─────────────────┐
#      │   Supervisor    │ ◀── 중앙 관리자
#      │   (감독관)      │
#      └────┬───────┬────┘
#           │       │
#     ┌─────▼───┐ ┌─▼─────┐
#     │Research │ │ Math  │ ◀── 전문 에이전트들
#     │ Agent   │ │ Agent │
#     └─────────┘ └───────┘
# ```
#
# ---
#
# ## 핵심 개념
#
# ### Supervisor Pattern이란?
#
# **Supervisor Pattern** 은 복잡한 작업을 여러 전문 에이전트에게 분배하고 조정하는 멀티에이전트 아키텍처입니다. 이 패턴의 핵심 특징은 다음과 같습니다.
#
# - **중앙 집중식 제어**: Supervisor가 모든 통신과 작업 위임을 관리합니다.
# - **전문화**: 각 에이전트는 특정 도메인에 특화되어 있습니다.
# - **조정**: Supervisor가 컨텍스트와 요구사항에 따라 적절한 에이전트를 선택합니다.
# - **통합**: 각 에이전트의 결과를 수집하고 최종 응답을 생성합니다.
#
# ### 언제 사용하나요?
#
# 실무에서 다음과 같은 상황에 Supervisor Pattern이 특히 유용합니다.
#
# - 여러 전문 분야가 필요한 복잡한 작업 (예: 시장 조사 + 재무 분석)
# - 명확한 작업 분배가 필요한 경우 (각 팀의 역할이 다를 때)
# - 중앙 통제가 중요한 시스템 (품질 관리와 일관성이 중요)
# - 에이전트 간 조정이 필요한 워크플로우 (순차적 작업 처리)

# %% [markdown]
# ## 환경 설정

# %%
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv(override=True)

# 기본 라이브러리 import
import os
import getpass
from typing import Annotated, List, Literal, TypedDict
from dataclasses import dataclass
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AnyMessage,
    convert_to_messages,
)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-Tutorial")

print("✅ 환경 설정 완료!")


# %% [markdown]
# ## 헬퍼 함수 정의
#
# 에이전트들의 출력 내용을 깔끔하게 표시하기 위한 유틸리티 함수들을 정의합니다. 멀티에이전트 시스템에서는 여러 에이전트가 동시에 작업하므로, 각 에이전트의 출력을 구분해서 보기 어려울 수 있습니다.
#
# ### 헬퍼 함수의 역할
#
# 이를 해결하기 위해 다음 함수들을 사용합니다.
#
# - **`pretty_print_message()`**: 개별 메시지를 보기 좋게 포맷팅합니다.
# - **`pretty_print_messages()`**: 여러 에이전트의 메시지 업데이트를 구분하여 표시합니다.
#
# 각 노드(에이전트)별로 색상과 구분선을 추가하여 가독성을 향상시킵니다. 아래 코드는 헬퍼 함수들을 정의합니다.

# %%
def pretty_print_message(message, indent=False):
    """개별 메시지를 포맷팅하여 출력합니다."""
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    # 들여쓰기 적용
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    """그래프 실행 결과의 메시지 업데이트를 포맷팅하여 출력합니다."""
    is_subgraph = False

    # 튜플인 경우 서브그래프 업데이트
    if isinstance(update, tuple):
        ns, update = update
        # 부모 그래프 업데이트는 건너뜀
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"[서브그래프 {graph_id}로부터 업데이트]")
        print("\n")
        is_subgraph = True

    # 각 노드의 업데이트 처리
    for node_name, node_update in update.items():
        update_label = f"[노드 {node_name}로부터 업데이트]"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        # 메시지 변환 및 출력
        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]  # 마지막 메시지만

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


print("헬퍼 함수 정의 완료")

# %% [markdown]
# ---
#
# ## Part 1: 전문 워커 에이전트 생성
#
# 먼저 특정 작업에 특화된 두 개의 워커 에이전트를 만들어봅시다. 각 에이전트는 자신의 전문 분야에 집중하며, Supervisor가 적절한 에이전트에게 작업을 위임합니다.
#
# 1. **Research Agent**: 웹 검색 도구를 사용하여 정보를 찾는 에이전트
# 2. **Math Agent**: 수학 계산을 수행하는 에이전트

# %% [markdown]
# ### Research Agent (연구 에이전트)
#
# Research Agent는 Tavily API를 사용하여 웹 검색 기능을 갖춘 연구 에이전트입니다. 이 에이전트는 최신 정보를 검색하고, 사실 확인을 수행하며, 검색 결과를 기반으로 응답을 생성합니다.
#
# `create_react_agent`는 LangGraph의 사전 구축된 에이전트 생성 함수로, ReAct(Reasoning + Acting) 패턴을 따르는 에이전트를 쉽게 생성할 수 있게 해줍니다. 아래 코드는 Tavily 웹 검색 도구를 초기화하고 Research Agent를 생성합니다.

# %%
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

# Tavily 웹 검색 도구 초기화 (최대 3개 결과 반환)
web_search = TavilySearch(max_results=3)

# 웹 검색 도구 테스트 실행
test_results = web_search.invoke("2024년 대한민국 GDP")
print("웹 검색 도구 테스트:")
print(f"검색 결과 수: {len(test_results['results'])}")
print(f"첫 번째 결과 미리보기: {test_results['results'][0]['content'][:200]}...\n")

# Research Agent 생성 (웹 검색 전문 에이전트)
research_agent = create_react_agent(
    model="gpt-4.1",  # 사용할 LLM 모델
    tools=[web_search],  # 웹 검색 도구 제공
    prompt=(
        "You are a research specialist agent.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "- Conduct thorough web searches to find accurate and up-to-date information\n"
        "- Provide detailed, factual responses based on search results\n"
        "- Always cite your sources when presenting information\n\n"
        "STRICT GUIDELINES:\n"
        "- Focus ONLY on research and information gathering tasks\n"
        "- DO NOT perform mathematical calculations or analysis\n"
        "- When task is complete, report findings directly to the supervisor\n"
        "- Present ONLY the research results without additional commentary\n\n"
        "OUTPUT FORMAT:\n"
        "- Provide clear, structured findings\n"
        "- Include relevant data points and sources\n"
        "- Use bullet points for clarity when appropriate"
    ),
    name="research_agent",  # 에이전트 이름
)

print("Research Agent 생성 완료")

# %%
# Research Agent 테스트
print("Research Agent 테스트\n")
print("질문: 'NYC의 현재 시장은 누구인가요?'\n")

for chunk in research_agent.stream(
    {"messages": [{"role": "user", "content": "NYC의 현재 시장은 누구인가요?"}]}
):
    pretty_print_messages(chunk)


# %% [markdown]
# ### Math Agent (수학 에이전트)
#
# Math Agent는 기본적인 수학 연산을 수행할 수 있는 에이전트입니다. 덧셈, 곱셈, 나눗셈, 거듭제곱, 백분율 계산 등의 도구를 갖추고 있으며, 수치 분석이 필요한 작업을 담당합니다.
#
# 각 수학 함수는 `@tool` 데코레이터 없이 직접 함수로 정의되어 있으며, `create_react_agent`에서 자동으로 도구로 변환됩니다. 아래 코드는 수학 도구 함수들을 정의하고 Math Agent를 생성합니다.

# %%
# 수학 도구 정의 (기본 계산 함수들)
def add(a: float, b: float) -> float:
    """두 숫자를 더합니다."""
    return a + b


def multiply(a: float, b: float) -> float:
    """두 숫자를 곱합니다."""
    return a * b


def divide(a: float, b: float) -> float:
    """두 숫자를 나눕니다. 0으로 나누면 무한대를 반환합니다."""
    if b == 0:
        return float("inf")  # 0으로 나누기 방지, 무한대 반환
    return a / b


def power(base: float, exponent: float) -> float:
    """밑(base)을 지수(exponent)만큼 거듭제곱합니다."""
    return base**exponent


def percentage(value: float, total: float) -> float:
    """전체(total) 대비 값(value)의 백분율을 계산합니다."""
    if total == 0:
        return 0.0  # 전체값이 0인 경우 0% 반환
    return (value / total) * 100


# Math Agent 생성 (수학 계산 전문 에이전트)
math_agent = create_react_agent(
    model="gpt-4.1",  # 사용할 LLM 모델
    tools=[add, multiply, divide, power, percentage],  # 수학 도구들 제공
    prompt=(
        "You are a mathematics specialist agent.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "- Perform accurate mathematical calculations\n"
        "- Handle arithmetic operations, percentages, and basic algebra\n"
        "- Show calculation steps when solving complex problems\n\n"
        "STRICT GUIDELINES:\n"
        "- Focus ONLY on mathematical computations and analysis\n"
        "- DO NOT perform web searches or gather external information\n"
        "- When calculations are complete, report results to the supervisor\n"
        "- Present ONLY the calculation results and methodology\n\n"
        "OUTPUT FORMAT:\n"
        "- Show clear calculation steps\n"
        "- Round results to 2 decimal places when appropriate\n"
        "- Include units if relevant"
    ),
    name="math_agent",  # 에이전트 이름
)

print("Math Agent 생성 완료")

# %%
# Math Agent 테스트
print("Math Agent 테스트\n")
print("질문: '(3 + 5) x 7은 얼마인가요?'\n")

for chunk in math_agent.stream(
    {"messages": [{"role": "user", "content": "(3 + 5) x 7은 얼마인가요?"}]}
):
    pretty_print_messages(chunk)

# %% [markdown]
# ---
#
# ## Part 2: langgraph-supervisor로 Supervisor 생성
#
# 이제 사전 구축된 `langgraph-supervisor` 라이브러리를 사용하여 supervisor를 빠르게 구현해봅시다. 이 라이브러리는 멀티에이전트 시스템에서 supervisor 패턴을 쉽게 구현할 수 있도록 해줍니다.
#
# `create_supervisor` 함수는 다음과 같은 기능을 제공합니다.
#
# - 여러 에이전트를 관리하는 supervisor 생성
# - 에이전트 간 작업 위임 및 조정
# - 핸드오프 메시지 자동 생성
# - 다양한 출력 모드 지원
#
# 아래 코드는 `langgraph-supervisor`를 사용하여 Research Agent와 Math Agent를 관리하는 Supervisor를 생성합니다.

# %%
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

# Supervisor 생성 - 사전 구축된 패턴 사용
supervisor = create_supervisor(
    model=init_chat_model("gpt-4.1"),  # Supervisor가 사용할 LLM 모델
    agents=[research_agent, math_agent],  # 관리할 에이전트들 (연구, 수학)
    prompt=(
        "You are a supervisor managing two specialized agents:\n\n"
        "1. RESEARCH AGENT:\n"
        "   - Specializes in web searches and information gathering\n"
        "   - Use for: finding facts, current events, statistics, definitions\n\n"
        "2. MATH AGENT:\n"
        "   - Specializes in mathematical calculations and analysis\n"
        "   - Use for: arithmetic, percentages, conversions, comparisons\n\n"
        "ORCHESTRATION GUIDELINES:\n"
        "- Analyze the user's request carefully\n"
        "- Delegate tasks to the appropriate specialist agent\n"
        "- Process tasks sequentially (one agent at a time)\n"
        "- Coordinate multiple agents when complex tasks require both research and calculation\n"
        "- Synthesize results from agents into a coherent final response\n\n"
        "IMPORTANT RULES:\n"
        "- DO NOT perform any work yourself - always delegate to agents\n"
        "- DO NOT call agents in parallel - use sequential processing\n"
        "- Always provide clear, complete answers based on agent results"
    ),
    add_handoff_back_messages=True,  # 에이전트 간 핸드오프 메시지 추가
    output_mode="full_history",  # 전체 히스토리 출력 모드
).compile()  # 그래프 컴파일

print("Supervisor 생성 완료")

# %%
# 그래프 시각화
try:
    from IPython.display import display, Image

    display(Image(supervisor.get_graph().draw_mermaid_png()))
except:
    # ASCII로 시각화
    print("Supervisor 그래프 구조:\n")
    print(supervisor.get_graph().draw_ascii())

# %% [markdown]
# ### Supervisor 테스트 - 복합 작업
#
# Research와 Math 에이전트가 모두 필요한 복잡한 질문으로 테스트해봅시다. 이 예제에서는 먼저 GDP 정보를 검색한 후, 검색된 데이터를 기반으로 비율을 계산합니다. Supervisor가 각 에이전트에게 적절히 작업을 위임하는 과정을 확인할 수 있습니다.
#
# 아래 코드는 Supervisor에게 복합 작업을 요청하고 실행 과정을 출력합니다.

# %%
# 복합 작업 테스트: GDP 정보 검색 + 비율 계산
print("Supervisor 복합 작업 테스트\n")
print(
    "질문: '2024년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요.'\n"
)
print("=" * 80 + "\n")

# Supervisor 실행
for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "2024년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요.",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

# 최종 메시지 히스토리 저장
final_message_history = chunk.get("supervisor", {}).get("messages", [])

# %%
# 전체 메시지 히스토리 확인
print("전체 메시지 히스토리:\n")
print("=" * 80 + "\n")

for i, message in enumerate(final_message_history, 1):
    print(f"[메시지 {i}]")
    message.pretty_print()
    print("-" * 40)

# %% [markdown]
# ---
#
# ## Part 3: 스크래치부터 Supervisor 구현
#
# 이제 동일한 멀티에이전트 시스템을 스크래치부터 구현해봅시다. 이를 통해 내부 동작 원리를 더 깊이 이해할 수 있습니다. `langgraph-supervisor` 라이브러리 없이 직접 핸드오프 메커니즘과 그래프를 구성하는 방법을 배웁니다.
#
# 구현 단계는 다음과 같습니다.
#
# 1. 에이전트 간 통신 설정 (핸드오프)
# 2. Supervisor 에이전트 생성
# 3. 전체 시스템을 하나의 그래프로 결합

# %% [markdown]
# ### 에이전트 간 통신 설정 (Handoff)
#
# **핸드오프(Handoff)** 는 한 에이전트가 다른 에이전트에게 제어권을 넘기는 메커니즘입니다. 실제 업무에서 **"이 업무는 A팀에서 처리하고, 완료되면 B팀으로 넘겨주세요"** 라고 하는 것과 같은 개념입니다.
#
# > **참고 문서**: [LangChain Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
#
# ### 핸드오프 작동 방식
#
# 예를 들어 고객이 "올해 애플 주가 동향을 분석해주세요"라고 요청하면 다음과 같이 처리됩니다.
#
# 1. **Supervisor**: "먼저 애플 주가 데이터를 조사해야겠네" → **Research Agent** 로 핸드오프
# 2. **Research Agent**: 웹에서 애플 주가 정보 수집 완료 → **Supervisor** 로 복귀
# 3. **Supervisor**: "이제 수집된 데이터로 계산해보자" → **Math Agent** 로 핸드오프
# 4. **Math Agent**: 상승률, 변동성 계산 완료 → **Supervisor** 로 복귀
# 5. **Supervisor**: 최종 결과를 사용자에게 전달
#
# LangGraph에서는 **Command** 객체를 사용하여 다음 에이전트로 제어권을 전달합니다. 아래 코드는 핸드오프 도구를 생성하는 함수를 정의합니다.

# %%
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """다른 에이전트로 제어권을 전달하는 핸드오프 도구를 생성합니다."""
    # 도구 이름 생성
    name = f"transfer_to_{agent_name}"
    description = (
        description or f"Transfer task to {agent_name} for specialized processing."
    )

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],  # 현재 상태 주입
        tool_call_id: Annotated[str, InjectedToolCallId],  # 도구 호출 ID 주입
    ) -> Command:
        """대상 에이전트로 핸드오프를 실행합니다."""
        # 핸드오프 성공 메시지 생성
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }

        # Command를 통해 다음 에이전트로 라우팅
        return Command(
            goto=agent_name,  # 대상 에이전트 지정
            update={  # 상태 업데이트
                **state,
                "messages": state["messages"] + [tool_message],
            },
            graph=Command.PARENT,  # 부모 그래프에서 네비게이션
        )

    return handoff_tool


# 각 에이전트로의 핸드오프 도구 생성
assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign information gathering and web search tasks to the research specialist.",
)

assign_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description="Assign mathematical calculations and numerical analysis to the math specialist.",
)

print("핸드오프 도구 생성 완료")
print(f"  - {assign_to_research_agent.name}: Research 에이전트로 작업 위임")
print(f"  - {assign_to_math_agent.name}: Math 에이전트로 작업 위임")

# %% [markdown]
# ### Supervisor 에이전트 생성
#
# 이제 핸드오프 도구를 사용하여 커스텀 Supervisor 에이전트를 생성합니다. Supervisor는 사용자의 요청을 분석하고 적절한 전문 에이전트에게 작업을 위임합니다.
#
# 아래 코드는 핸드오프 도구를 사용하는 Supervisor 에이전트를 생성합니다.

# %%
# 커스텀 Supervisor 에이전트 생성 (핸드오프 도구 활용)
supervisor_agent = create_react_agent(
    model="gpt-4.1",  # 사용할 LLM 모델
    tools=[assign_to_research_agent, assign_to_math_agent],  # 핸드오프 도구들
    prompt=(
        "You are a supervisor orchestrating specialized agents.\n\n"
        "AVAILABLE AGENTS:\n"
        "1. Research Agent: Web searches, fact-finding, information gathering\n"
        "2. Math Agent: Calculations, percentages, numerical analysis\n\n"
        "ORCHESTRATION STRATEGY:\n"
        "- Analyze requests to identify required capabilities\n"
        "- Delegate to appropriate agents using transfer tools\n"
        "- Coordinate multi-step tasks by sequencing agent calls\n"
        "- Synthesize results into comprehensive responses\n\n"
        "EXECUTION RULES:\n"
        "- Always delegate work to specialist agents\n"
        "- Process tasks sequentially (one agent at a time)\n"
        "- Wait for agent results before proceeding\n"
        "- Provide clear final answers based on collected information"
    ),
    name="supervisor",  # 에이전트 이름
)

print("Supervisor 에이전트 생성 완료")

# %% [markdown]
# ### 멀티에이전트 그래프 생성
#
# 이제 Supervisor와 워커 에이전트들을 하나의 그래프로 결합합니다. `StateGraph`를 사용하여 노드를 추가하고, 엣지를 정의하여 에이전트 간의 흐름을 설정합니다.
#
# 아래 코드는 멀티에이전트 시스템 그래프를 정의하고 컴파일합니다.

# %%
from langgraph.graph import END

# 멀티에이전트 시스템 그래프 정의
custom_supervisor = (
    StateGraph(MessagesState)
    # Supervisor 노드 추가 (목적지 힌트 포함 - 시각화용)
    .add_node(supervisor_agent, destinations=("research_agent", "math_agent", END))
    # 워커 에이전트 노드 추가
    .add_node(research_agent)
    .add_node(math_agent)
    # 시작점에서 Supervisor로
    .add_edge(START, "supervisor")
    # 워커 에이전트는 항상 Supervisor로 복귀
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    # 그래프 컴파일
    .compile()
)

print("커스텀 멀티에이전트 시스템 생성 완료")

# %%
# 커스텀 그래프 시각화
try:
    from IPython.display import display, Image

    display(Image(custom_supervisor.get_graph().draw_mermaid_png()))
except:
    print("커스텀 Supervisor 그래프 구조:\n")
    print(custom_supervisor.get_graph().draw_ascii())

# %%
# 커스텀 Supervisor 테스트
print("커스텀 Supervisor 테스트\n")
print(
    "질문: '2024년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요.'\n"
)
print("=" * 80 + "\n")

for chunk in custom_supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "2024년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요.",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_custom_history = chunk.get("supervisor", {}).get("messages", [])

# %% [markdown]
# ---
#
# ## Part 4: 고급 작업 위임 (Task Delegation)
#
# 지금까지 에이전트들은 전체 메시지 히스토리를 해석하여 작업을 수행했습니다. 이제 Supervisor가 **명시적인 작업 설명** 을 작성하여 에이전트에게 전달하는 더 정교한 방식을 구현해봅시다.
#
# ### 기존 방식 vs 고급 방식 비교
#
# **기존 방식 (암묵적 위임)**: 전체 대화 히스토리를 에이전트에게 전달하면, 에이전트가 직접 맥락을 파악하여 작업을 수행합니다.
#
# **고급 방식 (명시적 위임)**: Supervisor가 구체적인 작업 지시를 작성하여 전달합니다. 예를 들어 "Please find 2024 annual revenue data for Apple Inc. and Samsung Electronics. Focus on official financial reports and include currency information."와 같이 명확한 지시를 제공합니다.
#
# ### 고급 방식의 장점
#
# - **정확성 향상**: 에이전트가 정확히 무엇을 해야 하는지 명확히 알 수 있습니다.
# - **효율성 증대**: 불필요한 추론 과정 없이 바로 작업을 수행합니다.
# - **일관성 보장**: 동일한 유형의 작업에 대해 표준화된 지시가 가능합니다.
#
# 아래 코드는 명시적 작업 설명을 포함하는 핸드오프 도구를 생성합니다.

# %%
from langgraph.types import Send


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    """명시적 작업 설명을 포함하는 핸드오프 도구를 생성합니다."""
    name = f"transfer_to_{agent_name}"
    description = (
        description or f"Transfer task with detailed instructions to {agent_name}."
    )

    @tool(name, description=description)
    def handoff_tool(
        # Supervisor가 작성한 작업 설명
        task_description: Annotated[
            str,
            "Detailed description of what the agent should do, including all context and requirements.",
        ],
        # 상태 주입 (LLM은 무시)
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        """명시적 작업 설명과 함께 핸드오프를 실행합니다."""
        # 작업 설명을 사용자 메시지로 변환
        task_description_message = {"role": "user", "content": task_description}
        # 에이전트에게 전달할 새로운 상태 생성
        agent_input = {**state, "messages": [task_description_message]}

        return Command(
            # Send를 사용하여 특정 상태를 에이전트에게 전달
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# 작업 설명이 포함된 핸드오프 도구 생성
assign_research_with_description = create_task_description_handoff_tool(
    agent_name="research_agent",
    description="Assign research task with specific instructions.",
)

assign_math_with_description = create_task_description_handoff_tool(
    agent_name="math_agent",
    description="Assign calculation task with specific instructions.",
)

print("작업 설명 핸드오프 도구 생성 완료")

# %%
# 작업 설명을 사용하는 고급 Supervisor 생성
supervisor_with_delegation = create_react_agent(
    model="gpt-4.1",  # 사용할 LLM 모델
    tools=[
        assign_research_with_description,
        assign_math_with_description,
    ],  # 작업 설명 핸드오프 도구들
    prompt=(
        "You are an advanced supervisor that delegates tasks with explicit instructions.\n\n"
        "DELEGATION STRATEGY:\n"
        "When assigning tasks to agents, provide:\n"
        "1. Clear, specific instructions about what to find or calculate\n"
        "2. All necessary context and parameters\n"
        "3. Expected output format or requirements\n"
        "4. Any constraints or special considerations\n\n"
        "AGENT CAPABILITIES:\n"
        "- Research Agent: Web searches, data gathering, fact-checking\n"
        "- Math Agent: All mathematical operations and analysis\n\n"
        "WORKFLOW RULES:\n"
        "- Create detailed task descriptions for each delegation\n"
        "- Process complex requests step-by-step\n"
        "- Ensure each agent has all information needed to succeed\n"
        "- Synthesize results into comprehensive final answers"
    ),
    name="supervisor",  # 에이전트 이름
)

# 고급 Supervisor 시스템 그래프 생성
advanced_supervisor = (
    StateGraph(MessagesState)
    .add_node(
        supervisor_with_delegation,
        destinations=("research_agent", "math_agent"),  # 가능한 목적지 노드들
    )
    .add_node(research_agent)  # 연구 에이전트 노드 추가
    .add_node(math_agent)  # 수학 에이전트 노드 추가
    .add_edge(START, "supervisor")  # 시작점에서 supervisor로 연결
    .add_edge("research_agent", "supervisor")  # 연구 에이전트에서 supervisor로 복귀
    .add_edge("math_agent", "supervisor")  # 수학 에이전트에서 supervisor로 복귀
    .compile()  # 그래프 컴파일
)

print("고급 Supervisor 시스템 생성 완료")

# %%
# 고급 Supervisor 테스트 - 서브그래프 출력 포함
print("고급 Supervisor (작업 설명 포함) 테스트\n")
print(
    "질문: '2024년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요.'\n"
)
print("=" * 80 + "\n")

# subgraphs=True로 서브그래프 실행도 추적
for chunk in advanced_supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "2024년 미국과 뉴욕주의 GDP를 찾고, 뉴욕주가 미국 전체 GDP의 몇 %를 차지하는지 계산해주세요.",
            }
        ]
    },
    subgraphs=True,  # 서브그래프 실행 추적
):
    pretty_print_messages(chunk, last_message=True)

# %% [markdown]
# ---
#
# ## Part 5: 실전 응용 예제
#
# Supervisor Pattern의 실제 활용 사례를 살펴봅시다. 비즈니스 분석과 멀티스텝 문제 해결 시나리오를 통해 패턴의 실용성을 확인합니다.

# %% [markdown]
# ### 복잡한 비즈니스 분석
#
# 여러 회사의 재무 데이터를 조사하고 비교 분석하는 복합 작업입니다. Research Agent가 데이터를 수집하고, Math Agent가 성장률을 계산하며, Supervisor가 최종 분석 결과를 종합합니다.
#
# 아래 코드는 비즈니스 분석 시나리오를 실행합니다.

# %%
# 복잡한 비즈니스 분석 시나리오
business_query = """
애플과 마이크로소프트의 2024년 3분기 매출을 조사하고,
각 회사의 전년 동기 대비 성장률을 계산해주세요.
그리고 어느 회사가 더 높은 성장률을 보였는지 분석해주세요.
"""

print("비즈니스 분석 시나리오\n")
print(f"요청: {business_query}\n")
print("=" * 80 + "\n")

# 고급 Supervisor로 처리
for chunk in advanced_supervisor.stream(
    {"messages": [{"role": "user", "content": business_query}]},
    subgraphs=True,
):
    # 마지막 메시지만 출력하여 간결하게 표시
    if "supervisor" in chunk:
        messages = chunk["supervisor"].get("messages", [])
        if messages and messages[-1].content:
            print(f"\n{chunk.keys()} 업데이트:")
            print(
                messages[-1].content[:500] + "..."
                if len(messages[-1].content) > 500
                else messages[-1].content
            )

# %% [markdown]
# ### 멀티스텝 문제 해결
#
# 순차적으로 여러 단계의 작업을 수행하는 시나리오입니다. 각 단계의 결과가 다음 단계의 입력으로 사용되며, Supervisor가 전체 흐름을 관리합니다.
#
# 아래 코드는 멀티스텝 문제 해결 시나리오를 실행합니다.

# %%
# 멀티스텝 문제 해결 시나리오
multistep_query = """
다음 작업을 순서대로 수행해주세요:
1. 현재 비트코인 가격을 USD로 조사
2. 1년 전 비트코인 가격을 조사
3. 1년간 가격 변동률 계산
4. 만약 1년 전에 $10,000를 투자했다면 현재 가치는 얼마인지 계산
"""

print("멀티스텝 문제 해결\n")
print(f"요청: {multistep_query}\n")
print("=" * 80 + "\n")

# 처리 및 단계별 출력
step_counter = 0
for chunk in advanced_supervisor.stream(
    {"messages": [{"role": "user", "content": multistep_query}]},
):
    for node_name, node_update in chunk.items():
        if "messages" in node_update:
            step_counter += 1
            print(f"\n[Step {step_counter}] {node_name}:")
            last_msg = node_update["messages"][-1]
            if hasattr(last_msg, "content"):
                # 콘텐츠 요약 출력
                content = last_msg.content
                if len(content) > 300:
                    print(content[:300] + "...")
                else:
                    print(content)

# %% [markdown]
# ---
#
# ## Part 6: Supervisor Pattern 베스트 프랙티스
#
# Supervisor Pattern을 효과적으로 활용하기 위한 핵심 가이드라인을 정리합니다.

# %% [markdown]
# ### 에이전트 전문화 가이드라인
#
# 실제 회사 조직과 같이 **각 에이전트의 전문 분야를 명확히 구분** 하는 것이 중요합니다.
#
# **권장하는 전문화 방식:**
#
# - **Research Agent**: 정보 수집 전문 (마치 조사팀처럼)
# - **Math Agent**: 계산 전문 (재무팀의 애널리스트처럼)
# - **Analysis Agent**: 데이터 분석 전문 (전략기획팀처럼)
# - **Writer Agent**: 콘텐츠 생성 전문 (마케팅팀의 카피라이터처럼)
#
# **피해야 할 패턴:**
#
# - 여러 에이전트가 유사한 기능 수행 (예: Research Agent 1, 2, 3...)
# - 너무 일반적인 역할 정의 (예: "모든 걸 다 하는 Agent")
# - 명확한 경계 없는 책임 영역 (어떤 Agent가 해야 할지 애매함)
#
# **전문화 원칙:**
#
# 1. **단일 책임 원칙**: 각 에이전트는 하나의 명확한 역할만 담당
# 2. **상호 보완적**: 에이전트들이 서로 다른 강점을 가져야 함
# 3. **명확한 경계**: 언제 어떤 에이전트를 사용할지 명확해야 함

# %%
# 에이전트 역할 명세 예시
agent_specifications = {
    "research_agent": {
        "role": "Information Specialist",
        "capabilities": [
            "Web searches",
            "Fact verification",
            "Data gathering",
            "Source citation",
        ],
        "limitations": ["No calculations", "No opinion generation", "No data analysis"],
    },
    "math_agent": {
        "role": "Computational Specialist",
        "capabilities": [
            "Arithmetic operations",
            "Percentage calculations",
            "Statistical analysis",
            "Formula application",
        ],
        "limitations": [
            "No web searches",
            "No qualitative analysis",
            "No content generation",
        ],
    },
    "supervisor": {
        "role": "Orchestration Manager",
        "capabilities": [
            "Task decomposition",
            "Agent selection",
            "Workflow coordination",
            "Result synthesis",
        ],
        "limitations": ["No direct task execution", "Must delegate all work"],
    },
}

# 역할 명세 출력
print("에이전트 역할 명세\n")
for agent, spec in agent_specifications.items():
    print(f"\n[{agent.upper()}]")
    print(f"  역할: {spec['role']}")
    print(f"  가능한 작업: {', '.join(spec['capabilities'][:2])}...")
    print(f"  제한사항: {', '.join(spec['limitations'][:2])}...")

# %% [markdown]
# ### 오류 처리 및 복구 전략
#
# 실무에서 멀티에이전트 시스템을 운영할 때 발생할 수 있는 문제들과 대응 방법을 알아봅니다.
#
# **발생 가능한 문제점들:**
#
# 1. **에이전트 실패 상황**
#    - API 한도 초과: 웹 검색 API 호출 제한에 걸리는 경우
#    - 네트워크 오류: 인터넷 연결 문제로 정보 조회 실패
#    - 잘못된 계산: 0으로 나누기 같은 수학적 오류
#
# 2. **불완전한 결과**
#    - 일부 정보만 수집: 요청한 5개 회사 중 3개만 찾은 경우
#    - 오래된 데이터: 최신 정보가 아닌 작년 데이터를 가져온 경우
#
# **강건한 시스템 설계 방법:**
#
# 1. **재시도 메커니즘**: 실패 시 최대 2-3회 재시도
# 2. **대체 수단 준비**: Research Agent 실패 시 다른 정보 소스 활용
# 3. **부분 결과 활용**: 완전하지 않아도 사용 가능한 정보는 전달
# 4. **사용자 알림**: 문제 발생 시 투명하게 상황 공유

# %%
# 오류 처리가 포함된 강건한 Supervisor
robust_supervisor_prompt = """
You are a robust supervisor with error handling capabilities.

ERROR HANDLING STRATEGY:
1. AGENT FAILURES:
   - If an agent fails, retry with clearer instructions
   - Maximum 2 retries per agent
   - Fallback to alternative approach if needed

2. INCOMPLETE RESULTS:
   - Request additional information if results are partial
   - Combine multiple agent outputs when necessary
   - Provide best-effort answers with caveats

3. VALIDATION:
   - Verify agent outputs for consistency
   - Cross-check numerical results
   - Flag any contradictions or anomalies

4. GRACEFUL DEGRADATION:
   - If unable to complete fully, provide partial results
   - Clearly communicate limitations
   - Suggest alternative approaches

RECOVERY ACTIONS:
- Rephrase and retry failed tasks
- Use alternative agents when primary fails
- Aggregate partial results into useful output
- Always maintain transparency about issues
"""

print("강건한 Supervisor 프롬프트 정의 완료")
print("\n주요 오류 처리 전략:")
print("  1. 실패 시 재시도 (최대 2회)")
print("  2. 부분 결과 처리")
print("  3. 결과 검증")
print("  4. 우아한 성능 저하 (Graceful Degradation)")

# %% [markdown]
# ### 성능 최적화 팁
#
# 멀티에이전트 시스템의 성능을 향상시키기 위한 전략들입니다.

# %%
# 성능 최적화 구성
optimization_tips = {
    "병렬 처리": {
        "설명": "독립적인 작업은 동시에 여러 에이전트에게 위임",
        "예시": "여러 회사의 정보를 각각 다른 research agent에게 동시 요청",
        "주의": "상태 충돌 방지를 위한 적절한 동기화 필요",
    },
    "캐싱": {
        "설명": "반복되는 요청 결과를 캐시하여 재사용",
        "예시": "자주 조회되는 정보는 메모리에 저장",
        "주의": "시간에 민감한 데이터는 TTL 설정 필요",
    },
    "작업 배치": {
        "설명": "유사한 작업들을 묶어서 한 번에 처리",
        "예시": "여러 계산을 모아서 math agent에게 한 번에 전달",
        "주의": "배치 크기와 지연시간의 균형 고려",
    },
    "조기 종료": {
        "설명": "충분한 정보가 수집되면 즉시 종료",
        "예시": "필요한 답을 찾으면 추가 검색 중단",
        "주의": "완전성과 효율성의 균형 필요",
    },
}

print("성능 최적화 전략\n")
for strategy, details in optimization_tips.items():
    print(f"\n[{strategy}]")
    print(f"  설명: {details['설명']}")
    print(f"  예시: {details['예시']}")
    print(f"  주의: {details['주의']}")

# %% [markdown]
# ---
#
# ## 정리
#
# LangGraph Supervisor Pattern의 핵심 개념을 모두 학습했습니다.
#
# ### 학습 완료 체크리스트
#
# **이론 학습**
#
# - Supervisor Pattern 개념 이해 - 실무 조직과 유사한 멀티에이전트 아키텍처
# - 에이전트 전문화 원리 - 각 에이전트의 명확한 역할 분담
# - 핸드오프 메커니즘 - 에이전트 간 작업 전달 방식
#
# **실습 완료**
#
# - 전문 에이전트 구축 - Research 및 Math 에이전트 생성
# - 라이브러리 활용 - langgraph-supervisor 패턴 활용
# - 커스텀 구현 - 핸드오프와 Command를 사용한 직접 구현
# - 고급 작업 위임 - 명시적 작업 설명과 Send 활용
#
# **실무 역량**
#
# - 베스트 프랙티스 - 오류 처리, 성능 최적화 전략
# - 아키텍처 설계 - 확장 가능한 멀티에이전트 시스템 구조
#
# ### 실무 적용 방안
#
# **비즈니스 활용 사례**
#
# 1. **고객 지원 시스템**: FAQ Agent + 기술 지원 Agent + 결제 문의 Agent
# 2. **마케팅 분석**: 시장조사 Agent + 데이터분석 Agent + 보고서 작성 Agent  
# 3. **재무 분석**: 데이터 수집 Agent + 계산 Agent + 리스크 분석 Agent
#
# **확장 아이디어**
#
# 1. **더 많은 전문 에이전트 추가** - Data Analysis Agent, Content Writer Agent, Code Generator Agent
# 2. **계층적 Supervisor 구조** - Main Supervisor → Sub-Supervisors → Workers
# 3. **동적 에이전트 관리** - 작업 유형에 따라 필요한 에이전트만 동적 생성
#
# ### 핵심 인사이트
#
# - 명확한 역할 분리가 성공의 열쇠입니다. 각 에이전트가 자신의 전문 분야에만 집중해야 합니다.
# - Supervisor는 조정만, 실제 작업은 전문 에이전트가 수행합니다.
# - 핸드오프로 유연한 워크플로우를 구성합니다. 복잡한 작업도 단계별로 분해하여 처리할 수 있습니다.
# - 명시적 지시로 정확성을 향상시킵니다. 애매한 지시보다는 구체적인 작업 명세가 효과적입니다.
# - 오류 처리로 신뢰성을 확보합니다. 실무에서는 반드시 예외 상황 대비책이 필요합니다.
#
# ### 다음 단계 추천
#
# 이제 여러분은 **엔터프라이즈급 멀티에이전트 시스템** 을 구축할 실력을 갖추셨습니다. 다음 튜토리얼에서는 **메모리 관리** 와 **대화 맥락 유지** 에 대해 학습해보세요.
