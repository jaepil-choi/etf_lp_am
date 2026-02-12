# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-Agent 시스템
#
# Multi-agent 시스템은 복잡한 애플리케이션을 여러 전문화된 에이전트로 나누어 함께 문제를 해결하는 아키텍처입니다. 단일 에이전트가 모든 작업을 처리하는 대신, 더 작고 집중된 에이전트들을 조정된 워크플로우로 구성하여 각 에이전트가 자신의 전문 분야에 집중할 수 있게 합니다.
#
# 이 튜토리얼에서는 LangGraph의 `create_react_agent` 함수를 사용하여 multi-agent 시스템을 구축하는 방법을 학습합니다. 특히 **Tool Calling 패턴(Subagents)**을 중심으로 실용적인 예제를 통해 설명합니다.
#
# > **참고 문서**: [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
#
# ---
#
# ## 학습 목표
#
# 이 튜토리얼에서는 다음 내용을 학습합니다:
#
# - Multi-agent 시스템의 개념과 장점 이해
# - Tool Calling 패턴(Subagents)을 사용한 에이전트 조정
# - 실용적인 고객 지원 시스템 구축

# %% [markdown]
# ## Multi-agent가 유용한 경우
#
# Multi-agent 아키텍처는 다음과 같은 상황에서 특히 유용합니다:
#
# - **도구 과부하**: 단일 에이전트가 너무 많은 도구를 가지고 있어 어떤 것을 사용할지 잘못 결정하는 경우
# - **컨텍스트 오버플로우**: 컨텍스트 또는 메모리가 한 에이전트가 효과적으로 추적하기에 너무 큰 경우
# - **전문화 필요**: 작업에 전문화가 필요한 경우 (예: 계획자, 연구자, 수학 전문가)
#
# 예를 들어, 고객 지원 시스템에서 기술 지원, 청구 문의, 일반 문의를 모두 처리하는 단일 에이전트보다 각 분야의 전문 에이전트를 두는 것이 더 효과적입니다.

# %% [markdown]
# ---
#
# ## Multi-agent 패턴
#
# Multi-agent 시스템을 구축하는 두 가지 주요 패턴이 있습니다. 각 패턴은 서로 다른 제어 흐름과 사용 사례에 적합합니다.
#
# | 패턴 | 작동 방식 | 제어 흐름 | 사용 사례 |
# |------|----------|----------|----------|
# | **Tool Calling (Subagents)** | 수퍼바이저 에이전트가 다른 에이전트를 도구로 호출. 도구 에이전트는 사용자와 직접 대화하지 않고 작업을 실행하고 결과를 반환 | 중앙 집중식: 모든 라우팅이 호출 에이전트를 통과 | 작업 오케스트레이션, 구조화된 워크플로우 |
# | **Handoffs** | 현재 에이전트가 다른 에이전트로 제어를 전달. 활성 에이전트가 변경되고 사용자는 새 에이전트와 직접 상호작용 | 분산형: 에이전트가 활성 에이전트를 변경 가능 | 다중 도메인 대화, 전문가 인계 |
#
# > **참고 문서**: [LangGraph Subagents 패턴](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#subagents)

# %% [markdown]
# ### 패턴 선택 가이드
#
# 어떤 패턴을 선택할지 결정할 때 다음 질문들을 고려하세요. 프로젝트의 요구사항에 따라 적합한 패턴이 달라집니다.
#
# | 질문 | Tool Calling | Handoffs |
# |------|-------------|----------|
# | 워크플로우에 대한 중앙 집중식 제어가 필요한가? | 적합함 | 부적합 |
# | 에이전트가 사용자와 직접 상호작용하기를 원하는가? | 부적합 | 적합함 |
# | 전문가 간 복잡하고 인간과 같은 대화가 필요한가? | 제한적 | 강력함 |
#
# 두 패턴을 혼합할 수도 있습니다. 에이전트 전환에는 Handoffs를 사용하고, 각 에이전트가 전문 작업을 위해 하위 에이전트를 도구로 호출하도록 구성할 수 있습니다.

# %% [markdown]
# ---
#
# ## 환경 설정
#
# Multi-agent 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드하고, `langchain_teddynote`의 로깅 기능을 활성화하여 LangSmith에서 실행 추적을 확인할 수 있도록 합니다.
#
# LangSmith 추적을 활성화하면 에이전트 간의 호출 관계, 도구 실행 순서, 응답 생성 과정 등을 시각적으로 디버깅할 수 있어 개발에 큰 도움이 됩니다.
#
# 아래 코드는 환경 변수를 로드하고 LangSmith 프로젝트를 설정합니다.

# %%
from dotenv import load_dotenv
from langchain_teddynote import logging

# 환경 변수 로드 (override=True: 기존 환경변수 덮어쓰기)
load_dotenv(override=True)

# LangSmith 추적을 위한 프로젝트 이름 설정
logging.langsmith("LangGraph-Multi-Agent")

# %% [markdown]
# ---
#
# ## Tool Calling 패턴 (Subagents)
#
# Tool Calling 패턴에서는 하나의 에이전트(수퍼바이저 또는 컨트롤러)가 다른 에이전트들을 필요할 때 호출할 수 있는 도구로 취급합니다. 수퍼바이저는 전체 오케스트레이션을 관리하고, 하위 에이전트들은 특정 작업을 수행한 후 결과를 수퍼바이저에게 반환합니다.
#
# 이 패턴의 핵심 아이디어는 다음과 같습니다:
#
# - **중앙 집중식 제어**: 모든 라우팅 결정이 수퍼바이저를 통해 이루어집니다
# - **하위 에이전트 캡슐화**: 각 하위 에이전트는 `@tool` 데코레이터로 래핑되어 도구로 노출됩니다
# - **상태 비공유**: 하위 에이전트는 자체 상태를 유지하지 않고, 모든 메모리는 수퍼바이저에서 관리됩니다
#
# > **참고 문서**: [LangGraph Subagents](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#subagents)

# %% [markdown]
# ### 기본 구현
#
# 가장 기본적인 Tool Calling 패턴을 구현해보겠습니다. 수학 전문 하위 에이전트를 만들고, 이를 도구로 래핑하여 메인 에이전트가 호출할 수 있도록 합니다.
#
# 먼저 하위 에이전트가 사용할 도구를 정의한 후, 해당 에이전트를 생성합니다. 그런 다음 이 에이전트를 `@tool` 데코레이터로 래핑하여 수퍼바이저가 호출할 수 있는 도구로 변환합니다.
#
# 아래 코드는 수학 전문 하위 에이전트를 생성하고 이를 도구로 래핑하는 과정을 보여줍니다.

# %%
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# 모델 설정
model = ChatOpenAI(model="gpt-4.1-mini")


@tool
def calculator(expression: str) -> str:
    """수학 표현식 계산 도구

    주어진 수학 표현식을 계산하여 결과를 반환합니다.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"


# 수학 전문 하위 에이전트 생성
math_agent = create_react_agent(
    model=model,
    tools=[calculator],
    prompt="당신은 수학 전문가입니다. 수학 문제를 정확하게 해결하세요. 한국어로 답변하세요.",
)


@tool(
    "math_expert",
    description="수학 계산과 문제 해결에 이 도구를 사용하세요. 산술 연산, 방정식 풀이 등을 수행할 수 있습니다.",
)
def call_math_agent(query: str) -> str:
    """수학 전문 에이전트 호출 도구

    수학 관련 질문을 수학 전문 에이전트에게 전달하고 결과를 반환합니다.
    """
    result = math_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# 메인 에이전트 (수퍼바이저) 생성
main_agent = create_react_agent(
    model=model,
    tools=[call_math_agent],
    prompt="당신은 도움이 되는 어시스턴트입니다. 사용자가 수학 질문을 하면 math_expert 도구를 사용하세요. 한국어로 답변하세요.",
)

print("메인 에이전트와 수학 하위 에이전트가 생성되었습니다.")

# %% [markdown]
# ### 기본 구현 테스트
#
# 생성한 에이전트를 테스트해보겠습니다. 수학 질문을 하면 메인 에이전트가 자동으로 수학 전문 하위 에이전트를 호출합니다.
#
# `stream_graph` 함수를 사용하면 에이전트의 추론 과정과 도구 호출을 실시간으로 확인할 수 있습니다.
#
# 아래 코드는 메인 에이전트를 통해 수학 문제를 해결하는 예시입니다.

# %%
from langchain_teddynote.messages import stream_graph
from langchain_core.messages import HumanMessage

# 수학 질문 테스트: 메인 에이전트가 math_expert 도구를 호출
stream_graph(
    main_agent,
    inputs={"messages": [HumanMessage(content="15 곱하기 23에 47을 더하면 얼마인가요?")]},
)


# %% [markdown]
# ---
#
# ### 다중 하위 에이전트
#
# 실제 시스템에서는 여러 전문 에이전트가 필요한 경우가 많습니다. 이 섹션에서는 수학, 연구, 글쓰기 세 가지 전문 에이전트를 만들고 수퍼바이저가 이들을 조정하는 구조를 구축합니다.
#
# 각 하위 에이전트는 독립적으로 자신의 도구와 시스템 프롬프트를 가지며, 수퍼바이저는 사용자의 요청을 분석하여 적절한 에이전트를 선택합니다.
#
# 아래 코드는 세 가지 전문 에이전트와 이를 조정하는 수퍼바이저를 생성합니다.

# %%
# 연구 전문 하위 에이전트용 도구
@tool
def search_web(query: str) -> str:
    """웹 검색 도구

    웹에서 정보를 검색하고 결과를 반환합니다.
    """
    # 실제 구현에서는 웹 검색 API를 호출
    return f"'{query}'에 대한 검색 결과: [검색 결과가 여기에 표시됩니다]"


# 연구 전문 에이전트 생성
research_agent = create_react_agent(
    model=model,
    tools=[search_web],
    prompt="당신은 연구 전문가입니다. 주제에 대한 정보를 찾고 요약하세요. 한국어로 답변하세요.",
)


@tool(
    "researcher",
    description="주제를 연구하고 정보를 수집할 때 이 도구를 사용하세요. 웹 검색과 정보 요약을 수행할 수 있습니다.",
)
def call_research_agent(query: str) -> str:
    """연구 전문 에이전트 호출 도구

    연구 관련 질문을 연구 전문 에이전트에게 전달하고 결과를 반환합니다.
    """
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# 글쓰기 전문 하위 에이전트 생성
writing_agent = create_react_agent(
    model=model,
    tools=[],
    prompt="당신은 전문 작가입니다. 구조화되고 매력적인 콘텐츠를 작성하세요. 한국어로 작성하세요.",
)


@tool(
    "writer",
    description="기사, 보고서 또는 기타 문서를 작성할 때 이 도구를 사용하세요. 창의적이고 구조화된 글을 작성할 수 있습니다.",
)
def call_writing_agent(query: str) -> str:
    """글쓰기 전문 에이전트 호출 도구

    글쓰기 관련 요청을 글쓰기 전문 에이전트에게 전달하고 결과를 반환합니다.
    """
    result = writing_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# 수퍼바이저 에이전트 생성: 세 가지 전문 에이전트를 도구로 사용
supervisor = create_react_agent(
    model=model,
    tools=[call_math_agent, call_research_agent, call_writing_agent],
    prompt="""당신은 전문 에이전트들을 조정하는 수퍼바이저입니다.
사용자의 요청을 분석하여 적절한 전문 에이전트를 선택하세요:
- math_expert: 수학 계산과 문제 해결
- researcher: 정보 검색과 연구
- writer: 콘텐츠 작성과 문서 생성

한국어로 답변하세요.""",
)

print("수퍼바이저와 3개의 하위 에이전트가 생성되었습니다.")

# %% [markdown]
# ### 수퍼바이저 에이전트 테스트
#
# 수퍼바이저가 다양한 유형의 요청을 올바른 하위 에이전트로 라우팅하는지 테스트합니다. 수학 작업, 연구 작업, 글쓰기 작업 각각에 대해 수퍼바이저의 판단과 결과를 확인할 수 있습니다.
#
# 아래 코드는 세 가지 다른 유형의 작업을 수퍼바이저에게 요청하는 예시입니다.

# %%
# 수학 작업 테스트
print("=" * 50)
print("수학 작업")
print("=" * 50)
stream_graph(
    supervisor,
    inputs={"messages": [HumanMessage(content="144의 제곱근을 계산해주세요")]},
)

# %%
# 연구 작업 테스트
print("=" * 50)
print("연구 작업")
print("=" * 50)
stream_graph(
    supervisor,
    inputs={
        "messages": [HumanMessage(content="Python 프로그래밍 언어의 역사에 대해 조사해주세요")]
    },
)

# %%
# 글쓰기 작업 테스트
print("=" * 50)
print("글쓰기 작업")
print("=" * 50)
stream_graph(
    supervisor,
    inputs={"messages": [HumanMessage(content="AI 에이전트에 대한 짧은 소개글을 작성해주세요")]},
)

# %% [markdown]
# ---
#
# ## 컨텍스트 엔지니어링
#
# Multi-agent 설계의 핵심은 **컨텍스트 엔지니어링**입니다. 이는 각 에이전트가 보는 정보를 결정하고, 에이전트 간에 어떤 정보를 전달할지 제어하는 것을 의미합니다.
#
# 컨텍스트 엔지니어링의 주요 결정 사항:
#
# - **입력 커스터마이징**: 하위 에이전트에 전체 대화 기록을 전달할지, 현재 쿼리만 전달할지
# - **출력 제어**: 하위 에이전트의 결과만 반환할지, 추가 메타데이터도 포함할지
# - **상태 공유**: 에이전트 간에 어떤 상태 정보를 공유할지
#
# > **참고 문서**: [LangGraph Context Engineering](https://langchain-ai.github.io/langgraph/concepts/context-engineering/)

# %% [markdown]
# ### 하위 에이전트 입력 제어
#
# 하위 에이전트에 전달되는 입력을 커스터마이징하면 더 풍부한 컨텍스트를 제공할 수 있습니다. 사용자 컨텍스트, 작업 이력 등 추가 정보를 에이전트에 전달하여 더 맞춤화된 응답을 생성할 수 있습니다.
#
# 아래 코드는 사용자 컨텍스트를 포함하여 하위 에이전트를 호출하는 예시입니다.

# %%
# 컨텍스트를 활용하는 하위 에이전트 생성
context_aware_agent = create_react_agent(
    model=model,
    tools=[],
    prompt="당신은 컨텍스트를 인식하는 어시스턴트입니다. 사용자 컨텍스트와 이전 작업 이력을 고려하여 답변하세요. 한국어로 답변하세요.",
)


@tool(
    "context_aware_tool",
    description="상태에서 컨텍스트를 활용하는 도구입니다. 사용자 정보와 작업 이력을 고려한 답변을 제공합니다.",
)
def call_context_aware_agent(query: str, user_context: str = "", task_history: str = "") -> str:
    """컨텍스트를 포함하여 에이전트를 호출합니다.

    Args:
        query: 사용자 질문
        user_context: 사용자 관련 컨텍스트
        task_history: 이전 작업 이력

    Returns:
        에이전트 응답
    """
    # 컨텍스트를 포함한 향상된 쿼리 구성
    enhanced_query = f"""
사용자 컨텍스트: {user_context}
이전 작업 이력: {task_history}

현재 질문: {query}
"""
    result = context_aware_agent.invoke(
        {"messages": [{"role": "user", "content": enhanced_query}]}
    )
    return result["messages"][-1].content


# 컨텍스트를 사용하는 메인 에이전트
main_with_context = create_react_agent(
    model=model,
    tools=[call_context_aware_agent],
    prompt="당신은 도움이 되는 어시스턴트입니다. context_aware_tool을 사용하여 사용자에게 맞춤화된 답변을 제공하세요. 한국어로 답변하세요.",
)

print("컨텍스트 인식 에이전트가 생성되었습니다.")

# %% [markdown]
# ### 컨텍스트 인식 에이전트 테스트
#
# 사용자 컨텍스트와 작업 이력을 포함하여 에이전트를 호출합니다. 에이전트는 이 추가 정보를 활용하여 더 맞춤화된 응답을 생성합니다.
#
# 아래 코드는 프리미엄 사용자 컨텍스트와 이전 작업 이력을 함께 전달하는 예시입니다.

# %%
# 컨텍스트를 포함한 테스트
stream_graph(
    main_with_context,
    inputs={
        "messages": [
            HumanMessage(
                content="다음 프로젝트를 도와주세요. user_context: 프리미엄 사용자, 상세한 설명 선호. task_history: Python 기초 학습, 데이터 분석 프로젝트"
            )
        ],
    },
)

# %% [markdown]
# ### 하위 에이전트 출력 제어
#
# 하위 에이전트의 출력을 제어하면 결과와 함께 메타데이터를 반환할 수 있습니다. 에이전트 호출의 결과뿐만 아니라 실행 정보, 타임스탬프, 소스 등의 메타데이터가 필요한 경우에 유용합니다.
#
# 아래 코드는 분석 결과와 메타데이터를 함께 반환하는 에이전트를 구현합니다.

# %%
import datetime

# 메타데이터를 생성하는 분석 에이전트
analysis_agent = create_react_agent(
    model=model,
    tools=[],
    prompt="쿼리를 분석하고 인사이트를 제공하세요. 한국어로 답변하세요.",
)


@tool("analyzer", description="데이터를 분석하고 메타데이터와 함께 결과를 반환합니다.")
def call_analyzer(query: str) -> str:
    """분석 에이전트를 호출하고 메타데이터와 함께 결과를 반환합니다.

    Args:
        query: 분석할 쿼리

    Returns:
        분석 결과와 메타데이터를 포함한 문자열
    """
    # 분석 에이전트 호출
    result = analysis_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    # 메타데이터 생성
    metadata = {
        "query_length": len(query),
        "timestamp": datetime.datetime.now().isoformat(),
        "agent": "analyzer",
    }

    # 결과와 메타데이터를 함께 반환
    return f"""분석 결과:
{result["messages"][-1].content}

메타데이터:
- 쿼리 길이: {metadata['query_length']}
- 타임스탬프: {metadata['timestamp']}
- 에이전트: {metadata['agent']}"""


# 메타데이터를 활용하는 메인 에이전트
main_with_metadata = create_react_agent(
    model=model,
    tools=[call_analyzer],
    prompt="당신은 분석 어시스턴트입니다. analyzer 도구를 사용하여 데이터를 분석하세요. 한국어로 답변하세요.",
)

print("메타데이터를 반환하는 분석 에이전트가 생성되었습니다.")

# %% [markdown]
# ### 출력 제어 테스트
#
# 분석 에이전트를 호출하고 결과와 메타데이터를 확인합니다. 메타데이터에는 쿼리 길이, 타임스탬프, 에이전트 정보 등이 포함됩니다.
#
# 아래 코드는 분석 요청을 수행하고 결과를 확인하는 예시입니다.

# %%
# 분석 테스트
stream_graph(
    main_with_metadata,
    inputs={
        "messages": [
            HumanMessage(content="AI 기술의 발전 동향을 분석해주세요")
        ],
    },
)


# %% [markdown]
# ---
#
# ## 실용적인 예제: 고객 지원 시스템
#
# 이제 실제 비즈니스 시나리오에 적용할 수 있는 고객 지원 시스템을 구축해보겠습니다. 이 시스템은 세 가지 전문 에이전트로 구성됩니다:
#
# - **기술 지원**: 시스템 문제, 오류, 기술적 질문 처리
# - **청구 지원**: 인보이스, 결제, 환불 관련 문의 처리
# - **일반 지원**: 제품 정보, 서비스 안내 등 일반 문의 처리
#
# 수퍼바이저는 고객의 문의 내용을 분석하여 가장 적합한 전문 에이전트로 라우팅합니다.
#
# 아래 코드는 전체 고객 지원 시스템을 구현합니다.

# %%
# 기술 지원 에이전트용 도구
@tool
def check_system_status(system: str) -> str:
    """시스템 상태 확인 도구

    지정된 시스템의 상태를 확인하고 결과를 반환합니다.
    """
    return f"시스템 '{system}'이(가) 정상 작동 중입니다."


@tool
def restart_service(service: str) -> str:
    """서비스 재시작 도구

    지정된 서비스를 재시작하고 결과를 반환합니다.
    """
    return f"서비스 '{service}'이(가) 성공적으로 재시작되었습니다."


# 기술 지원 에이전트 생성
tech_support = create_react_agent(
    model=model,
    tools=[check_system_status, restart_service],
    prompt="""당신은 기술 지원 전문가입니다.
사용자의 기술적 문제를 해결하세요.
해결 방법을 제안하기 전에 항상 시스템 상태를 먼저 확인하세요.
한국어로 답변하세요.""",
)


@tool(
    "tech_support",
    description="기술 지원 문제, 문제 해결, 시스템 오류를 처리합니다. 애플리케이션 충돌, 연결 문제, 기술적 질문에 사용하세요.",
)
def call_tech_support(issue: str) -> str:
    """기술 지원 에이전트 호출 도구"""
    result = tech_support.invoke({"messages": [{"role": "user", "content": issue}]})
    return result["messages"][-1].content


# 청구 지원 에이전트용 도구
@tool
def check_invoice(invoice_id: str) -> str:
    """인보이스 확인 도구

    인보이스 ID를 사용하여 세부 정보를 조회합니다.
    """
    return f"인보이스 {invoice_id}: 금액 100,000원, 상태: 결제 완료"


@tool
def process_refund(order_id: str) -> str:
    """환불 처리 도구

    주문 ID를 사용하여 환불을 처리합니다.
    """
    return f"주문 {order_id}에 대한 환불이 처리되었습니다."


# 청구 지원 에이전트 생성
billing_support = create_react_agent(
    model=model,
    tools=[check_invoice, process_refund],
    prompt="""당신은 청구 지원 전문가입니다.
인보이스, 결제, 환불과 관련된 문의를 처리하세요.
환불을 처리하기 전에 항상 인보이스 세부 정보를 먼저 확인하세요.
한국어로 답변하세요.""",
)


@tool(
    "billing_support",
    description="청구 질문, 인보이스, 결제, 환불을 처리합니다. 금액, 결제 상태, 환불 요청에 사용하세요.",
)
def call_billing_support(issue: str) -> str:
    """청구 지원 에이전트 호출 도구"""
    result = billing_support.invoke({"messages": [{"role": "user", "content": issue}]})
    return result["messages"][-1].content


# 일반 지원 에이전트 생성
general_support = create_react_agent(
    model=model,
    tools=[],
    prompt="""당신은 일반 고객 지원 담당자입니다.
제품과 서비스에 대한 일반적인 질문에 답변하세요.
친절하고 도움이 되도록 하세요.
한국어로 답변하세요.""",
)


@tool(
    "general_support",
    description="제품, 서비스, 회사 정보에 대한 일반 문의를 처리합니다. 영업 시간, 제품 기능, 일반적인 안내에 사용하세요.",
)
def call_general_support(question: str) -> str:
    """일반 지원 에이전트 호출 도구"""
    result = general_support.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].content


# 수퍼바이저 고객 지원 에이전트 생성
customer_support = create_react_agent(
    model=model,
    tools=[call_tech_support, call_billing_support, call_general_support],
    prompt="""당신은 고객 지원 수퍼바이저입니다.
고객 문의를 분석하여 적절한 전문가에게 라우팅하세요:
- tech_support: 기술 문제, 오류, 시스템 문제
- billing_support: 인보이스, 결제, 환불
- general_support: 일반 질문, 제품 정보

고객의 문제를 파악하고 적절한 전문가를 선택하세요.
한국어로 답변하세요.""",
)

print("고객 지원 시스템이 생성되었습니다.")

# %% [markdown]
# ### 고객 지원 시스템 테스트
#
# 다양한 유형의 고객 문의를 테스트하여 수퍼바이저가 올바른 전문 에이전트로 라우팅하는지 확인합니다. 각 테스트에서 수퍼바이저의 라우팅 결정과 전문 에이전트의 응답을 확인할 수 있습니다.
#
# 아래 코드는 기술, 청구, 일반 문의 각각에 대한 테스트입니다.

# %%
# 기술 문제 테스트
print("=" * 50)
print("기술 문제 테스트")
print("=" * 50)
stream_graph(
    customer_support,
    inputs={
        "messages": [
            HumanMessage(content="애플리케이션이 계속 충돌합니다. 도와주실 수 있나요?")
        ]
    },
)

# %%
# 청구 문제 테스트
print("=" * 50)
print("청구 문제 테스트")
print("=" * 50)
stream_graph(
    customer_support,
    inputs={
        "messages": [HumanMessage(content="주문 #12345에 대해 환불을 요청하고 싶습니다")]
    },
)

# %%
# 일반 문의 테스트
print("=" * 50)
print("일반 문의 테스트")
print("=" * 50)
stream_graph(
    customer_support,
    inputs={"messages": [HumanMessage(content="영업 시간이 어떻게 되나요?")]},
)


# %% [markdown]
# ---
#
# ## 고급 패턴: 계층적 에이전트
#
# 더 복잡한 시스템에서는 에이전트가 여러 레벨의 계층 구조를 가질 수 있습니다. 최상위 에이전트가 중간 레벨 에이전트를 호출하고, 중간 레벨 에이전트가 다시 하위 에이전트를 호출하는 구조입니다.
#
# 이 패턴은 다음과 같은 상황에서 유용합니다:
#
# - 복잡한 문제를 단계적으로 분해해야 하는 경우
# - 각 레벨에서 다른 수준의 추상화가 필요한 경우
# - 팀 구조를 반영한 에이전트 조직이 필요한 경우
#
# 아래 코드는 3단계 계층 구조의 수학 에이전트 시스템을 구현합니다.

# %%
# Level 3: 기본 작업 에이전트 (최하위)
@tool
def add_numbers(a: int, b: int) -> int:
    """덧셈 도구

    두 숫자를 더합니다.
    """
    return a + b


# 기본 수학 에이전트 생성
basic_math = create_react_agent(
    model=model,
    tools=[add_numbers],
    prompt="당신은 기본 산술 연산을 수행합니다. 한국어로 답변하세요.",
)


@tool("basic_math", description="덧셈, 뺄셈 등 기본 산술 연산을 수행합니다.")
def call_basic_math(query: str) -> str:
    """기본 수학 에이전트 호출 도구"""
    result = basic_math.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# Level 2: 중간 수준 에이전트
intermediate_math = create_react_agent(
    model=model,
    tools=[call_basic_math],
    prompt="당신은 기본 연산을 사용하여 중간 수준의 수학 문제를 해결합니다. 한국어로 답변하세요.",
)


@tool("intermediate_math", description="중간 수준의 수학 문제를 해결합니다.")
def call_intermediate_math(query: str) -> str:
    """중간 수준 수학 에이전트 호출 도구"""
    result = intermediate_math.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


# Level 1: 최상위 에이전트
advanced_math = create_react_agent(
    model=model,
    tools=[call_intermediate_math],
    prompt="당신은 복잡한 수학 문제를 더 간단한 부분으로 나누어 해결합니다. 한국어로 답변하세요.",
)

print("3단계 계층 구조의 수학 에이전트가 생성되었습니다.")

# %% [markdown]
# ### 계층적 에이전트 테스트
#
# 계층적 에이전트 시스템을 테스트합니다. 최상위 에이전트가 문제를 분해하고, 중간 에이전트를 통해 기본 에이전트까지 단계적으로 호출이 이루어집니다.
#
# 아래 코드는 복합 산술 문제를 계층적으로 해결하는 예시입니다.

# %%
# 계층적 에이전트 테스트
stream_graph(
    advanced_math,
    inputs={"messages": [HumanMessage(content="(5 + 3) + (10 + 7)을 계산해주세요")]},
)


# %% [markdown]
# ---
#
# ## 모범 사례
#
# Multi-agent 시스템을 효과적으로 구축하기 위한 모범 사례들을 소개합니다. 이러한 패턴들을 따르면 더 안정적이고 유지보수하기 쉬운 시스템을 만들 수 있습니다.

# %% [markdown]
# ### 1. 명확한 도구 설명
#
# 하위 에이전트의 이름과 설명을 명확하게 작성하세요. 수퍼바이저가 올바른 에이전트를 선택할 수 있도록 도구가 무엇을 할 수 있고 무엇을 할 수 없는지 구체적으로 설명해야 합니다.
#
# 아래 코드는 좋은 도구 설명과 나쁜 도구 설명의 예시를 보여줍니다.

# %%
# 좋은 예: 명확하고 구체적인 설명
@tool(
    "sql_expert",
    description="""데이터베이스 쿼리와 SQL 관련 작업에 사용하세요.

가능한 작업:
- SELECT, INSERT, UPDATE 쿼리 작성
- 데이터베이스 성능 최적화
- 쿼리 실행 계획 설명

사용하지 말아야 할 경우:
- 일반 프로그래밍 질문 (programmer 사용)
- 데이터 분석 (data_analyst 사용)""",
)
def call_sql_expert_good(query: str) -> str:
    """SQL 전문가 에이전트 호출 도구 (좋은 예시)"""
    pass


# 나쁜 예: 모호하고 불명확한 설명
@tool("helper", description="여러 가지를 도와줍니다")
def call_helper_bad(query: str) -> str:
    """헬퍼 호출 도구 (나쁜 예시)"""
    pass


print("도구 설명 예시가 정의되었습니다.")


# %% [markdown]
# ### 2. 적절한 컨텍스트 전달
#
# 하위 에이전트에 필요한 컨텍스트만 전달하세요. 불필요한 정보를 전달하면 에이전트의 혼란을 야기하고 토큰 사용량이 증가합니다.
#
# 아래 코드는 컨텍스트 전달의 좋은 예와 나쁜 예를 보여줍니다.

# %%
# 좋은 예: 필요한 컨텍스트만 전달
def call_subagent_good(query: str, user_id: str, task_type: str) -> str:
    """필요한 컨텍스트만 추출하여 에이전트를 호출합니다 (좋은 예시).

    필요한 정보만 선택적으로 추출하여 전달합니다.
    """
    relevant_context = {
        "user_id": user_id,
        "task_type": task_type,
    }
    return f"컨텍스트: {relevant_context}로 처리됨"


# 나쁜 예: 전체 상태를 무분별하게 전달
def call_subagent_bad(query: str, entire_state: dict) -> str:
    """전체 상태를 전달하면 불필요한 정보가 포함됩니다 (나쁜 예시).

    전체 상태를 전달하면 토큰 낭비와 혼란을 야기합니다.
    """
    return f"전체 상태로 처리됨"


print("컨텍스트 전달 예시가 정의되었습니다.")


# %% [markdown]
# ### 3. 결과 포맷 표준화
#
# 하위 에이전트의 출력 형식을 일관되게 유지하세요. 표준화된 응답 형식을 사용하면 수퍼바이저가 결과를 더 쉽게 처리할 수 있습니다.
#
# 아래 코드는 응답 형식을 표준화하는 유틸리티 함수와 사용 예시입니다.

# %%
def standardize_response(agent_result: dict) -> str:
    """에이전트 응답 표준화 함수

    에이전트의 응답을 표준 형식으로 변환합니다.
    """
    content = agent_result["messages"][-1].content

    # 표준화된 형식으로 반환
    return f"[에이전트 응답]\n{content}\n[응답 끝]"


@tool
def call_standardized_agent(query: str) -> str:
    """표준화된 형식으로 응답을 반환하는 도구"""
    # 실제 구현에서는 에이전트 호출
    mock_result = {"messages": [{"content": "처리된 결과입니다."}]}
    return standardize_response(mock_result)


print("응답 표준화 함수가 정의되었습니다.")
print(call_standardized_agent.invoke("테스트 쿼리"))


# %% [markdown]
# ### 4. 에러 처리
#
# 하위 에이전트 호출 시 적절한 에러 처리를 구현하세요. 에러가 발생해도 시스템이 우아하게 실패하고 사용자에게 유용한 피드백을 제공해야 합니다.
#
# 아래 코드는 에러 처리가 포함된 에이전트 호출 패턴입니다.

# %%
@tool
def call_agent_with_error_handling(query: str) -> str:
    """에러 처리가 포함된 에이전트 호출 도구

    에러 발생 시 명확한 메시지와 대안을 제시합니다.
    """
    try:
        # 데모를 위한 시뮬레이션
        if "에러" in query:
            raise ValueError("시뮬레이션된 에러")
        return f"'{query}'에 대한 응답입니다."

    except Exception as e:
        # 에러를 명확히 보고하고 대안 제시
        return f"에이전트 호출 중 오류 발생: {str(e)}. 질문을 다시 작성해 주세요."


print("에러 처리 예시:")
print("정상 호출:", call_agent_with_error_handling.invoke("안녕하세요"))
print("에러 발생:", call_agent_with_error_handling.invoke("에러 테스트"))

# %% [markdown]
# ### 5. 성능 모니터링
#
# 각 에이전트의 성능을 추적하세요. 실행 시간, 토큰 사용량 등을 모니터링하면 병목 지점을 식별하고 시스템을 최적화할 수 있습니다.
#
# 아래 코드는 실행 시간을 측정하는 모니터링 패턴입니다.

# %%
import time


@tool
def call_monitored_agent(query: str) -> str:
    """성능 모니터링이 포함된 에이전트 호출 도구

    실행 시간을 측정하고 출력합니다.
    """
    start_time = time.time()

    # 데모를 위한 시뮬레이션
    time.sleep(0.1)  # 처리 시간 시뮬레이션
    result = f"'{query}'에 대한 응답"

    duration = time.time() - start_time
    print(f"에이전트 실행 시간: {duration:.2f}초")

    return result


print("성능 모니터링 테스트:")
print(call_monitored_agent.invoke("테스트 쿼리"))

# %% [markdown]
# ---
#
# ## 정리
#
# 이 튜토리얼에서는 LangGraph를 사용한 Multi-agent 시스템의 구축 방법을 학습했습니다. 주요 내용을 정리하면 다음과 같습니다.
#
# ### Multi-agent 시스템의 장점
#
# - **전문화**: 각 에이전트가 특정 작업에 집중하여 더 정확한 결과 제공
# - **확장성**: 새로운 에이전트를 쉽게 추가하여 기능 확장 가능
# - **유지보수성**: 독립적인 에이전트로 구성되어 관리 용이
# - **신뢰성**: 전문 에이전트가 더 나은 결정을 내림
#
# ### Tool Calling 패턴 (Subagents)
#
# - 중앙 집중식 제어: 수퍼바이저가 모든 라우팅 결정
# - 구조화된 워크플로우: 하위 에이전트는 도구로 작동
# - 상태 관리: 메모리는 수퍼바이저에서 중앙 관리
#
# ### 핵심 설계 원칙
#
# - **명확한 도구 설명**: 각 에이전트의 역할과 한계를 명확히 정의
# - **적절한 컨텍스트**: 필요한 정보만 선택적으로 전달
# - **표준화된 출력**: 일관된 응답 형식 유지
# - **에러 처리**: 우아한 실패와 유용한 피드백
# - **성능 모니터링**: 지속적인 시스템 최적화
#
# ### 추가 학습 자료
#
# 더 깊이 있는 학습을 원한다면 아래 공식 문서를 참고하세요:
#
# - [LangGraph Multi-Agent 개념](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
# - [LangGraph Subagents 패턴](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#subagents)
# - [LangGraph Handoffs 패턴](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs)
