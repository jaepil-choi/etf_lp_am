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
# # 에이전트 (Agent)
#
# 에이전트는 언어 모델(LLM)과 도구(Tool)를 결합하여 복잡한 작업을 수행하는 시스템입니다. 에이전트는 주어진 작업에 대해 추론하고, 필요한 도구를 선택하며, 목표를 향해 반복적으로 작업을 수행합니다.
#
# LangChain의 `create_agent` 함수는 프로덕션 수준의 에이전트 구현을 제공합니다. 이 함수를 사용하면 모델 선택, 도구 연동, 미들웨어 설정 등을 손쉽게 구성할 수 있습니다.
#
# > 참고 문서: [LangGraph Agents](https://docs.langchain.com/oss/python/langgraph/agents.md)

# %% [markdown]
# ## 환경 설정
#
# 에이전트 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드하고, `langchain_teddynote`의 로깅 기능을 활성화하여 LangSmith에서 실행 추적을 확인할 수 있도록 합니다.
#
# LangSmith 추적을 활성화하면 에이전트의 추론 과정, 도구 호출, 응답 생성 등을 시각적으로 디버깅할 수 있어 개발에 큰 도움이 됩니다.
#
# 아래 코드는 환경 변수를 로드하고 LangSmith 프로젝트를 설정합니다.

# %%
from dotenv import load_dotenv
from langchain_teddynote import logging

# 환경 변수 로드
load_dotenv(override=True)
# 추적을 위한 프로젝트 이름 설정
logging.langsmith("LangChain-V1-Tutorial")

# %% [markdown]
# ## 모델 (Model)
#
# 에이전트의 추론 엔진인 LLM은 `create_agent` 함수의 첫 번째 인자로 전달합니다. 가장 간단한 방법은 `provider:model` 형식의 문자열을 사용하는 것입니다. 이 방식은 빠른 프로토타이핑에 적합합니다.
#
# 아래 코드는 모델 식별자 문자열을 사용하여 기본 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent

# 모델 식별자 문자열을 사용한 간단한 방법
agent = create_agent("openai:gpt-4.1-mini", tools=[])

# %% [markdown]
# ### 모델 세부 설정
#
# 더 세밀한 제어가 필요한 경우, 모델 클래스를 직접 인스턴스화하여 다양한 옵션을 설정할 수 있습니다. `temperature`는 응답의 무작위성을, `max_tokens`는 생성할 최대 토큰 수를, `timeout`은 요청 타임아웃을 제어합니다.
#
# 아래 코드는 ChatOpenAI 클래스를 사용하여 세부 옵션이 설정된 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 모델 인스턴스를 직접 초기화하여 더 세밀한 제어
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.1,  # 응답의 무작위성 제어
    max_tokens=1000,  # 최대 생성 토큰 수
    timeout=30,  # 요청 타임아웃(초)
)

agent = create_agent(model, tools=[])

# %% [markdown]
# ### 동적 모델 선택
#
# 동적 모델 선택은 런타임에 현재 상태와 컨텍스트를 기반으로 사용할 모델을 결정하는 패턴입니다. 이를 통해 정교한 라우팅 로직과 비용 최적화가 가능합니다. 예를 들어, 간단한 질문에는 경량 모델을, 복잡한 대화에는 고급 모델을 사용할 수 있습니다.
#
# `wrap_model_call` 데코레이터를 사용하면 모델 호출 전에 요청을 검사하고 수정할 수 있는 미들웨어를 생성할 수 있습니다.
#
# ![](assets/wrap_model_call.png)
#
# 아래 코드는 대화 길이에 따라 모델을 동적으로 선택하는 예시입니다.

# %% [markdown]
# ### ModelRequest 속성
#
# `ModelRequest`는 에이전트의 모델 호출 정보를 담는 데이터 클래스로, 미들웨어에서 요청을 검사하고 수정할 때 사용됩니다. `override()` 메서드를 통해 여러 속성을 동시에 변경할 수 있습니다.
#
# 아래 코드는 ModelRequest를 사용하여 동적으로 모델과 시스템 프롬프트를 변경하는 예시입니다.

# %%
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# 기본 모델과 고급 모델 정의
basic_model = ChatOpenAI(model="gpt-4.1-mini")
advanced_model = ChatOpenAI(model="gpt-4.1")


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """대화 복잡도에 따라 모델 선택"""
    message_count = len(request.state["messages"])

    # 긴 대화에는 고급 모델 사용
    if message_count > 10:
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)


agent = create_agent(
    model=basic_model, tools=[], middleware=[dynamic_model_selection]  # 기본 모델
)

# %%
from langchain_teddynote.messages import stream_graph
from langchain_core.messages import HumanMessage

stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해줘")]
    },
)


# %% [markdown]
# **ModelRequest 주요 속성:**
#
# | 속성 | 설명 |
# |:---|:---|
# | `model` | 사용할 `BaseChatModel` 인스턴스 |
# | `system_prompt` | 시스템 프롬프트 (선택적) |
# | `messages` | 대화 메시지 리스트 (시스템 프롬프트 제외) |
# | `tool_choice` | 도구 선택 설정 |
# | `tools` | 사용 가능한 도구 리스트 |
# | `response_format` | 응답 형식 지정 |
# | `state` | 현재 에이전트 상태 (`AgentState`) |
# | `runtime` | 에이전트 런타임 정보 |
# | `model_settings` | 추가 모델 설정 (dict) |
#
# 아래 코드는 `override()` 메서드를 사용하여 여러 속성을 동시에 변경하는 예시입니다.

# %%
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """대화 복잡도에 따라 모델 선택"""
    message_count = len(request.state["messages"][-1].content)
    print(f"글자수: {message_count}")

    # 긴 대화에는 고급 모델 사용
    if message_count > 10:
        # 여러 속성 동시 변경
        new_request = request.override(
            model=advanced_model,
            system_prompt="emoji 를 사용해서 답변해줘",
            tool_choice="auto",
        )
        return handler(new_request)
    else:
        new_request = request.override(
            system_prompt="한 문장으로 간결하게 답변해줘. emoji 는 사용하지 말아줘.",
            tool_choice="auto",
            model=basic_model,
        )
        return handler(new_request)


agent = create_agent(
    model=basic_model, tools=[], middleware=[dynamic_model_selection]  # 기본 모델
)

# %% [markdown]
# ### 글자수 기반 모델 선택 테스트
#
# 아래는 글자수 10자 미만일 때의 응답입니다. 간결한 답변을 생성하도록 설정되어 있습니다.

# %%
stream_graph(agent, inputs={"messages": [HumanMessage(content="머신러닝 동작원리")]})

# %% [markdown]
# 아래는 글자수 10자 이상일 때의 응답입니다. 이모지를 사용하여 친근한 답변을 생성하도록 설정되어 있습니다.

# %%
stream_graph(
    agent,
    inputs={
        "messages": [
            HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해 주세요.")
        ]
    },
)

# %% [markdown]
# ---
#
# ## 프롬프트
#
# 에이전트의 동작을 제어하는 핵심 요소 중 하나는 시스템 프롬프트입니다. 시스템 프롬프트를 통해 에이전트의 역할, 응답 스타일, 제약 조건 등을 정의할 수 있습니다.

# %% [markdown]
# ### 시스템 프롬프트
#
# `system_prompt` 매개변수를 사용하여 에이전트의 기본 동작을 정의할 수 있습니다. 시스템 프롬프트는 모든 대화에서 일관되게 적용되며, 에이전트의 페르소나와 응답 가이드라인을 설정하는 데 사용됩니다.
#
# 아래 코드는 간결하고 정확한 응답을 생성하도록 시스템 프롬프트를 설정한 에이전트를 생성합니다.

# %%
agent = create_agent(
    "openai:gpt-4.1-mini",
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)

# %% [markdown]
# 아래는 설정된 시스템 프롬프트를 사용한 에이전트의 응답 예시입니다.

# %%
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="대한민국의 수도는 어디야?")]},
)

# %% [markdown]
# ### 동적 시스템 프롬프트 (Dynamic Prompting)
#
# 런타임 컨텍스트나 에이전트 상태를 기반으로 시스템 프롬프트를 동적으로 생성해야 하는 경우가 있습니다. `dynamic_prompt` 데코레이터를 사용하면 요청마다 다른 시스템 프롬프트를 적용할 수 있습니다.
#
# 이 기능은 사용자 역할, 언어 설정, 응답 형식 등을 런타임에 결정해야 할 때 유용합니다. `context_schema`를 정의하여 에이전트 호출 시 필요한 컨텍스트 정보를 전달할 수 있습니다.
#
# 아래 코드는 답변 형식과 길이를 동적으로 설정하는 에이전트를 생성합니다.

# %%
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    prompt_type: str
    length: int


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """사용자 역할에 따라 시스템 프롬프트 생성"""
    # 답변 형식 설정
    answer_type = (
        request.runtime.context.get("prompt_type", "default")
        if request.runtime.context
        else "default"
    )
    # 답변 길이 설정
    answer_length = (
        request.runtime.context.get("length", 20) if request.runtime.context else 20
    )
    base_prompt = "You are a helpful assistant. Answer in Korean.\n"

    # 답변 형식에 따라 시스템 프롬프트 생성(동적 프롬프팅)
    if answer_type == "default":
        return f"{base_prompt} [답변 형식] 간결하게 답변해줘. 답변 길이는 {answer_length}자 이하로 해줘."
    elif answer_type == "sns":
        return f"{base_prompt} [답변 형식] SNS 형식으로 답변해줘. 답변 길이는 {answer_length}자 이하로 해줘."
    elif answer_type == "article":
        return f"{base_prompt} [답변 형식] 뉴스 기사 형식으로 답변해줘. 답변 길이는 {answer_length}자 이하로 해줘."
    else:
        return f"{base_prompt} [답변 형식] 간결하게 답변해줘. 답변 길이는 {answer_length}자 이하로 해줘."


# 컨텍스트 스키마와 user_role_prompt 미들웨어를 사용하여 에이전트 생성
agent = create_agent(
    model="openai:gpt-4.1-mini",
    middleware=[user_role_prompt],
    context_schema=Context,
)

# %%
# 컨텍스트에 따라 시스템 프롬프트가 동적으로 설정됩니다
stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해줘")]
    },
    context=Context(prompt_type="article", length=1000),
)

# %%
stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해줘")]
    },
    context=Context(prompt_type="sns", length=50),
)

# %% [markdown]
# ---
#
# ## 구조화된 답변 출력 (Response Format)
#
# 특정 형식으로 에이전트의 출력을 반환하고 싶을 때가 있습니다. LangChain은 `response_format` 매개변수를 통해 구조화된 출력 전략을 제공합니다. 이를 통해 자연어 응답 대신 JSON 객체, Pydantic 모델 등의 형태로 구조화된 데이터를 얻을 수 있습니다.
#
# **지원 타입:**
# - **Pydantic model class**: Pydantic 모델 클래스
# - **`ToolStrategy`**: 도구 기반 구조화 전략 (대부분의 모델에서 작동)
# - **`ProviderStrategy`**: Provider 기반 구조화 전략 (OpenAI 등 지원 모델에서만 작동)
#
# 구조화된 응답은 `response_format` 설정에 따라 에이전트 상태의 `structured_response` 키에 반환됩니다.

# %% [markdown]
# ### Pydantic 모델 기반 처리
#
# Pydantic 모델을 사용하면 스키마 검증이 자동으로 이루어지며, 필드 설명을 통해 모델이 각 필드의 의미를 이해할 수 있습니다. `Field`의 `description` 매개변수는 모델이 올바른 값을 추출하는 데 도움이 됩니다.
#
# 아래 코드는 연락처 정보를 추출하는 Pydantic 모델을 정의합니다.

# %%
from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Response schema for the agent."""

    name: str = Field(description="The name of the person")
    email: str = Field(description="The email of the person")
    phone: str = Field(description="The phone number of the person")


# %%
agent = create_agent(model="openai:gpt-4.1-mini", tools=[], response_format=ContactInfo)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: 테디는 AI 엔지니어 입니다. 그의 이메일은 teddy@example.com 이고, 전화번호는 010-1234-5678 입니다.",
            }
        ]
    }
)

# %% [markdown]
# 아래에서 정형화된 출력 결과를 확인할 수 있습니다. `structured_response` 키에 Pydantic 모델 인스턴스가 반환됩니다.

# %%
result["structured_response"]

# %% [markdown]
# ### ToolStrategy
#
# `ToolStrategy`는 도구 호출(Tool Calling)을 사용하여 구조화된 출력을 생성합니다. 이 방식은 도구 호출을 지원하는 대부분의 모델에서 작동하므로 호환성이 높습니다.
#
# 아래 코드는 ToolStrategy를 사용하여 연락처 정보를 추출하는 예시입니다.

# %%
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# 응답 스키마 정의
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


agent = create_agent(
    model="openai:gpt-4.1-mini", tools=[], response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: 테디는 AI 엔지니어 입니다. 그의 이메일은 teddy@example.com 이고, 전화번호는 010-1234-5678 입니다.",
            }
        ]
    }
)

result["structured_response"]

# %% [markdown]
# ### ProviderStrategy
#
# `ProviderStrategy`는 모델 제공자의 네이티브 구조화된 출력 기능을 사용합니다. OpenAI와 같이 네이티브 구조화된 출력을 지원하는 제공자에서만 작동하지만, 더 안정적인 결과를 제공합니다.
#
# 아래 코드는 ProviderStrategy를 사용하는 예시입니다.

# %%
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="openai:gpt-4.1", response_format=ProviderStrategy(ContactInfo)
)

# %%
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: 테디는 AI 엔지니어 입니다. 그의 이메일은 teddy@example.com 이고, 전화번호는 010-1234-5678 입니다.",
            }
        ]
    }
)

# %%
result["structured_response"]

# %% [markdown]
# ---
#
# ## 미들웨어
#
# 미들웨어를 사용하면 모델 호출 전후에 커스텀 로직을 실행할 수 있습니다. `@before_model` 및 `@after_model` 데코레이터를 사용하여 모델 호출을 감싸는 훅을 정의할 수 있습니다.
#
# **미들웨어 활용 사례:**
# - 모델 호출 전 메시지 전처리 (예: 쿼리 재작성)
# - 모델 호출 후 응답 후처리 (예: 필터링, 로깅)
# - 상태 기반 동적 라우팅
#
# 아래 코드는 모델 호출 전후에 로깅을 수행하는 미들웨어 예시입니다.

# %%
from langchain.agents.middleware import (
    before_model,
    after_model,
)
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage
from langchain_teddynote.messages import invoke_graph
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime
from typing import Any, Callable


# 노드 스타일: 모델 호출 전 로깅
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(
        f"\033[95m\n\n모델 호출 전 메시지 {len(state['messages'])}개가 있습니다\033[0m"
    )
    last_message = state["messages"][-1].content
    llm = init_chat_model("openai:gpt-4.1-mini")

    query_rewrite = (
        PromptTemplate.from_template(
            "Rewrite the following query to be more understandable. Do not change the original meaning. Make it one sentence: {query}"
        )
        | llm
    )
    rewritten_query = query_rewrite.invoke({"query": last_message})

    return {"messages": [rewritten_query.content]}


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:

    print(
        f"\033[95m\n\n모델 호출 후 메시지 {len(state['messages'])}개가 있습니다\033[0m"
    )
    for i, message in enumerate(state["messages"]):
        print(f"[{i}] {message.content}")
    return None


# %%
agent = create_agent(
    "openai:gpt-4.1-mini",
    middleware=[
        log_before_model,
        log_after_model,
    ],
)

# %%
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="대한민국 수도")]},
)

# %% [markdown]
# ### 클래스 기반 미들웨어
#
# 데코레이터 대신 클래스 기반 미들웨어를 사용할 수 있습니다. `AgentMiddleware` 클래스를 상속하고 `before_model` 및 `after_model` 메서드를 오버라이드하여 커스텀 로직을 구현합니다.
#
# 클래스 기반 미들웨어는 커스텀 상태 스키마를 정의하거나 복잡한 미들웨어 로직을 구조화할 때 유용합니다.
#
# 아래 코드는 클래스 기반 미들웨어를 사용하여 커스텀 상태를 관리하는 예시입니다.

# %%
from typing import Any
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware


# 커스텀 상태 스키마 정의
class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = []

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 모델 호출 전 커스텀 로직
        pass


agent = create_agent("openai:gpt-4.1-mini", tools=[], middleware=[CustomMiddleware()])

# 에이전트는 이제 메시지 외에 추가 상태를 추적할 수 있습니다
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "I prefer technical explanations"}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},
    }
)


# %% [markdown]
# ### 모델 오류 시 재시도 로직
#
# `wrap_model_call` 데코레이터를 사용하면 모델 호출 실패 시 자동으로 재시도하는 로직을 구현할 수 있습니다. 이는 네트워크 오류나 일시적인 API 장애에 대응하는 데 유용합니다.
#
# 아래 코드는 최대 3회까지 재시도하는 미들웨어 예시입니다.

# %%
@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"오류 발생으로 {attempt + 1}/3 번째 재시도합니다: {e}")


# %%
agent = create_agent(
    "openai:gpt-4.1-minis",  # 일부러 모델 호출 실패하도록 설정(모델명 오류)
    middleware=[retry_model],
)

# %%
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="대한민국의 수도는?")]},
)
