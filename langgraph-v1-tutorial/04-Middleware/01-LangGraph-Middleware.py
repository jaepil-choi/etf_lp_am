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
# # 미들웨어(Middleware)
#
# 미들웨어는 에이전트 실행의 모든 단계를 제어하고 커스터마이징하는 강력한 방법을 제공합니다. 핵심 에이전트 루프는 모델을 호출하고, 모델이 실행할 도구를 선택한 다음, 더 이상 도구를 호출하지 않으면 종료하는 과정을 포함합니다. 미들웨어를 사용하면 이 각 단계 전후에 커스텀 로직을 삽입할 수 있습니다.
#
# ![](./assets/langgraph-middleware.avif)
#
# 미들웨어는 다음과 같은 후크(Hook)를 노출합니다:
#
# - 에이전트 시작 전/후
# - 모델 호출 전/후
# - 도구 실행 전/후
#
# 이 튜토리얼에서는 내장 미들웨어 사용법과 커스텀 미들웨어 구현 방법을 학습합니다.

# %% [markdown]
# ## 사전 준비
#
# LangGraph 미들웨어를 사용하기 위해서는 먼저 환경 변수와 LangSmith 추적을 설정해야 합니다. 환경 변수에는 OpenAI API 키, Anthropic API 키 등 LLM 서비스 인증 정보가 포함됩니다.
#
# 아래 코드는 `.env` 파일에서 환경 변수를 로드하고, LangSmith 추적을 활성화합니다.

# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %%
from langchain_teddynote import logging

logging.langsmith("LangChain-V1-Tutorial")

# %% [markdown]
# ## 미들웨어가 할 수 있는 것
#
# 미들웨어는 다음과 같은 다양한 작업을 수행할 수 있습니다.
#
# - **모니터링** - 로깅, 분석 및 디버깅으로 에이전트 동작 추적
# - **수정** - 프롬프트, 도구 선택 및 출력 형식 변환
# - **제어** - 재시도, 폴백 및 조기 종료 로직 추가
# - **강제** - 속도 제한, 가드레일 및 PII 감지 적용

# %% [markdown]
# ## 기본 예제
#
# 미들웨어를 에이전트에 추가하려면 `create_agent` 함수의 `middleware` 매개변수에 미들웨어 리스트를 전달합니다. 미들웨어는 에이전트 실행의 각 단계에서 순차적으로 실행되며, 요청과 응답을 가로채어 수정하거나 로깅할 수 있습니다.
#
# 아래 코드는 간단한 날씨 조회 도구를 정의하고, 빈 미들웨어 리스트와 함께 에이전트를 생성하는 기본 예제입니다.

# %%
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_teddynote.messages import stream_graph
from langchain_core.runnables import RunnableConfig


# 간단한 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's sunny in {city}!"


# 모델 및 에이전트 생성
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[],  # 여기에 미들웨어를 추가합니다
)

# 에이전트 실행
stream_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "What's the weather in Seoul?"}]},
    config=RunnableConfig(),
)

# %% [markdown]
# ## 내장 미들웨어
#
# LangChain은 일반적인 사용 사례를 위한 사전 구축된 미들웨어를 제공합니다. 이러한 내장 미들웨어를 사용하면 별도의 구현 없이 요약, 호출 제한, 폴백, PII 감지 등 다양한 기능을 쉽게 추가할 수 있습니다.
#
# ### 요약 (Summarization)
#
# 대화가 길어지면 토큰 제한에 도달할 수 있습니다. `SummarizationMiddleware`는 대화 기록이 특정 토큰 수를 초과할 때 자동으로 요약하여 컨텍스트 창을 효율적으로 관리합니다. 이를 통해 긴 대화에서도 중요한 정보를 유지하면서 토큰 비용을 절감할 수 있습니다.
#
# **적합한 경우:**
# - 컨텍스트 창을 초과하는 장기 실행 대화
# - 광범위한 기록이 있는 다중 턴 대화
# - 전체 대화 컨텍스트 보존이 중요한 애플리케이션
#
# 아래 코드는 `SummarizationMiddleware`를 사용하여 4000 토큰 초과 시 자동 요약을 수행하는 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",
            max_tokens_before_summary=4000,  # 4000 토큰에서 요약 트리거
            messages_to_keep=20,  # 요약 후 최근 20개 메시지 유지
        ),
    ],
)

# %%
# Summarization
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hi",
            }
        ]
    }
)

# %% [markdown]
# ### 모델 호출 제한 (ModelCallLimitMiddleware)
#
# 에이전트가 무한 루프에 빠지거나 과도한 API 호출을 하는 것을 방지하기 위해 모델 호출 수를 제한할 수 있습니다. `ModelCallLimitMiddleware`는 스레드(전체 대화) 단위와 실행(단일 호출) 단위로 모델 호출 횟수를 제한합니다. 제한에 도달하면 에이전트를 종료하거나 예외를 발생시킬 수 있습니다.
#
# **적합한 경우:**
# - 에이전트가 너무 많은 API 호출을 하는 것을 방지
# - 프로덕션 배포에 대한 비용 제어 시행
# - 특정 호출 예산 내에서 에이전트 동작 테스트
#
# 아래 코드는 스레드당 최대 3회, 실행당 최대 2회로 모델 호출을 제한하는 에이전트를 생성합니다.

# %%
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=3,  # 스레드당 최대 10회 호출 (실행 전반)
            run_limit=2,  # 실행당 최대 5회 호출 (단일 호출)
            exit_behavior="end",  # 또는 "error"로 예외 발생
        ),
    ],
)

# %% [markdown]
# ### 도구 호출 제한 (Tool Call Limit)
#
# 특정 도구 또는 모든 도구에 대한 호출 수를 제한할 수 있습니다. `ToolCallLimitMiddleware`는 비용이 많이 드는 외부 API 호출이나 데이터베이스 쿼리에 대한 속도 제한을 구현하는 데 유용합니다. 전역적으로 모든 도구에 적용하거나, 특정 도구에만 선택적으로 적용할 수 있습니다.
#
# **적합한 경우:**
# - 비용이 많이 드는 외부 API에 대한 과도한 호출 방지
# - 웹 검색 또는 데이터베이스 쿼리 제한
# - 특정 도구 사용에 대한 속도 제한 시행
#
# 아래 코드는 전역 도구 호출 제한과 특정 도구(`get_weather`)에 대한 개별 제한을 설정하는 예제입니다.

# %%
from langchain.agents.middleware import ToolCallLimitMiddleware

# 모든 도구 호출 제한
global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)

# 특정 도구 제한
weather_limiter = ToolCallLimitMiddleware(
    tool_name="get_weather",
    thread_limit=5,
    run_limit=3,
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[global_limiter, weather_limiter],
)

# %% [markdown]
# ### 모델 폴백 (Model Fallback)
#
# 기본 모델이 실패할 때 대체 모델로 자동 폴백하는 복원력 있는 에이전트를 구축할 수 있습니다. `ModelFallbackMiddleware`는 모델 제공자의 중단이나 오류 발생 시 다른 모델로 자동 전환하여 서비스 연속성을 보장합니다. OpenAI, Anthropic 등 여러 제공자에 걸친 중복성을 구현하는 데 유용합니다.
#
# **적합한 경우:**
# - 모델 중단을 처리하는 복원력 있는 에이전트 구축
# - 더 저렴한 모델로 폴백하여 비용 최적화
# - OpenAI, Anthropic 등에 걸친 제공자 중복성
#
# 아래 코드는 기본 모델(`gpt-4.1`) 실패 시 순차적으로 대체 모델로 폴백하는 에이전트를 생성합니다.

# %%
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="openai:gpt-4.1",  # 기본 모델
    tools=[get_weather],
    middleware=[
        ModelFallbackMiddleware(
            "openai:gpt-4.1-mini",  # 오류 시 먼저 시도
            "anthropic:claude-4-5-haiku",  # 그 다음 이것
        ),
    ],
)

# %% [markdown]
# ### PII 감지 (PII Detection)
#
# 대화에서 개인 식별 정보(PII)를 감지하고 처리하는 것은 규정 준수와 보안에 필수적입니다. `PIIMiddleware`는 이메일, 신용카드, 전화번호 등 다양한 PII 유형을 자동으로 감지하고, 수정(redact), 마스킹(mask), 차단(block) 등 다양한 전략으로 처리할 수 있습니다.
#
# **적합한 경우:**
# - 규정 준수 요구 사항이 있는 의료 및 금융 애플리케이션
# - 로그를 정화해야 하는 고객 서비스 에이전트
# - 민감한 사용자 데이터를 처리하는 모든 애플리케이션
#
# 아래 코드는 이메일 수정, 신용카드 마스킹, 커스텀 API 키 패턴 탐지를 적용하는 에이전트를 생성합니다.

# %%
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[
        # 사용자 입력에서 이메일 수정
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # 신용카드 마스킹 (마지막 4자리 표시)
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        # 정규식을 사용한 커스텀 PII 유형
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="mask",  # 감지 시 오류 발생
        ),
    ],
)

# PII 감지 테스트
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My credit card number is 1234-5678-9012-3456, and my API key is sk-12345678901234567890123456789012, My email is teddy@example.com. Can you help me?",
            }
        ]
    }
)
print(result["messages"][-1].content)

# %% [markdown]
# ### 도구 재시도 (Tool Retry)
#
# 외부 API 호출 시 네트워크 오류나 일시적인 실패가 발생할 수 있습니다. `ToolRetryMiddleware`는 구성 가능한 지수 백오프(exponential backoff)로 실패한 도구 호출을 자동으로 재시도합니다. 이를 통해 일시적인 오류를 우아하게 처리하는 복원력 있는 에이전트를 구축할 수 있습니다.
#
# **적합한 경우:**
# - 외부 API 호출의 일시적인 실패 처리
# - 네트워크 종속 도구의 안정성 향상
# - 일시적인 오류를 우아하게 처리하는 복원력 있는 에이전트 구축
#
# 아래 코드는 최대 3회 재시도, 지수 백오프, 무작위 지터를 적용하는 에이전트를 생성합니다.

# %%
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,  # 최대 3회 재시도
            backoff_factor=2.0,  # 지수 백오프 승수
            initial_delay=1.0,  # 1초 지연으로 시작
            max_delay=60.0,  # 지연을 60초로 제한
            jitter=True,  # 무작위 지터 추가
        ),
    ],
)

# %% [markdown]
# ## 커스텀 미들웨어
#
# 내장 미들웨어 외에도 에이전트 실행 흐름의 특정 지점에서 실행되는 커스텀 미들웨어를 직접 구현할 수 있습니다. 커스텀 미들웨어를 통해 로깅, 검증, 변환 등 비즈니스 요구사항에 맞는 다양한 기능을 추가할 수 있습니다.
#
# 미들웨어를 만드는 두 가지 방법이 있습니다:
#
# 1. **데코레이터 기반** - 단일 후크 미들웨어에 빠르고 간단
# 2. **클래스 기반** - 여러 후크가 있는 복잡한 미들웨어에 더 강력

# %% [markdown]
# ### 데코레이터 기반 미들웨어
#
# 단일 후크만 필요한 간단한 미들웨어의 경우 데코레이터가 가장 빠르고 직관적인 방법입니다. `@before_agent`, `@before_model`, `@after_model`, `@after_agent`, `@wrap_model_call`, `@wrap_tool_call`, `@dynamic_prompt` 등 다양한 데코레이터를 사용할 수 있습니다. 각 데코레이터는 에이전트 실행의 특정 단계에서 동작합니다.
#
# 아래 코드는 다양한 데코레이터를 사용하여 에이전트 시작 전/후, 모델 호출 전/후 로깅, 출력 검증, 재시도 로직, 동적 프롬프트 등을 구현하는 예제입니다.

# %%
from langchain.agents.middleware import (
    before_agent,
    before_model,
    after_model,
    after_agent,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
)
from langchain.messages import AIMessage
from langchain_teddynote.messages import invoke_graph
from langgraph.runtime import Runtime
from typing import Any, Callable


# 에이전트 시작전
@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"에이전트를 시작하기 전에 메시지 {len(state['messages'])}개가 있습니다")
    return None


# 모델 호출 전
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"모델을 호출하기 전에 메시지 {len(state['messages'])}개가 있습니다")
    return None


# 모델 호출 후
@after_model
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
        }
    return None


# 에이전트 종료 후
@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"에이전트가 종료되었습니다. 총 메시지 수: {len(state['messages'])}개")

    return None


# wrap_model_call 재시도 로직
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


# 동적 프롬프트
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {user_id}. Greeting with user's name. Be concise and friendly."


# 에이전트에서 데코레이터 사용
agent = create_agent(
    model=model,
    middleware=[
        log_before_model,
        validate_output,
        retry_model,
        personalized_prompt,
        log_before_agent,
        log_after_agent,
    ],
    tools=[get_weather],
)

invoke_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]},
    context={"user_id": "teddy"},
    config=RunnableConfig(),
)

# %% [markdown]
# ### 클래스 기반 미들웨어
#
# 여러 후크를 함께 사용하거나 상태를 유지해야 하는 복잡한 미들웨어의 경우 `AgentMiddleware` 클래스를 상속받아 구현합니다. 클래스 기반 접근 방식은 관련 로직을 하나의 클래스에 캡슐화하여 코드의 구조화와 재사용성을 높입니다.
#
# **노드 스타일 후크**는 실행 흐름의 특정 지점에서 실행됩니다:
# - `before_agent` - 에이전트 시작 전 (호출당 한 번)
# - `before_model` - 각 모델 호출 전
# - `after_model` - 각 모델 응답 후
# - `after_agent` - 에이전트 완료 후 (호출당 최대 한 번)
#
# 아래 코드는 모델 호출 전/후에 메시지 수를 로깅하는 `LoggingMiddleware` 클래스를 구현합니다.

# %%
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any


# 로깅 미들웨어
class LoggingMiddleware(AgentMiddleware):
    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content[:50]}...")
        return None


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[LoggingMiddleware()],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}
)
print("\nFinal:", result["messages"][-1].content)

# %% [markdown]
# ### 대화 길이 제한 예제
#
# 클래스 기반 미들웨어의 실용적인 예제로, 대화 메시지 수가 특정 임계값을 초과하면 대화를 종료하는 미들웨어를 구현할 수 있습니다. 이는 리소스 관리나 비용 제어에 유용합니다.
#
# 아래 코드는 `before_model` 후크에서 메시지 수를 확인하고, 제한을 초과하면 종료 메시지를 반환하는 `MessageLimitMiddleware` 클래스를 구현합니다.

# %%
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if len(state["messages"]) >= self.max_messages:
            return {
                "messages": [AIMessage("Conversation limit reached.")],
            }
        return None


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[MessageLimitMiddleware(max_messages=10)],
)

# %% [markdown]
# ### 랩 스타일 후크
#
# 랩 스타일 후크(`wrap_model_call`, `wrap_tool_call`)는 실행을 가로채고 핸들러가 호출되는 시기를 완전히 제어할 수 있습니다. 핸들러를 0번(단락/short-circuit), 1번(정상 흐름), 또는 여러 번(재시도 로직) 호출할지 결정할 수 있어 재시도, 캐싱, 조건부 실행 등 고급 패턴을 구현하는 데 적합합니다.
#
# 아래 코드는 `wrap_model_call` 후크를 사용하여 모델 호출 실패 시 최대 3회까지 재시도하는 `RetryMiddleware` 클래스를 구현합니다.

# %%
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable


class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[RetryMiddleware(max_retries=3)],
)

# %% [markdown]
# ### 동적 모델 선택 예제
#
# 랩 스타일 후크의 또 다른 활용 예제로, 대화 길이나 상황에 따라 다른 모델을 선택하는 미들웨어를 구현할 수 있습니다. 짧은 대화에는 가벼운 모델을, 긴 대화에는 더 큰 컨텍스트 창을 가진 모델을 사용하여 비용과 성능을 최적화할 수 있습니다.
#
# 아래 코드는 메시지 수에 따라 `gpt-4.1` 또는 `gpt-4.1-nano` 모델을 동적으로 선택하는 `DynamicModelMiddleware` 클래스를 구현합니다.

# %%
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable


class DynamicModelMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 대화 길이에 따라 다른 모델 사용
        if len(request.messages) > 10:
            request.model = init_chat_model("openai:gpt-4.1")
            print("Using gpt-4.1 for long conversation")
        else:
            request.model = init_chat_model("openai:gpt-4.1-nano")
            print("Using gpt-4.1-mini for short conversation")

        return handler(request)


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[DynamicModelMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
print(result["messages"][-1].content)

# %% [markdown]
# ## 실행 순서
#
# 여러 미들웨어를 사용할 때 실행 순서를 이해하는 것이 중요합니다. 미들웨어 리스트에 추가된 순서에 따라 실행 순서가 결정되며, 후크 타입에 따라 다른 규칙이 적용됩니다.
#
# **주요 규칙:**
# - `before_*` 후크: 첫 번째부터 마지막까지 순차 실행
# - `after_*` 후크: 마지막부터 첫 번째까지 역순 실행
# - `wrap_*` 후크: 중첩됨 (첫 번째 미들웨어가 가장 바깥쪽에서 다른 모든 것을 래핑)
#
# 아래 코드는 두 개의 미들웨어를 사용하여 실행 순서를 확인하는 예제입니다.

# %%
from langchain.agents.middleware import AgentMiddleware


class Middleware1(AgentMiddleware):
    def before_model(self, state, runtime):
        print("1: before_model")
        return None

    def after_model(self, state, runtime):
        print("1: after_model")
        return None


class Middleware2(AgentMiddleware):
    def before_model(self, state, runtime):
        print("2: before_model")
        return None

    def after_model(self, state, runtime):
        print("2: after_model")
        return None


# 실행 순서 확인
agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[Middleware1(), Middleware2()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

# 출력:
# 1: before_model
# 2: before_model
# (모델 호출)
# 2: after_model
# 1: after_model
