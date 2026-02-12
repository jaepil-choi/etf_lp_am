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
# # 도구 (Tools)
#
# 도구는 에이전트가 외부 시스템과 상호작용할 수 있게 해주는 핵심 구성 요소입니다. API 호출, 데이터베이스 쿼리, 파일 시스템 접근 등 다양한 작업을 수행할 수 있으며, 잘 정의된 입력과 출력을 통해 모델의 기능을 확장합니다.
#
# 도구는 호출 가능한 함수와 입력 스키마를 캡슐화합니다. 호환 가능한 채팅 모델에 전달되면, 모델은 도구를 호출할지 여부와 어떤 인수로 호출할지를 자율적으로 결정합니다. 이를 통해 에이전트는 복잡한 작업을 단계별로 수행할 수 있습니다.
#
# > 참고 문서: [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools.md)

# %% [markdown]
# ## 환경 설정
#
# 도구 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드하고, `langchain_teddynote`의 로깅 기능을 활성화하여 LangSmith에서 도구 호출 과정을 추적할 수 있도록 합니다.
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
# ## 도구 생성
#
# ### 기본 도구 정의
#
# 도구를 생성하는 가장 간단한 방법은 `@tool` 데코레이터를 사용하는 것입니다. 데코레이터가 적용된 함수는 자동으로 도구로 변환되며, 함수의 docstring이 도구 설명이 됩니다.
#
# **도구 정의 규칙:**
# 1. `@tool` 데코레이터를 함수에 적용합니다
# 2. 도구 이름은 함수 이름에서 자동으로 가져옵니다
# 3. 도구 설명은 함수의 docstring에서 가져옵니다
# 4. 입력 스키마는 함수의 매개변수와 타입 힌트에서 자동으로 생성됩니다
#
# 아래 코드는 데이터베이스 검색 도구를 정의하는 예시입니다.

# %%
from langchain.tools import tool


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"


# 도구 정보 확인
print(f"Tool name: {search_database.name}")
print(f"Tool description: {search_database.description}")


# %% [markdown]
# ### 타입 힌트의 중요성
#
# 타입 힌트는 도구의 입력 스키마를 정의하므로 필수입니다. 모델은 타입 힌트를 통해 각 매개변수의 타입을 이해하고 올바른 값을 전달합니다. docstring은 모델이 도구의 목적을 이해하는 데 도움이 되므로 정보가 풍부하고 간결해야 합니다.

# %% [markdown]
# ### 커스텀 도구 이름
#
# 기본적으로 도구 이름은 함수 이름에서 가져옵니다. 더 설명적인 이름이 필요한 경우 `@tool("커스텀_이름")` 형식으로 재정의할 수 있습니다. 도구 이름은 모델이 어떤 도구를 호출할지 결정하는 데 사용되므로 명확하고 직관적이어야 합니다.
#
# 아래 코드는 커스텀 도구 이름을 설정하는 예시입니다.

# %%
@tool("web_search")  # 커스텀 이름
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


# 도구 이름 확인
print(f"Tool name: {search.name}")  # web_search


# %% [markdown]
# ### 커스텀 도구 설명
#
# 더 명확한 모델 가이드를 위해 자동 생성된 도구 설명을 재정의할 수 있습니다. `description` 매개변수를 사용하면 docstring과 별도로 모델에게 전달되는 설명을 지정할 수 있습니다.
#
# 좋은 도구 설명은 도구의 목적, 사용 시점, 예상 결과를 명확히 전달해야 합니다.
#
# 아래 코드는 커스텀 도구 설명을 설정하는 예시입니다.

# %%
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems.",
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))


# 도구 설명 확인
print(f"Tool description: {calc.description}")

# %% [markdown]
# ---
#
# ## Pydantic 모델로 입력 스키마 정의
#
# 복잡한 입력 검증이 필요한 경우 Pydantic 모델을 사용하여 명확한 입력 스키마를 정의할 수 있습니다. `args_schema` 매개변수에 Pydantic 모델을 전달하면, 도구 호출 시 자동으로 입력 검증이 수행됩니다.
#
# Pydantic 모델을 사용하면 다음과 같은 이점이 있습니다:
# - 타입 검증 및 변환 자동화
# - `Literal` 타입을 통한 허용값 제한
# - `Field`의 description을 통한 상세한 매개변수 설명
#
# 아래 코드는 Pydantic 모델을 사용한 날씨 조회 도구 예시입니다.

# %%
from pydantic import BaseModel, Field
from typing import Literal


class WeatherInput(BaseModel):
    """Input for weather queries."""

    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="Temperature unit preference"
    )
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")


@tool(args_schema=WeatherInput)
def get_weather(
    location: str, units: str = "celsius", include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"현재 {location} 지역의 날씨는 {temp} {units[0].upper()} 도"
    if include_forecast:
        result += "\n다음 5일 날씨: 맑음"
    return result


# %%
# 도구 테스트
print(
    get_weather.invoke(
        {"location": "Seoul", "units": "celsius", "include_forecast": True}
    )
)

# %% [markdown]
# ### 입력 검증 오류 예시
#
# 아래는 입력 스키마에서 `Literal["celsius", "fahrenheit"]` 타입을 사용했지만, 실제 입력값으로 유효하지 않은 `celsiuss`를 입력했을 때의 오류 예시입니다. Pydantic이 자동으로 입력을 검증하고 오류를 발생시킵니다.

# %%
# 도구 테스트
print(
    get_weather.invoke(
        {"location": "Seoul", "units": "celsiuss", "include_forecast": True}
    )
)

# %% [markdown]
# ---
#
# ## 컨텍스트 접근
#
# 도구는 에이전트 상태, 런타임 컨텍스트 및 장기 메모리에 액세스할 수 있을 때 가장 강력합니다. 이를 통해 도구는 컨텍스트 인식 결정을 내리고, 응답을 개인화하며, 대화 전반에 걸쳐 정보를 유지할 수 있습니다.
#
# `ToolRuntime` 매개변수를 통해 다음 런타임 정보에 액세스할 수 있습니다:
#
# | 속성 | 설명 |
# |:---|:---|
# | **state** | 실행을 통해 흐르는 변경 가능한 데이터 (메시지, 카운터, 커스텀 필드) |
# | **context** | 사용자 ID, 세션 세부 정보 등 불변 구성 정보 |
# | **store** | 대화 전반에 걸친 영구 장기 메모리 |
# | **stream_writer** | 도구 실행 중 커스텀 업데이트 스트리밍 |
# | **config** | 실행을 위한 RunnableConfig |
# | **tool_call_id** | 현재 도구 호출의 고유 ID |

# %% [markdown]
# ### ToolRuntime 사용
#
# `ToolRuntime`을 사용하면 단일 매개변수로 모든 런타임 정보에 액세스할 수 있습니다. 도구 시그니처에 `runtime: ToolRuntime`을 추가하면 LLM에는 노출되지 않고 자동으로 주입됩니다.
#
# `runtime.state`를 통해 현재 그래프 상태에 접근하고, `runtime.context`를 통해 컨텍스트 정보에 접근할 수 있습니다.
#
# 아래 코드는 ToolRuntime을 사용하여 상태와 컨텍스트에 접근하는 도구 예시입니다.

# %%
from typing import Literal, Optional, Dict, Any, List, Annotated, TypedDict
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_teddynote.messages import stream_graph, invoke_graph
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


# 현재 대화 상태 접근
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far."""
    # state 에서 메시지 접근
    messages = runtime.state.get("messages", [])
    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")
    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"


@tool
def get_user_preference(
    preference_name: Literal["food", "coding", "sports"],
    runtime: ToolRuntime,  # ToolRuntime 매개변수는 모델에 보이지 않습니다 (자동 주입)
) -> str:
    """Get a user preference value."""

    # context는 state에 저장하지 않고 별도의 context 객체로 inject됨
    preferences = {}
    if getattr(runtime, "context", None) is not None:
        # context dict 내 user_preferences
        preferences = runtime.context.user_preferences or {}
    return preferences.get(preference_name, "Have no information")


class CustomContext(BaseModel):
    user_preferences: Optional[Dict[str, Any]] = None


class CustomState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_preferences: Optional[Dict[str, Any]] = None


# 에이전트 생성
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model,
    tools=[summarize_conversation, get_user_preference],
    system_prompt="You are a helpful assistant.",
    # checkpointer=InMemorySaver(),
    context_schema=CustomContext,
    # state_schema=CustomState,
)

config = {"configurable": {"thread_id": "3"}}
inputs = {"messages": [{"role": "user", "content": "내가 좋아하는 음식 알려줘"}]}

invoke_graph(
    agent,
    inputs=inputs,
    config=config,
    context=CustomContext(user_preferences={"food": "pizza"}),
)

# %% [markdown]
# ### 상태 업데이트
#
# `Command`를 사용하면 도구 내에서 에이전트의 상태를 직접 업데이트하거나 그래프의 실행 흐름을 제어할 수 있습니다. `update` 필드로 상태를 업데이트하고, `goto` 필드로 다음 노드를 지정할 수 있습니다.
#
# 상태 업데이트 시 `ToolMessage`를 포함해야 하며, `tool_call_id`는 `runtime.tool_call_id`에서 가져와야 합니다.
#
# 아래 코드는 Command를 사용하여 사용자 이름을 업데이트하는 도구 예시입니다.

# %%
from dataclasses import dataclass
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import AnyMessage, RemoveMessage, ToolMessage
from langchain_teddynote.messages import invoke_graph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CustomContext(BaseModel):
    user_preferences: Optional[Dict[str, Any]] = None


class CustomState(BaseModel):
    user_name: str = Field(default="", description="The user's name")
    messages: Annotated[List[AnyMessage], add_messages]


# User Name 업데이트 도구
@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """Update the user's name."""
    return Command(
        update={
            "user_name": new_name,  # user_name 상태에 업데이트
            "messages": [
                ToolMessage(
                    content=f"Successfully updated user name to {new_name}",
                    tool_call_id=runtime.tool_call_id,  # runtime 에서 얻어온 tool_call_id 정보를 활용하여 업데이트
                )
            ],
        }
    )


@tool
def clear_messages(runtime: ToolRuntime) -> Command:
    """Clear all messages from the conversation history except the one whose tool_call_id matches the id we don't want to delete."""
    from langchain.messages import AIMessage

    messages = runtime.state.get("messages", [])

    to_remove_messages = []
    tool_call_id = runtime.tool_call_id

    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            # Tool Call ID 가 일치하지 않으면 삭제. Tool Call ID 가 일치하면 유지.
            if not any(call.get("id") == tool_call_id for call in m.tool_calls):
                to_remove_messages.append(m)
        else:
            to_remove_messages.append(m)

    removals = [RemoveMessage(id=m.id) for m in to_remove_messages]
    return Command(
        update={
            "messages": removals
            + [
                ToolMessage(
                    content=f"Successfully cleared all previous messages. Total of {len(removals)} deleted messages.",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


# 에이전트 생성
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model,
    tools=[update_user_name, clear_messages],
    system_prompt="You are a helpful assistant.",
    state_schema=CustomState,
    context_schema=CustomContext,  # 클래스 자체를 전달 (인스턴스가 아님)
    checkpointer=InMemorySaver(),
)

# %%
# 에이전트 실행
config = {"configurable": {"thread_id": "1"}}
invoke_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "내 이름은 테디야"}]},
    config=config,
)

# %%
invoke_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "내 이름은 사실 셜리야"}]},
    config=config,
)

# %%
messages = agent.get_state(config).values["messages"]
messages

# %%
invoke_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "메시지 전부 삭제해줘"}]},
    config=config,
)

# %%
invoke_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "내 이름이 사실 뭐라고 했지?"}]},
    config=config,
)

# %% [markdown]
# ---
#
# ### 컨텍스트
#
# `runtime.context`를 통해 사용자 ID, 세션 정보, 애플리케이션별 구성 등 불변 컨텍스트 데이터에 액세스할 수 있습니다. 컨텍스트는 에이전트 호출 시 전달되며, 도구 실행 중에 변경되지 않습니다.
#
# 아래 코드는 사용자 컨텍스트를 활용하여 계정 정보를 조회하는 도구 예시입니다.

# %%
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# 사용자 데이터베이스 시뮬레이션
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com",
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com",
    },
}


@dataclass
class UserContext:
    user_id: str


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"


# 에이전트 생성
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

# 컨텍스트와 함께 에이전트 실행
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123"),
)

print(result["messages"][-1].content)

# %% [markdown]
# ### 메모리 (Store)
#
# `runtime.store`를 사용하면 대화 전반에 걸쳐 영구 데이터에 액세스할 수 있습니다. Store는 사용자별 또는 애플리케이션별 데이터를 저장하고 검색하는 장기 메모리 역할을 합니다.
#
# Store는 `get()`, `put()` 메서드를 통해 데이터를 읽고 쓸 수 있으며, 네임스페이스(튜플)와 키를 사용하여 데이터를 구조화합니다.
#
# 아래 코드는 Store를 사용하여 사용자 정보를 저장하고 조회하는 도구 예시입니다.

# %%
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# 메모리 접근
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


# 메모리 업데이트
@tool
def save_user_info(
    user_id: str, user_info: dict[str, Any], runtime: ToolRuntime
) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."


# 스토어와 에이전트 생성
store = InMemoryStore()
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(model, tools=[get_user_info, save_user_info], store=store)

# 첫 번째 세션: 사용자 정보 저장
print("=== Saving user info ===")
result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev",
            }
        ]
    }
)
print(result1["messages"][-1].content)

# 두 번째 세션: 사용자 정보 가져오기
print("\n=== Getting user info ===")
result2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Get user info for user with id 'abc123'"}
        ]
    }
)
print(result2["messages"][-1].content)

# %% [markdown]
# ### Stream Writer
#
# `runtime.stream_writer`를 사용하면 도구 실행 중 커스텀 업데이트를 스트리밍할 수 있습니다. 이는 장시간 실행되는 도구에서 사용자에게 진행 상황을 실시간으로 알려줄 때 유용합니다.
#
# 스트리밍된 업데이트는 `stream_mode="custom"`으로 수신할 수 있습니다. Stream Writer는 LangGraph 실행 컨텍스트 내에서만 사용할 수 있습니다.
#
# 아래 코드는 Stream Writer를 사용하여 진행 상황을 스트리밍하는 도구 예시입니다.

# %%
from langchain.tools import tool, ToolRuntime


@tool
def get_weather_with_updates(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # 도구가 실행될 때 커스텀 업데이트 스트리밍
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"


# 참고: runtime.stream_writer를 도구 내에서 사용하는 경우,
# 도구는 LangGraph 실행 컨텍스트 내에서 호출되어야 합니다.

# %%
from langchain_teddynote.messages import stream_graph

# 스토어와 에이전트 생성
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(model, tools=[get_weather_with_updates])

inputs = {"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]}

for chunk in agent.stream(inputs, stream_mode="custom"):
    print(chunk)

# stream_graph(agent, inputs=inputs)

# %% [markdown]
# ---
#
# ## 정리
#
# 이 튜토리얼에서는 LangGraph 에이전트에서 도구를 정의하고 활용하는 방법을 학습했습니다.
#
# **핵심 개념 요약:**
#
# | 개념 | 설명 |
# |:---|:---|
# | **@tool 데코레이터** | 함수를 도구로 변환하며, docstring이 도구 설명이 됩니다 |
# | **Pydantic 스키마** | `args_schema`로 복잡한 입력 검증을 자동화합니다 |
# | **ToolRuntime** | state, context, store 등 런타임 정보에 접근합니다 |
# | **Command** | 도구 내에서 상태 업데이트 및 실행 흐름을 제어합니다 |
# | **Store** | 대화 전반에 걸친 영구적인 장기 메모리를 제공합니다 |
# | **Stream Writer** | 도구 실행 중 진행 상황을 실시간으로 스트리밍합니다 |
#
# **다음 단계:**
# - 스트리밍 모드를 활용한 실시간 응답 처리 학습
# - 런타임 컨텍스트를 활용한 고급 에이전트 구축
