# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LangChain 단기 메모리
#
# 메모리는 이전 상호작용에 대한 정보를 기억하는 시스템입니다. AI 에이전트의 경우 메모리는 이전 상호작용을 기억하고, 피드백으로부터 학습하며, 사용자 선호도에 적응할 수 있게 해주므로 매우 중요합니다.
#
# 단기 메모리는 애플리케이션이 단일 스레드 또는 대화 내에서 이전 상호작용을 기억할 수 있게 해줍니다.
#
# **스레드**는 이메일이 단일 대화에서 메시지를 그룹화하는 방식과 유사하게 세션에서 여러 상호작용을 구성합니다.
#
# 대화 기록은 가장 일반적인 형태의 단기 메모리입니다. 긴 대화는 오늘날의 LLM에 도전 과제를 제시합니다. 전체 기록이 LLM의 컨텍스트 창에 맞지 않아 컨텍스트 손실이나 오류가 발생할 수 있습니다.

# %% [markdown]
# ## 사전 준비
#
# 환경 변수를 설정합니다.

# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %% [markdown]
# ## 기본 사용법
#
# 에이전트에 단기 메모리(스레드 수준 지속성)를 추가하려면 에이전트를 생성할 때 `checkpointer`를 지정해야 합니다.
#
# LangChain의 에이전트는 단기 메모리를 에이전트 상태의 일부로 관리합니다. 그래프의 상태에 저장함으로써 에이전트는 서로 다른 스레드 간의 분리를 유지하면서 특정 대화에 대한 전체 컨텍스트에 액세스할 수 있습니다.

# %%
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

# 간단한 도구 정의
@tool
def get_user_info(user_id: str) -> str:
    """Get user information."""
    return f"User info for {user_id}"

# 모델 및 에이전트 생성 (체크포인터 포함)
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model=model,
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  # 메모리 저장소
)

# thread_id를 사용하여 대화 추적
config = {"configurable": {"thread_id": "1"}}

# 첫 번째 메시지
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    config
)
print("Response 1:", result1["messages"][-1].content)

# 두 번째 메시지 (이전 대화 기억)
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config
)
print("Response 2:", result2["messages"][-1].content)

# %% [markdown]
# ### 프로덕션 환경
#
# 프로덕션 환경에서는 데이터베이스를 기반으로 하는 체크포인터를 사용합니다.

# %%
# PostgreSQL 체크포인터 예제 (설치 필요: pip install langgraph-checkpoint-postgres)
# from langgraph.checkpoint.postgres import PostgresSaver

# DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     checkpointer.setup()  # PostgreSQL에 테이블 자동 생성
#     agent = create_agent(
#         model=model,
#         tools=[get_user_info],
#         checkpointer=checkpointer,
#     )

# %% [markdown]
# ## 에이전트 메모리 커스터마이징
#
# 기본적으로 에이전트는 `AgentState`를 사용하여 단기 메모리를 관리합니다. 특히 `messages` 키를 통한 대화 기록을 관리합니다.
#
# `AgentState`를 확장하여 추가 필드를 추가할 수 있습니다.

# %%
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

# 커스텀 상태 정의
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    model=model,
    tools=[get_user_info],
    state_schema=CustomAgentState,  # 커스텀 상태 스키마
    checkpointer=InMemorySaver(),
)

# 커스텀 상태를 invoke에 전달
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)

print(result["messages"][-1].content)

# %% [markdown]
# ## 일반적인 패턴
#
# 단기 메모리가 활성화된 상태에서 긴 대화는 LLM의 컨텍스트 창을 초과할 수 있습니다. 일반적인 해결책은:
#
# 1. **메시지 트리밍** - 처음 또는 마지막 N개의 메시지 제거 (LLM 호출 전)
# 2. **메시지 삭제** - LangGraph 상태에서 메시지를 영구적으로 삭제
# 3. **메시지 요약** - 기록의 이전 메시지를 요약하고 요약으로 대체
# 4. **커스텀 전략** - 메시지 필터링 등의 커스텀 전략

# %% [markdown]
# ### 메시지 트리밍
#
# 대부분의 LLM에는 최대 지원 컨텍스트 창(토큰 단위)이 있습니다. 메시지를 트리밍하는 시기를 결정하는 한 가지 방법은 메시지 기록의 토큰을 세고 한계에 접근할 때마다 트리밍하는 것입니다.
#
# 에이전트에서 메시지 기록을 트리밍하려면 `@before_model` 미들웨어 데코레이터를 사용합니다.

# %%
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """컨텍스트 창에 맞도록 최근 몇 개의 메시지만 유지"""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 변경 필요 없음

    # 첫 번째 메시지와 최근 메시지만 유지
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model=model,
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": [{"role": "user", "content": "hi, my name is bob"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "write a short poem about cats"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "now do the same but for dogs"}]}, config)
final_response = agent.invoke({"messages": [{"role": "user", "content": "what's my name?"}]}, config)

print(final_response["messages"][-1].content)

# %% [markdown]
# ### 메시지 삭제
#
# 그래프 상태에서 메시지를 삭제하여 메시지 기록을 관리할 수 있습니다. 특정 메시지를 제거하거나 전체 메시지 기록을 지우려는 경우에 유용합니다.
#
# 그래프 상태에서 메시지를 삭제하려면 `RemoveMessage`를 사용할 수 있습니다.

# %%
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """대화를 관리 가능하게 유지하기 위해 오래된 메시지 제거"""
    messages = state["messages"]
    if len(messages) > 2:
        # 가장 오래된 두 개의 메시지 제거
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

# 첫 번째 메시지
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config
)
print("Messages after first invoke:", len(result1["messages"]))

# 두 번째 메시지
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config
)
print("Messages after second invoke:", len(result2["messages"]))
print("Last message:", result2["messages"][-1].content)

# %% [markdown]
# ### 메시지 요약
#
# 메시지를 트리밍하거나 제거하는 문제는 메시지 큐를 제거하여 정보를 잃을 수 있다는 것입니다. 이 때문에 일부 애플리케이션은 채팅 모델을 사용하여 메시지 기록을 요약하는 더 정교한 접근 방식의 이점을 얻습니다.
#
# 에이전트에서 메시지 기록을 요약하려면 내장된 `SummarizationMiddleware`를 사용합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",
            max_tokens_before_summary=4000,  # 4000 토큰에서 요약 트리거
            messages_to_keep=20,             # 요약 후 최근 20개 메시지 유지
        )
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": [{"role": "user", "content": "hi, my name is bob"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "write a short poem about cats"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "now do the same but for dogs"}]}, config)
final_response = agent.invoke({"messages": [{"role": "user", "content": "what's my name?"}]}, config)

print(final_response["messages"][-1].content)

# %% [markdown]
# ## 메모리 액세스
#
# 여러 가지 방법으로 에이전트의 단기 메모리(상태)에 액세스하고 수정할 수 있습니다.

# %% [markdown]
# ### 도구에서 단기 메모리 읽기
#
# `ToolRuntime` 매개변수를 사용하여 도구에서 단기 메모리(상태)에 액세스할 수 있습니다.
#
# `tool_runtime` 매개변수는 도구 시그니처에서 숨겨져 있지만(모델이 볼 수 없음) 도구는 이를 통해 상태에 액세스할 수 있습니다.

# %%
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime

class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model=model,
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "look up user information"}],
    "user_id": "user_123"
})

print(result["messages"][-1].content)

# %% [markdown]
# ### 도구에서 단기 메모리 쓰기
#
# 실행 중에 에이전트의 단기 메모리(상태)를 수정하려면 도구에서 직접 상태 업데이트를 반환할 수 있습니다.

# %%
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"

agent = create_agent(
    model=model,
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)

print(result["messages"][-1].content)

# %% [markdown]
# ### 프롬프트에서 메모리 액세스
#
# 대화 기록이나 커스텀 상태 필드를 기반으로 동적 프롬프트를 생성하기 위해 미들웨어에서 단기 메모리(상태)에 액세스할 수 있습니다.

# %%
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool

class CustomContext(TypedDict):
    user_name: str

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)

print(result["messages"][-1].content)

# %% [markdown]
# ## 종합 예제
#
# 다양한 메모리 관리 기법을 결합한 실용적인 예제입니다.

# %%
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from typing import Any

# 커스텀 상태
class ConversationState(AgentState):
    user_preferences: dict
    conversation_count: int

# 도구 정의
@tool
def save_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime
) -> str:
    """Save user preference."""
    return f"Saved preference: {preference_key} = {preference_value}"

# 메시지 트리밍 미들웨어
@before_model
def count_and_trim(state: ConversationState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    count = state.get("conversation_count", 0) + 1
    
    updates = {"conversation_count": count}
    
    if len(messages) > 10:
        print(f"Trimming messages (conversation #{count})")
        from langchain.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
        updates["messages"] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            messages[0],
            *messages[-5:]
        ]
    
    return updates

# 에이전트 생성
agent = create_agent(
    model=model,
    tools=[save_preference],
    state_schema=ConversationState,
    middleware=[
        count_and_trim,
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",
            max_tokens_before_summary=5000,
            messages_to_keep=10,
        )
    ],
    checkpointer=InMemorySaver(),
)

# 테스트
config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hi! I prefer dark mode."}],
        "user_preferences": {},
        "conversation_count": 0
    },
    config
)

print("\nFinal state:")
print(f"Conversation count: {result.get('conversation_count', 0)}")
print(f"Message count: {len(result['messages'])}")
print(f"Last message: {result['messages'][-1].content}")
