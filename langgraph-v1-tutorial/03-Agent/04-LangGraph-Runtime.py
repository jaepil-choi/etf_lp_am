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
# # Runtime
#
# LangChain의 `create_agent`는 내부적으로 LangGraph의 런타임을 사용합니다. Runtime은 에이전트 실행 중 도구와 미들웨어에서 접근할 수 있는 컨텍스트 정보를 제공합니다.
#
# **Runtime 구성 요소:**
#
# | 구성 요소 | 설명 |
# |:---|:---|
# | **Context** | 사용자 ID, 데이터베이스 연결 등 정적 정보 |
# | **Store** | 장기 메모리를 위한 `BaseStore` 인스턴스 |
# | **Stream Writer** | `"custom"` 스트림 모드로 정보 스트리밍 |
#
# 런타임 정보는 도구와 미들웨어 내에서 `runtime` 매개변수를 통해 액세스할 수 있습니다.
#
# > 참고 문서: [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence.md)

# %% [markdown]
# ## 환경 설정
#
# Runtime 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드합니다.
#
# 아래 코드는 환경 변수를 로드합니다.

# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %% [markdown]
# ---
#
# ## Context 정의 및 사용
#
# `create_agent`로 에이전트를 생성할 때 `context_schema`를 지정하여 에이전트 `Runtime`에 저장될 `context`의 구조를 정의할 수 있습니다. Context는 dataclass 또는 Pydantic 모델로 정의하며, 에이전트 호출 시 `context` 매개변수로 전달합니다.
#
# Context는 도구와 미들웨어에서 `runtime.context`를 통해 접근할 수 있으며, 사용자별 설정이나 세션 정보를 전달하는 데 유용합니다.
#
# 아래 코드는 Context 스키마를 정의하고 에이전트에 전달하는 예시입니다.

# %%
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool


@dataclass
class Context:
    user_name: str


@tool
def greet_user() -> str:
    """Greet the user."""
    return "Hello!"


model = ChatOpenAI(model="gpt-4.1-mini")

agent = create_agent(
    model=model, tools=[greet_user], context_schema=Context  # Context 스키마 정의
)

# Context를 전달하여 에이전트 호출
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith"),  # Context 전달
)

print(result["messages"][-1].content)

# %% [markdown]
# ---
#
# ## 도구에서 Runtime 액세스
#
# 도구 내에서 `ToolRuntime` 매개변수를 사용하여 `Runtime` 객체에 액세스할 수 있습니다. 이를 통해 다음 기능을 수행할 수 있습니다:
#
# - **Context 접근**: 사용자 정보, 세션 데이터 등
# - **Store 읽기/쓰기**: 장기 메모리 관리
# - **Stream Writer**: 진행 상황 스트리밍
#
# `ToolRuntime` 매개변수는 도구 시그니처에 추가하면 자동으로 주입되며, LLM에는 노출되지 않습니다.

# %% [markdown]
# ### Context 액세스
#
# 도구에서 `runtime.context`를 통해 Context 객체에 접근할 수 있습니다. `ToolRuntime[ContextType]` 형태로 타입 힌트를 지정하면 IDE에서 자동 완성을 지원받을 수 있습니다.
#
# 아래 코드는 도구에서 Context에 접근하여 사용자 정보를 활용하는 예시입니다.

# %%
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langchain_teddynote.messages import invoke_graph


@dataclass
class UserContext:
    user_id: str
    user_name: str
    user_email: str


@tool
def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get information about the current user."""
    # Context에서 사용자 정보 가져오기
    user_id = runtime.context.user_id
    user_name = runtime.context.user_name
    user_email = runtime.context.user_email

    return f"User ID: {user_id}, Name: {user_name}, Email: {user_email}"


@tool
def personalized_greeting(runtime: ToolRuntime[UserContext]) -> str:
    """Generate a personalized greeting for the user."""
    user_name = runtime.context.user_name
    return f"안녕하세요, {user_name}님! 무엇을 도와드릴까요?"


agent = create_agent(
    model=model,
    tools=[get_user_info, personalized_greeting],
    context_schema=UserContext,
)

# %%
# Context를 전달하여 호출
invoke_graph(
    agent,
    inputs={"messages": [{"role": "user", "content": "안녕하세요? 반갑습니다."}]},
    context=UserContext(
        user_id="user_123", user_name="김철수", user_email="chulsoo@example.com"
    ),
)

# %% [markdown]
# ### Store 액세스 (장기 메모리)
#
# 도구 내에서 `runtime.store`를 사용하여 장기 메모리에 액세스할 수 있습니다. Store는 대화 세션을 넘어서 데이터를 영구 저장하며, `get()`, `put()` 메서드로 데이터를 읽고 씁니다.
#
# Store의 키는 네임스페이스(튜플)와 키(문자열)로 구성되어 데이터를 체계적으로 관리할 수 있습니다.
#
# 아래 코드는 Store를 사용하여 사용자 설정을 저장하고 조회하는 예시입니다.

# %%
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id

    # 기본 설정
    preferences: str = "The user prefers you to write a brief and polite email."

    # Store에서 사용자 설정 가져오기
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences


@tool
def save_user_preference(preference: str, runtime: ToolRuntime[Context]) -> str:
    """Save user preference to the store."""
    user_id = runtime.context.user_id

    if runtime.store:
        runtime.store.put(("users",), user_id, {"preferences": preference})
        return f"Saved preference: {preference}"

    return "Store not available"


# Store와 함께 에이전트 생성
store = InMemoryStore()

# 초기 데이터 설정
store.put(
    ("users",),
    "user_123",
    {"preferences": "The user prefers detailed and technical explanations."},
)

agent = create_agent(
    model=model,
    tools=[fetch_user_email_preferences, save_user_preference],
    context_schema=Context,
    store=store,  # Store 전달
)

# Store에서 설정 가져오기
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my email preferences?"}]},
    context=Context(user_id="user_123"),
)

print(result["messages"][-1].content)

# %% [markdown]
# ### Stream Writer 액세스
#
# 도구 내에서 `runtime.get_stream_writer()` 또는 `runtime.stream_writer`를 사용하여 커스텀 업데이트를 스트리밍할 수 있습니다. 이는 장시간 실행되는 작업에서 사용자에게 진행 상황을 실시간으로 알려줄 때 유용합니다.
#
# 스트리밍된 업데이트는 `stream_mode="custom"`으로 수신할 수 있습니다.
#
# 아래 코드는 Stream Writer를 사용하여 진행 상황을 스트리밍하는 예시입니다.

# %%
from langchain.tools import tool, ToolRuntime
import time


@tool
def process_large_dataset(num_items: int, runtime: ToolRuntime) -> str:
    """Process a large dataset and report progress."""
    writer = runtime.get_stream_writer()

    # 진행 상황 스트리밍
    for i in range(0, num_items, 10):
        progress = min(i + 10, num_items)
        writer({"stage": "processing", "progress": progress, "total": num_items})
        time.sleep(0.1)  # 작업 시뮬레이션

    writer({"stage": "completed", "total": num_items})
    return f"Successfully processed {num_items} items!"


agent = create_agent(
    model=model,
    tools=[process_large_dataset],
)

# 커스텀 스트림 모드로 진행 상황 추적
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Process 50 items"}]},
    stream_mode="custom",
):
    if "progress" in chunk:
        percentage = (chunk["progress"] / chunk["total"]) * 100
        print(f"Progress: {percentage:.0f}%")
    elif "stage" in chunk and chunk["stage"] == "completed":
        print(f"Completed processing {chunk['total']} items!")

# %% [markdown]
# ---
#
# ## 미들웨어에서 Runtime 액세스
#
# 미들웨어에서 `Runtime` 객체에 액세스하여 동적 프롬프트를 생성하거나, 메시지를 수정하거나, 사용자 컨텍스트에 따라 에이전트 동작을 제어할 수 있습니다. 미들웨어 함수의 `runtime` 매개변수를 통해 Context, Store 등에 접근합니다.

# %% [markdown]
# ### Dynamic Prompt에서 Runtime 사용
#
# `@dynamic_prompt` 데코레이터로 정의된 미들웨어에서 `request.runtime`을 통해 Context에 접근할 수 있습니다. 이를 활용하여 사용자별로 다른 시스템 프롬프트를 동적으로 생성할 수 있습니다.
#
# 아래 코드는 사용자 역할에 따라 다른 시스템 프롬프트를 생성하는 예시입니다.

# %%
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool


@dataclass
class Context:
    user_name: str
    user_role: str


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    # Runtime에서 Context 가져오기
    user_name = request.runtime.context.user_name
    user_role = request.runtime.context.user_role

    # 사용자 역할에 따라 다른 프롬프트
    if user_role == "admin":
        system_prompt = f"You are a helpful assistant with full access. Address the user as {user_name}."
    else:
        system_prompt = f"You are a helpful assistant. Address the user as {user_name}. Provide brief answers."

    return system_prompt


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=Context,
)

# Admin 사용자로 호출
print("=== Admin User ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=Context(user_name="Admin Kim", user_role="admin"),
)
print(result["messages"][-1].content)

# 일반 사용자로 호출
print("\n=== Regular User ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=Context(user_name="User Lee", user_role="user"),
)
print(result["messages"][-1].content)

# %% [markdown]
# ### Before/After Model에서 Runtime 사용
#
# `@before_model`과 `@after_model` 데코레이터로 정의된 미들웨어에서도 `runtime` 매개변수를 통해 Context에 접근할 수 있습니다. 이를 활용하여 모델 호출 전후에 로깅, 검증, 변환 등의 작업을 수행할 수 있습니다.
#
# 아래 코드는 모델 호출 전후에 사용자 정보를 로깅하는 예시입니다.

# %%
from langchain.agents import AgentState
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime
from dataclasses import dataclass


@dataclass
class Context:
    user_name: str
    session_id: str


@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    """모델 호출 전 로깅"""
    print(
        f"[Before Model] User: {runtime.context.user_name}, Session: {runtime.context.session_id}"
    )
    print(f"[Before Model] Messages count: {len(state['messages'])}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    """모델 호출 후 로깅"""
    print(f"[After Model] User: {runtime.context.user_name}")
    print(f"[After Model] Response generated for session: {runtime.context.session_id}")
    return None


agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[log_before_model, log_after_model],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Seoul?"}]},
    context=Context(user_name="John Smith", session_id="session_456"),
)

print(f"\nFinal response: {result['messages'][-1].content}")

# %% [markdown]
# ---
#
# ## 종합 예제: 사용자 컨텍스트 기반 에이전트
#
# Runtime의 모든 기능을 활용한 실용적인 예제입니다. Context로 사용자 정보를 전달하고, Store로 검색 기록을 관리하며, 미들웨어로 동적 프롬프트와 사용량 추적을 구현합니다.
#
# 아래 코드는 사용자 등급에 따라 다른 기능을 제공하는 에이전트 예시입니다.

# %%
from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


@dataclass
class UserContext:
    user_id: str
    user_name: str
    user_tier: str  # "free", "premium", "enterprise"
    language: str  # "ko", "en"


# 도구 정의
@tool
def search_database(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search the database. Access level depends on user tier."""
    user_tier = runtime.context.user_tier

    # 사용자 등급에 따라 다른 결과 제공
    if user_tier == "enterprise":
        return f"Full database search results for: {query} (Enterprise access)"
    elif user_tier == "premium":
        return f"Premium search results for: {query}"
    else:
        return f"Basic search results for: {query} (Limited to 10 results)"


@tool
def get_user_history(runtime: ToolRuntime[UserContext]) -> str:
    """Get user's search history from store."""
    user_id = runtime.context.user_id

    if runtime.store:
        if history := runtime.store.get(("history",), user_id):
            return f"Recent searches: {history.value['searches']}"

    return "No search history found"


@tool
def save_search(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save search query to user history."""
    user_id = runtime.context.user_id

    if runtime.store:
        # 기존 히스토리 가져오기
        existing = runtime.store.get(("history",), user_id)
        searches = existing.value["searches"] if existing else []

        # 새 검색어 추가
        searches.append(query)
        runtime.store.put(
            ("history",), user_id, {"searches": searches[-5:]}
        )  # 최근 5개만 유지

        return f"Saved search: {query}"

    return "Store not available"


# 동적 프롬프트 - 사용자 언어에 따라 변경
@dynamic_prompt
def multilingual_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    language = request.runtime.context.language
    user_tier = request.runtime.context.user_tier

    if language == "ko":
        prompt = f"당신은 도움이 되는 어시스턴트입니다. 사용자를 '{user_name}'님으로 호칭하세요."
        if user_tier == "enterprise":
            prompt += (
                " 이 사용자는 엔터프라이즈 회원이므로 모든 기능에 액세스할 수 있습니다."
            )
    else:
        prompt = f"You are a helpful assistant. Address the user as {user_name}."
        if user_tier == "enterprise":
            prompt += " This is an enterprise user with full access."

    return prompt


# 사용량 추적 미들웨어
@before_model
def track_usage(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """Track API usage for billing"""
    user_id = runtime.context.user_id
    user_tier = runtime.context.user_tier

    print(f"[Usage Tracker] User: {user_id}, Tier: {user_tier}")

    # 무료 사용자의 경우 사용량 제한 확인
    if user_tier == "free":
        if runtime.store:
            usage = runtime.store.get(("usage",), user_id)
            count = usage.value["count"] if usage else 0

            if count >= 10:
                print("[Usage Tracker] Free tier limit reached!")
                # 실제로는 여기서 실행을 중단할 수 있음

            # 사용량 업데이트
            runtime.store.put(("usage",), user_id, {"count": count + 1})

    return None


# Store 생성 및 초기 데이터 설정
store = InMemoryStore()
store.put(
    ("history",), "user_001", {"searches": ["Python tutorial", "LangChain guide"]}
)
store.put(("usage",), "user_002", {"count": 5})

# 에이전트 생성
agent = create_agent(
    model=model,
    tools=[search_database, get_user_history, save_search],
    middleware=[multilingual_prompt, track_usage],
    context_schema=UserContext,
    store=store,
)

# 테스트 1: 엔터프라이즈 사용자 (한국어)
print("=== Test 1: Enterprise User (Korean) ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for 'machine learning'"}]},
    context=UserContext(
        user_id="user_001", user_name="김철수", user_tier="enterprise", language="ko"
    ),
)
print(result["messages"][-1].content)

# 테스트 2: 무료 사용자 (영어)
print("\n=== Test 2: Free User (English) ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Search for 'data science'"}]},
    context=UserContext(
        user_id="user_002", user_name="John Doe", user_tier="free", language="en"
    ),
)
print(result["messages"][-1].content)

# 테스트 3: 검색 기록 조회
print("\n=== Test 3: Check Search History ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my search history?"}]},
    context=UserContext(
        user_id="user_001", user_name="김철수", user_tier="enterprise", language="ko"
    ),
)
print(result["messages"][-1].content)

# %% [markdown]
# ---
#
# ## 실전 패턴
#
# ### 데이터베이스 연결 전달
#
# Context를 사용하여 데이터베이스 연결 객체를 도구에 전달할 수 있습니다. 이 패턴을 사용하면 도구에서 직접 데이터베이스에 접근하여 쿼리를 실행할 수 있습니다.
#
# 아래 코드는 데이터베이스 연결을 Context로 전달하는 예시입니다.

# %%
from dataclasses import dataclass
from typing import Any


@dataclass
class DatabaseContext:
    db_connection: Any  # 실제로는 데이터베이스 연결 객체
    user_id: str


@tool
def query_database(sql: str, runtime: ToolRuntime[DatabaseContext]) -> str:
    """Execute SQL query on the database."""
    db = runtime.context.db_connection
    user_id = runtime.context.user_id

    # 실제 데이터베이스 쿼리 실행
    # result = db.execute(sql)

    return f"Query executed for user {user_id}: {sql}"


# 사용 예시 (실제 DB 연결 대신 None 사용)
# agent = create_agent(
#     model=model,
#     tools=[query_database],
#     context_schema=DatabaseContext
# )
#
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "Query user data"}]},
#     context=DatabaseContext(db_connection=db, user_id="user_123")
# )

# %% [markdown]
# ### 인증 및 권한 검사
#
# 미들웨어에서 Context를 사용하여 사용자 인증 및 권한을 검사할 수 있습니다. `@before_agent` 미들웨어에서 권한이 없는 요청을 사전에 차단하면 보안을 강화할 수 있습니다.
#
# 아래 코드는 사용자 권한을 검사하는 미들웨어 예시입니다.

# %%
from langchain.agents.middleware import before_agent, hook_config
from typing import Any


@dataclass
class AuthContext:
    user_id: str
    permissions: list[str]


@before_agent(can_jump_to=["end"])
def check_permissions(
    state: AgentState, runtime: Runtime[AuthContext]
) -> dict[str, Any] | None:
    """Check if user has required permissions"""
    permissions = runtime.context.permissions

    # 메시지에서 요청된 작업 확인
    if state["messages"]:
        content = state["messages"][0].content.lower()

        # 관리자 작업 요청 시 권한 확인
        if "delete" in content or "remove" in content:
            if "admin" not in permissions:
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "You don't have permission to perform this action.",
                        }
                    ],
                    "jump_to": "end",
                }

    return None


# 사용 예시
# agent = create_agent(
#     model=model,
#     tools=[search_tool],
#     middleware=[check_permissions],
#     context_schema=AuthContext
# )
#
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "Delete user data"}]},
#     context=AuthContext(user_id="user_123", permissions=["read", "write"])
# )

# %% [markdown]
# ### 요청별 설정
#
# 각 요청에 대한 특정 설정을 Context를 통해 전달할 수 있습니다. 타임아웃, 토큰 제한, 로깅 수준 등 요청마다 다른 설정이 필요한 경우 유용합니다.
#
# 아래 코드는 요청별 설정을 Context로 전달하는 예시입니다.

# %%
@dataclass
class RequestContext:
    user_id: str
    verbose: bool
    timeout: int
    max_tokens: int


@tool
def process_request(query: str, runtime: ToolRuntime[RequestContext]) -> str:
    """Process request with custom settings."""
    verbose = runtime.context.verbose
    timeout = runtime.context.timeout

    if verbose:
        print(f"Processing with timeout: {timeout}s")

    # 설정에 따라 처리
    return f"Processed: {query}"


agent = create_agent(
    model=model, tools=[process_request], context_schema=RequestContext
)

# 요청별로 다른 설정 사용
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Process my request"}]},
    context=RequestContext(
        user_id="user_123", verbose=True, timeout=30, max_tokens=1000
    ),
)

print(result["messages"][-1].content)

# %% [markdown]
# ---
#
# ## 정리
#
# 이 튜토리얼에서는 LangGraph 에이전트의 Runtime 기능을 학습했습니다.
#
# **핵심 개념 요약:**
#
# | 개념 | 설명 | 접근 방법 |
# |:---|:---|:---|
# | **Context** | 사용자 정보, 세션 데이터 등 정적 정보 | `runtime.context` |
# | **Store** | 대화 세션을 넘어선 장기 메모리 | `runtime.store.get()`, `put()` |
# | **Stream Writer** | 커스텀 진행 상황 스트리밍 | `runtime.get_stream_writer()` |
#
# **실전 패턴:**
# - 데이터베이스 연결을 Context로 전달하여 도구에서 직접 쿼리 실행
# - 미들웨어에서 사용자 권한 검사로 보안 강화
# - 요청별 설정(타임아웃, 토큰 제한 등)을 Context로 전달
#
# **다음 단계:**
# - 구조화된 출력(Structured Output)을 사용한 에이전트 구축 학습
