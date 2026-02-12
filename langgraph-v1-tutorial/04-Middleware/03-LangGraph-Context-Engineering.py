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
# # 컨텍스트 엔지니어링(Context Engineering)
#
# 에이전트를 구축하는 데 있어 가장 어려운 부분은 충분히 신뢰할 수 있게 만드는 것입니다. MVP 단계에서 잘 동작하는 에이전트도 실제 환경에서는 종종 실패하기도 합니다. 이 튜토리얼에서는 에이전트의 신뢰성을 높이는 핵심 기술인 컨텍스트 엔지니어링에 대해 학습합니다.
#
# ## 에이전트가 실패하는 이유
#
# 에이전트가 실패할 때는 일반적으로 에이전트 내부의 LLM 호출이 잘못된 작업을 수행하거나 예상대로 작동하지 않았기 때문입니다. LLM은 다음 두 가지 이유 중 하나로 실패합니다:
#
# 1. 기본 LLM이 충분히 능력이 없음
# 2. **"올바른" 컨텍스트**가 LLM에 전달되지 않음
#
# 대부분의 경우 실제로는 두 번째 이유가 에이전트의 신뢰성을 떨어뜨립니다. **컨텍스트 엔지니어링**은 LLM이 작업을 완수할 수 있도록 올바른 형식으로 올바른 정보와 도구를 제공하는 것입니다. 이것이 AI 엔지니어의 가장 중요한 업무입니다.

# %% [markdown]
# ## 사전 준비
#
# 컨텍스트 엔지니어링 기법을 실습하기 위해 환경 변수와 LangSmith 추적을 설정합니다. 환경 변수에는 LLM 서비스 인증 정보가 포함되며, LangSmith를 통해 에이전트의 컨텍스트 흐름을 상세히 모니터링할 수 있습니다.
#
# 아래 코드는 `.env` 파일에서 환경 변수를 로드하고, LangSmith 추적을 활성화합니다.

# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %%
from langchain_teddynote import logging

logging.langsmith("LangChain-V1-Tutorial")

# %% [markdown]
# ## 컨텍스트의 종류
#
# 에이전트는 세 가지 종류의 컨텍스트를 제어합니다. 각 컨텍스트 타입은 서로 다른 범위와 지속성을 가지며, 에이전트의 동작을 세밀하게 조정하는 데 사용됩니다.
#
# | 컨텍스트 타입 | 제어 대상 | 지속성 |
# |------------|---------|-------|
# | **Model Context** | 모델 호출에 들어가는 내용 (지시사항, 메시지 기록, 도구, 응답 형식) | Transient |
# | **Tool Context** | 도구가 액세스하고 생성하는 내용 (상태, 저장소, 런타임 컨텍스트에 읽기/쓰기) | Persistent |
# | **Life-cycle Context** | 모델 및 도구 호출 사이에 발생하는 작업 (요약, 가드레일, 로깅 등) | Persistent |
#
# **Transient Context**: LLM이 단일 호출에서 보는 내용입니다. 상태에 저장된 내용을 변경하지 않고 메시지, 도구 또는 프롬프트를 수정할 수 있습니다.
#
# **Persistent Context**: 여러 턴에 걸쳐 상태에 저장되는 내용입니다. 라이프사이클 훅과 도구 쓰기는 이를 영구적으로 수정합니다.

# %% [markdown]
# ## 데이터 소스
#
# 에이전트는 다양한 데이터 소스에 액세스(읽기/쓰기)합니다. 각 데이터 소스는 서로 다른 범위와 용도를 가지며, 적절한 데이터 소스를 선택하는 것이 중요합니다.
#
# | 데이터 소스 | 다른 이름 | 범위 | 예시 |
# |----------|---------|------|-----|
# | **Runtime Context** | 정적 구성 | 대화 범위 | 사용자 ID, API 키, DB 연결, 권한 |
# | **State** | 단기 메모리 | 대화 범위 | 현재 메시지, 업로드된 파일, 인증 상태 |
# | **Store** | 장기 메모리 | 대화 간 공유 | 사용자 선호도, 추출된 인사이트, 기록 데이터 |
#
# **Runtime Context**는 에이전트 호출 시 전달되는 정적 구성이고, **State**는 현재 대화 세션 내에서 유지되는 데이터이며, **Store**는 여러 대화 세션에 걸쳐 지속되는 장기 데이터입니다.
#
# <system-reminder>
# The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user
#
#
# Here are the existing contents of your todo list:
#
# [1. [completed] 01-LangGraph-Middleware.ipynb 교정
# 2. [completed] 02-LangGraph-Human-In-The-Loop.ipynb 교정
# 3. [in_progress] 03-LangGraph-Context-Engineering.ipynb 교정
# 4. [pending] 04-LangGraph-Guardrail.ipynb 교정
# 5. [pending] 전체 교정 결과 정리 및 보고]
# </system-reminder>

# %% [markdown]
# ## Model Context
#
# Model Context는 각 모델 호출에 들어가는 내용을 제어합니다. 여기에는 지시사항(System Prompt), 사용 가능한 도구, 사용할 모델, 출력 형식 등이 포함됩니다. Model Context를 잘 설계하면 LLM이 적절한 정보를 바탕으로 올바른 결정을 내릴 수 있습니다.

# %% [markdown]
# ### System Prompt
#
# 시스템 프롬프트는 LLM의 동작과 능력을 설정하는 핵심 요소입니다. 다양한 사용자, 컨텍스트, 또는 대화 단계에 따라 다른 지시사항이 필요할 수 있습니다. `@dynamic_prompt` 데코레이터를 사용하면 런타임에 시스템 프롬프트를 동적으로 생성할 수 있습니다.

# %% [markdown]
# ### State 기반 System Prompt
#
# 대화 State에 따라 시스템 프롬프트를 동적으로 조정할 수 있습니다. 예를 들어, 대화가 길어지면 더 간결하게 응답하도록 지시하거나, 특정 상태에 따라 다른 동작을 수행하도록 설정할 수 있습니다.
#
# 아래 코드는 메시지 수에 따라 시스템 프롬프트를 조정하는 `@dynamic_prompt` 미들웨어를 구현합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_openai import ChatOpenAI
from langchain.tools import tool


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """State의 메시지 수에 따라 프롬프트 조정"""
    # request.messages는 request.state["messages"]의 단축형
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."

    return base


model = ChatOpenAI(model="gpt-4.1-mini")

agent = create_agent(model=model, tools=[search_tool], middleware=[state_aware_prompt])

# 짧은 대화
result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
print("Short conversation:", result["messages"][-1].content[:100])

# %% [markdown]
# ### Store 기반 System Prompt
#
# Store는 여러 대화 세션에 걸쳐 지속되는 장기 데이터를 저장합니다. 사용자 선호도, 커뮤니케이션 스타일 등을 Store에 저장하고 시스템 프롬프트에 반영하면 개인화된 경험을 제공할 수 있습니다.
#
# 아래 코드는 Store에서 사용자 선호도를 읽어 시스템 프롬프트에 반영하는 예제입니다. 동일한 질문에 대해 사용자별로 다른 스타일의 응답을 제공합니다.

# %%
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore
from langchain_teddynote.messages import invoke_graph, stream_graph
from langchain_core.messages import HumanMessage


@dataclass
class Context:
    user_id: str


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """Store에서 사용자 선호도를 가져와서 프롬프트 조정"""
    user_id = request.runtime.context.user_id

    # Store에서 사용자 선호도 읽기
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "")
        base += f"\nUser prefers {style} responses."
        print(f"User prefers {style} responses.")
    else:
        base += "\nUser prefers professional tone. Answer in 3 sentences."

    return base


# Store 생성 및 초기화
store = InMemoryStore()
store.put(
    ("preferences",),
    "teddy",
    {
        "communication_style": "아주 간결하고 emoji 를 사용하여 친근감 있는 스타일. bullet point 로 정리된 답변."
    },
)

agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=store,
)

# %%
# user_id: teddy
stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해줘")]
    },
    context=Context(user_id="teddy"),
)

# %%
# user_id: other
stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="머신러닝의 동작 원리에 대해서 설명해줘")]
    },
    context=Context(user_id="other"),
)


# %% [markdown]
# ### Runtime Context 기반 System Prompt
#
# Runtime Context는 에이전트 호출 시 전달되는 정적 구성입니다. 사용자 역할, 배포 환경 등 호출 시점에 결정되는 정보를 기반으로 시스템 프롬프트를 조정할 수 있습니다.
#
# 아래 코드는 사용자 역할(`admin`/`viewer`)과 배포 환경(`production`)에 따라 시스템 프롬프트를 조정하는 예제입니다.

# %%
@dataclass
class RoleContext:
    user_role: str
    deployment_env: str


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """Runtime Context에서 사용자 역할과 환경을 기반으로 프롬프트 조정"""
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."

    if env == "production":
        base += "\nBe extra careful with any data modifications."
    print(f"User role: {user_role}, Deployment environment: {env}")
    return base


agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[context_aware_prompt],
    context_schema=RoleContext,
)

# %%
# Admin 사용자
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="데이터 추가가 가능한가요?")]},
    context=RoleContext(user_role="admin", deployment_env="production"),
)

# %%
# viewer 사용자
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="데이터 추가가 가능한가요?")]},
    context=RoleContext(user_role="viewer", deployment_env="production"),
)

# %% [markdown]
# ### Messages
#
# 메시지는 LLM에 전송되는 프롬프트를 구성합니다. LLM이 올바른 정보를 가지고 잘 응답할 수 있도록 메시지 내용을 관리하는 것이 중요합니다. `@wrap_model_call` 데코레이터를 사용하면 모델에 전달되는 메시지를 동적으로 수정할 수 있습니다.

# %% [markdown]
# ### State에서 파일 컨텍스트 주입
#
# 사용자가 업로드한 파일이나 첨부 자료가 있을 때, 이 정보를 메시지에 자동으로 추가하여 LLM이 참조할 수 있도록 할 수 있습니다. `@wrap_model_call` 데코레이터를 사용하여 State에 저장된 파일 메타데이터를 읽고 컨텍스트로 주입합니다.
#
# 아래 코드는 State의 `uploaded_files` 필드에서 파일 정보를 읽어 모델 요청에 추가하는 예제입니다.

# %%
from langchain.agents.middleware import wrap_model_call, ModelResponse
from typing import Callable


@wrap_model_call
def inject_file_context(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """사용자가 업로드한 파일 컨텍스트를 주입"""
    # State에서 업로드된 파일 메타데이터 가져오기
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        # 사용 가능한 파일에 대한 컨텍스트 구축
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        # 최근 메시지 앞에 파일 컨텍스트 주입
        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)


agent = create_agent(model=model, tools=[search_tool], middleware=[inject_file_context])

# 파일이 업로드된 상태로 호출
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What files do I have?"}],
        "uploaded_files": [
            {"name": "report.pdf", "type": "PDF", "summary": "Q4 sales report"},
            {"name": "data.csv", "type": "CSV", "summary": "Customer data"},
        ],
    }
)

print(result["messages"][-1].content)

# %% [markdown]
# ### Tools
#
# 도구를 통해 모델이 데이터베이스, API, 외부 시스템과 상호 작용할 수 있습니다. 도구를 정의하고 선택하는 방법은 모델이 작업을 효과적으로 완료할 수 있는지에 직접적인 영향을 미칩니다. 좋은 도구 정의는 모델의 추론을 안내하고, 올바른 도구를 올바른 시점에 사용하도록 돕습니다.

# %% [markdown]
# ### 도구 정의
#
# 각 도구에는 명확한 이름, 설명, 인수 이름 및 인수 설명이 필요합니다. 이것들은 단순한 메타데이터가 아니라 모델이 도구를 언제 어떻게 사용할지에 대한 추론을 안내합니다. `parse_docstring=True` 옵션을 사용하면 docstring에서 인수 설명을 자동으로 추출할 수 있습니다.
#
# 아래 코드는 명확한 설명과 인수 정의를 가진 도구를 생성하는 예제입니다.

# %%
from langchain.tools import tool


@tool(parse_docstring=True)
def search_orders(user_id: str, status: str, limit: int = 10) -> str:
    """Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.

    Args:
        user_id: Unique identifier for the user
        status: Order status: 'pending', 'shipped', or 'delivered'
        limit: Maximum number of results to return
    """
    return f"Found orders for {user_id} with status {status} (limit: {limit})"


agent = create_agent(model=model, tools=[search_orders])

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Show me my pending orders for user_123"}
        ]
    }
)

print(result["messages"][-1].content)

# %% [markdown]
# ### State 기반 도구 선택
#
# 대화 단계나 상태에 따라 사용 가능한 도구를 동적으로 조정할 수 있습니다. 예를 들어, 인증되지 않은 사용자에게는 공개 도구만 제공하고, 인증된 사용자에게는 더 많은 도구를 제공할 수 있습니다. 이를 통해 보안과 사용자 경험을 동시에 관리할 수 있습니다.
#
# 아래 코드는 인증 상태와 대화 길이에 따라 도구를 필터링하는 예제입니다.

# %%
from langchain.tools import tool


@tool
def public_search(query: str) -> str:
    """Public search - available to all users."""
    return f"Public results for: {query}"


@tool
def private_search(query: str) -> str:
    """Private search - requires authentication."""
    return f"Private results for: {query}"


@tool
def advanced_search(query: str) -> str:
    """Advanced search - requires authentication and conversation history."""
    return f"Advanced results for: {query}"


@wrap_model_call
def state_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """대화 State에 따라 도구 필터링"""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # 인증되지 않은 경우 공개 도구만 활성화
    if not is_authenticated:
        tools = [t for t in request.tools if t.name == "public_search"]
        request = request.override(tools=tools)
    elif message_count < 5:
        # 대화 초반에는 고급 도구 제한
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model=model,
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools],
)

# 인증되지 않은 사용자
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Search for Python tutorials"}],
        "authenticated": False,
    }
)
print("Unauthenticated:", result["messages"][-1].content)

# 인증된 사용자
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Search for Python tutorials"}],
        "authenticated": True,
    }
)
print("\nAuthenticated:", result["messages"][-1].content)


# %% [markdown]
# ### Runtime Context 기반 도구 선택
#
# Runtime Context에 전달된 사용자 권한에 따라 도구를 필터링할 수 있습니다. 관리자는 모든 도구를, 편집자는 삭제 도구를 제외한 도구를, 뷰어는 읽기 도구만 사용할 수 있도록 설정하면 역할 기반 접근 제어(RBAC)를 구현할 수 있습니다.
#
# 아래 코드는 사용자 역할에 따라 도구 접근을 제한하는 예제입니다.

# %%
@tool
def read_data(table: str) -> str:
    """테이블에서 데이터를 읽어옵니다."""
    return f"{table} 테이블에서 데이터를 읽었습니다."


@tool
def write_data(table: str) -> str:
    """테이블에 데이터를 작성합니다."""
    return f"{table} 테이블에 데이터를 작성했습니다."


@tool
def delete_data(table: str, data_id: str) -> str:
    """테이블에서 데이터를 삭제합니다."""
    return f"{table} 테이블에서 데이터를 삭제했습니다."


@dataclass
class UserRole:
    user_role: str


@wrap_model_call
def context_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Runtime Context 권한에 따라 도구 필터링"""
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        # 관리자는 모든 도구 사용 가능
        pass
    elif user_role == "editor":
        # 편집자는 삭제 도구를 사용할 수 없습니다.
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # 뷰어는 읽기 전용 도구만 사용할 수 있습니다.
        tools = [t for t in request.tools if t.name == "read_data"]
        request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model=model,
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=UserRole,
    system_prompt="사용자의 요구사항을 바로 수행해 주세요. 주어진 도구를 사용해 주세요. 사용할 도구가 없다면, 권한이 없다고 답변하세요.",
)

# %%
# 뷰어
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="User 테이블을 조회하세요.")]},
    context=UserRole(user_role="viewer"),
)

# %%
# 뷰어
stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="User 테이블에서 abc 레코드를 삭제해 주세요")]
    },
    context=UserRole(user_role="viewer"),
)

# %%
# 관리자
stream_graph(
    agent,
    inputs={
        "messages": [HumanMessage(content="User 테이블에서 abc 레코드를 삭제해 주세요")]
    },
    context=UserRole(user_role="admin"),
)

# %% [markdown]
# ### Model
#
# 다양한 모델은 각기 다른 강점, 비용, 컨텍스트 창 크기를 가지고 있습니다. 작업의 복잡성, 대화 길이, 비용 요구사항에 따라 적합한 모델을 동적으로 선택할 수 있습니다. `@wrap_model_call` 데코레이터를 사용하면 런타임에 모델을 변경할 수 있습니다.
#
# 아래 코드는 대화 길이에 따라 효율적인 모델과 큰 모델을 동적으로 선택하는 예제입니다.

# %%
from langchain.chat_models import init_chat_model

# 모델을 미들웨어 외부에서 한 번만 초기화
large_model = init_chat_model("openai:gpt-4.1")
efficient_model = init_chat_model("openai:gpt-4.1-mini")


@wrap_model_call
def state_based_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """대화 길이에 따라 모델 선택"""
    message_count = len(request.messages)

    if message_count > 10:
        # 긴 대화 - 큰 컨텍스트 창을 가진 모델 사용
        model = large_model
        print(f"Using large model for {message_count} messages")
    else:
        # 짧은 대화 - 효율적인 모델 사용
        model = efficient_model
        print(f"Using efficient model for {message_count} messages")

    request = request.override(model=model)
    return handler(request)


agent = create_agent(
    model=efficient_model, tools=[search_tool], middleware=[state_based_model]
)

# 짧은 대화
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
print(result["messages"][-1].content[:100])

# %% [markdown]
# ### Response Format
#
# 구조화된 출력은 비구조화된 텍스트를 검증된 구조화 데이터로 변환합니다. 특정 필드를 추출하거나 다운스트림 시스템을 위한 데이터를 반환할 때 자유 형식 텍스트로는 충분하지 않습니다. Pydantic 모델을 사용하여 출력 스키마를 정의하면 모델이 해당 형식으로 응답하도록 강제할 수 있습니다.
#
# 아래 코드는 고객 지원 티켓 정보를 구조화된 형식으로 추출하는 예제입니다.

# %%
from pydantic import BaseModel, Field


class CustomerSupportTicket(BaseModel):
    """고객 메시지에서 추출된 구조화된 티켓 정보"""

    category: str = Field(
        description="Issue category: 'billing', 'technical', 'account', or 'product'"
    )
    priority: str = Field(
        description="Urgency level: 'low', 'medium', 'high', or 'critical'"
    )
    summary: str = Field(description="One-sentence summary of the customer's issue")
    customer_sentiment: str = Field(
        description="Customer's emotional tone: 'frustrated', 'neutral', or 'satisfied'"
    )


agent = create_agent(
    model=model, tools=[search_tool], response_format=CustomerSupportTicket
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I can't login to my account! I've been trying for an hour and it keeps saying invalid credentials.",
            }
        ]
    }
)

# 결과는 CustomerSupportTicket 형식으로 반환됨
print("Ticket:", result["messages"][-1].content)


# %% [markdown]
# ### State 기반 Response Format 선택
#
# 대화 상태에 따라 다른 출력 형식을 적용할 수 있습니다. 초기 대화에서는 간단한 형식을, 대화가 진행되면서 더 상세한 형식을 사용하면 상황에 맞는 응답을 제공할 수 있습니다.
#
# 아래 코드는 메시지 수에 따라 간단한 응답 형식과 상세한 응답 형식을 동적으로 선택하는 예제입니다.

# %%
class SimpleResponse(BaseModel):
    """초기 대화를 위한 간단한 응답"""

    answer: str = Field(description="A brief answer")


class DetailedResponse(BaseModel):
    """확립된 대화를 위한 상세한 응답"""

    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")


@wrap_model_call
def state_based_output(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """State에 따라 출력 형식 선택"""
    message_count = len(request.messages)

    if message_count < 3:
        # 초기 대화 - 간단한 형식 사용
        request = request.override(response_format=SimpleResponse)
    else:
        # 확립된 대화 - 상세한 형식 사용
        request = request.override(response_format=DetailedResponse)

    return handler(request)


agent = create_agent(model=model, tools=[search_tool], middleware=[state_based_output])

# 첫 번째 메시지 - 간단한 응답
result = agent.invoke({"messages": [{"role": "user", "content": "What is Python?"}]})
print("Simple response:", result["messages"][-1].content)

# %% [markdown]
# ## Tool Context
#
# 도구는 컨텍스트를 읽고 쓰는 두 가지 작업을 모두 수행합니다. `ToolRuntime` 객체를 통해 State, Store, Runtime Context에 접근할 수 있으며, `Command` 객체를 반환하여 State를 업데이트할 수 있습니다. 이를 통해 도구가 단순한 함수 호출 이상의 역할을 수행할 수 있습니다.

# %% [markdown]
# ### Reads - State에서 읽기
#
# 도구 함수에 `runtime: ToolRuntime` 매개변수를 추가하면 현재 State에 접근할 수 있습니다. 인증 상태, 세션 정보 등 State에 저장된 데이터를 읽어 도구의 동작을 결정할 수 있습니다.
#
# 아래 코드는 State에서 인증 상태를 확인하는 도구를 구현합니다.

# %%
from langchain.tools import tool, ToolRuntime


@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check if user is authenticated."""
    # State에서 현재 인증 상태 확인
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)

    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"


agent = create_agent(model=model, tools=[check_authentication])

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Am I authenticated?"}],
        "authenticated": True,
    }
)

print(result["messages"][-1].content)


# %% [markdown]
# ### Reads - Store에서 읽기
#
# Store는 여러 대화 세션에 걸쳐 지속되는 장기 데이터를 저장합니다. `runtime.store.get()` 메서드를 사용하여 Store에서 데이터를 읽을 수 있습니다. `ToolRuntime[Context]` 타입 힌트를 사용하면 Runtime Context 타입도 함께 지정할 수 있습니다.
#
# 아래 코드는 Store에서 사용자 선호도를 읽어 반환하는 도구를 구현합니다.

# %%
@dataclass
class Context:
    user_id: str


@tool
def get_preference(preference_key: str, runtime: ToolRuntime[Context]) -> str:
    """Get user preference from Store."""
    user_id = runtime.context.user_id

    # Store에서 기존 선호도 읽기
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    if existing_prefs:
        value = existing_prefs.value.get(preference_key)
        return (
            f"{preference_key}: {value}"
            if value
            else f"No preference set for {preference_key}"
        )
    else:
        return "No preferences found"


store = InMemoryStore()
store.put(("preferences",), "user_123", {"theme": "dark", "language": "ko"})

agent = create_agent(
    model=model, tools=[get_preference], context_schema=Context, store=store
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my theme preference?"}]},
    context=Context(user_id="user_123"),
)

print(result["messages"][-1].content)

# %% [markdown]
# ### Writes - State에 쓰기
#
# 도구에서 `Command(update={...})`를 반환하면 State를 업데이트할 수 있습니다. 인증 상태 변경, 세션 데이터 저장 등 도구 실행 결과를 State에 기록하여 후속 작업에서 참조할 수 있습니다.
#
# 아래 코드는 비밀번호를 확인하고 인증 상태를 State에 기록하는 도구를 구현합니다.

# %%
from langgraph.types import Command


@tool
def authenticate_user(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate user and update State."""
    # 인증 수행 (단순화됨)
    if password == "correct":
        # Command를 사용하여 State에 인증 상태 기록
        return Command(update={"authenticated": True})
    else:
        return Command(update={"authenticated": False})


agent = create_agent(model=model, tools=[authenticate_user, check_authentication])

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Authenticate with password 'correct' then check my status",
            }
        ]
    }
)

print(result["messages"][-1].content)
print("Authenticated in state:", result.get("authenticated", False))


# %% [markdown]
# ### Writes - Store에 쓰기
#
# `runtime.store.put()` 메서드를 사용하면 Store에 데이터를 저장할 수 있습니다. Store에 저장된 데이터는 여러 대화 세션에 걸쳐 지속되므로 사용자 선호도, 히스토리 등 장기 데이터를 저장하는 데 적합합니다.
#
# 아래 코드는 사용자 선호도를 Store에 저장하는 도구를 구현합니다.

# %%
@tool
def save_preference(
    preference_key: str, preference_value: str, runtime: ToolRuntime[Context]
) -> str:
    """Save user preference to Store."""
    user_id = runtime.context.user_id

    # 기존 선호도 읽기
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    # 새 선호도와 병합
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value

    # Store에 업데이트된 선호도 저장
    store.put(("preferences",), user_id, prefs)

    return f"Saved preference: {preference_key} = {preference_value}"


store = InMemoryStore()

agent = create_agent(
    model=model,
    tools=[save_preference, get_preference],
    context_schema=Context,
    store=store,
)

# 선호도 저장
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Set my theme to dark"}]},
    context=Context(user_id="user_456"),
)
print(result["messages"][-1].content)

# 저장된 선호도 확인
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my theme?"}]},
    context=Context(user_id="user_456"),
)
print(result["messages"][-1].content)

# %% [markdown]
# ## Life-cycle Context
#
# Life-cycle Context는 핵심 에이전트 단계 **사이**에서 발생하는 작업을 제어합니다. 데이터 흐름을 가로채어 요약, 가드레일, 로깅과 같은 교차 관심사(cross-cutting concerns)를 구현할 수 있습니다. 미들웨어를 통해 모든 모델 호출과 도구 호출에 일관된 로직을 적용할 수 있습니다.

# %% [markdown]
# ### Summarization
#
# 가장 일반적인 라이프사이클 패턴 중 하나는 대화 기록이 너무 길어질 때 자동으로 압축하는 것입니다. 요약은 **상태를 영구적으로 업데이트**합니다 - 오래된 메시지를 요약으로 영구적으로 대체하여 모든 향후 턴에 대해 저장됩니다. 이를 통해 토큰 비용을 절감하면서도 중요한 컨텍스트를 유지할 수 있습니다.
#
# 아래 코드는 `SummarizationMiddleware`를 사용하여 4000 토큰 초과 시 자동 요약을 수행하는 에이전트를 생성합니다.

# %%
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",
            max_tokens_before_summary=4000,  # 4000 토큰에서 요약 트리거
            messages_to_keep=20,  # 요약 후 최근 20개 메시지 유지
        ),
    ],
)

# 대화가 길어지면 자동으로 요약됨
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about Python"}]}
)

print(result["messages"][-1].content[:200])

# %% [markdown]
# ## 종합 예제: 다층 컨텍스트 엔지니어링
#
# 이 섹션에서는 지금까지 학습한 모든 컨텍스트 엔지니어링 기술을 결합한 실용적인 예제를 구현합니다. 사용자 역할과 구독 등급에 따라 시스템 프롬프트와 도구 접근을 조정하고, Store를 활용하여 사용자별 히스토리를 관리합니다.
#
# 아래 코드는 Runtime Context, State, Store, 동적 프롬프트, 도구 필터링을 모두 활용하는 종합 예제입니다.

# %%
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from typing import Callable


@dataclass
class UserContext:
    user_id: str
    user_role: str
    subscription_tier: str


# 도구 정의
@tool
def get_user_history(runtime: ToolRuntime[UserContext]) -> str:
    """Get user's search history."""
    user_id = runtime.context.user_id
    store = runtime.store

    history = store.get(("history",), user_id)
    if history:
        return f"Recent searches: {history.value.get('searches', [])}"
    return "No history found"


@tool
def save_search(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save search query."""
    user_id = runtime.context.user_id
    store = runtime.store

    # 기존 히스토리 가져오기
    existing = store.get(("history",), user_id)
    searches = existing.value.get("searches", []) if existing else []

    # 새 검색 추가
    searches.append(query)
    store.put(("history",), user_id, {"searches": searches[-5:]})

    return f"Saved: {query}"


@tool
def advanced_analysis(data: str, runtime: ToolRuntime[UserContext]) -> str:
    """Perform advanced analysis (premium feature)."""
    tier = runtime.context.subscription_tier
    if tier != "premium":
        return "This feature requires a premium subscription"
    return f"Advanced analysis results for: {data}"


# 동적 시스템 프롬프트
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    tier = request.runtime.context.subscription_tier

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += " You have admin privileges."

    if tier == "premium":
        base += " This user has premium features enabled."

    return base


# 구독 등급에 따른 도구 필터링
@wrap_model_call
def tier_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    tier = request.runtime.context.subscription_tier

    if tier != "premium":
        # 무료 사용자는 고급 분석 불가
        tools = [t for t in request.tools if t.name != "advanced_analysis"]
        request = request.override(tools=tools)

    return handler(request)


# Store 초기화
store = InMemoryStore()
store.put(("history",), "user_001", {"searches": ["Python", "Machine Learning"]})

# 에이전트 생성
agent = create_agent(
    model=model,
    tools=[get_user_history, save_search, advanced_analysis],
    middleware=[
        role_based_prompt,
        tier_based_tools,
    ],
    context_schema=UserContext,
    store=store,
)

# 테스트 1: 프리미엄 사용자
print("=== Premium User ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Get my search history and perform advanced analysis",
            }
        ]
    },
    context=UserContext(
        user_id="user_001", user_role="admin", subscription_tier="premium"
    ),
)
print(result["messages"][-1].content)

# 테스트 2: 무료 사용자
print("\n=== Free User ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Perform advanced analysis on my data"}]},
    context=UserContext(user_id="user_002", user_role="user", subscription_tier="free"),
)
print(result["messages"][-1].content)
