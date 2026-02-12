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
# # Long-term Memory (장기 메모리)
#
# LangChain 에이전트는 LangGraph persistence를 사용하여 장기 메모리를 활성화합니다.
#
# ## 개요
#
# 장기 메모리는 대화 세션 간에 정보를 유지하여 에이전트가 사용자 선호도, 과거 상호작용 및 컨텍스트를 기억할 수 있게 합니다.
#
# ## 단기 메모리 vs 장기 메모리
#
# | 특성 | 단기 메모리 | 장기 메모리 |
# |------|-----------|----------|
# | **범위** | 단일 대화 세션 | 여러 세션에 걸쳐 유지 |
# | **저장 위치** | Checkpointer (상태) | Store (영구 저장소) |
# | **데이터 유형** | 메시지, 임시 상태 | 사용자 프로필, 선호도, 학습된 정보 |
# | **수명** | 세션 종료 시 사라질 수 있음 | 명시적으로 삭제할 때까지 유지 |
# | **예시** | 현재 대화 내용 | 사용자 이름, 언어 설정, 과거 구매 이력 |

# %% [markdown]
# ## 사전 준비
#
# 환경 변수를 설정합니다.

# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %% [markdown]
# ## Memory Storage
#
# LangGraph는 장기 메모리를 Store에 JSON 문서로 저장합니다.
#
# 각 메모리는 다음과 같이 구성됩니다:
# - **Namespace**: 폴더와 유사한 커스텀 네임스페이스 (예: 사용자 ID, 조직 ID)
# - **Key**: 파일 이름과 유사한 고유 키
# - **Value**: 저장할 데이터 (JSON)

# %%
from langgraph.store.memory import InMemoryStore

# InMemoryStore는 메모리 내 딕셔너리에 데이터 저장 (프로덕션에서는 DB 기반 저장소 사용)
store = InMemoryStore()

# 네임스페이스 정의
user_id = "user_123"
application_context = "preferences"
namespace = (user_id, application_context)

# 데이터 저장
store.put(
    namespace,
    "language_preferences",  # Key
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "preferred_language": "English",
        "communication_style": "concise"
    }
)

print(f"Stored data in namespace: {namespace}")

# ID로 메모리 가져오기
item = store.get(namespace, "language_preferences")
print(f"\nRetrieved item: {item.value}")

# 네임스페이스 내에서 검색
items = store.search(
    namespace,
    filter={"preferred_language": "English"}
)
print(f"\nSearch results: {[item.value for item in items]}")

# %% [markdown]
# ## 도구에서 장기 메모리 읽기
#
# 도구는 `runtime.store`를 통해 장기 메모리에 액세스할 수 있습니다.

# %%
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

@dataclass
class Context:
    user_id: str

# Store 생성 및 샘플 데이터 저장
store = InMemoryStore()

store.put(
    ("users",),
    "user_123",
    {
        "name": "김철수",
        "language": "Korean",
        "email": "chulsoo@example.com",
        "subscription": "premium"
    }
)

store.put(
    ("users",),
    "user_456",
    {
        "name": "Jane Doe",
        "language": "English",
        "email": "jane@example.com",
        "subscription": "free"
    }
)

# 사용자 정보를 조회하는 도구
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user information from the store."""
    # Store 액세스
    store = runtime.store
    user_id = runtime.context.user_id
    
    # Store에서 데이터 검색
    user_info = store.get(("users",), user_id)
    
    if user_info:
        return f"User: {user_info.value}"
    else:
        return "Unknown user"

model = ChatOpenAI(model="gpt-4.1-mini")

# 에이전트 생성
agent = create_agent(
    model=model,
    tools=[get_user_info],
    store=store,  # Store 전달
    context_schema=Context
)

# 에이전트 실행
result = agent.invoke(
    {"messages": [{"role": "user", "content": "내 정보를 조회해줘"}]},
    context=Context(user_id="user_123")
)

print(result["messages"][-1].content)

# %% [markdown]
# ## 도구에서 장기 메모리 쓰기
#
# 도구는 `runtime.store`를 사용하여 장기 메모리를 업데이트할 수 있습니다.

# %%
from typing_extensions import TypedDict

# 사용자 정보 구조 정의
class UserInfo(TypedDict):
    name: str
    email: str
    language: str

# 사용자 정보를 저장하는 도구
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save or update user information."""
    # Store 액세스
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 기존 정보 가져오기
    existing = store.get(("users",), user_id)
    
    if existing:
        # 기존 정보와 병합
        updated_info = {**existing.value, **user_info}
        store.put(("users",), user_id, updated_info)
        return f"Successfully updated user info: {updated_info}"
    else:
        # 새로운 정보 저장
        store.put(("users",), user_id, user_info)
        return f"Successfully saved new user info: {user_info}"

# 에이전트 생성
agent = create_agent(
    model=model,
    tools=[get_user_info, save_user_info],
    store=store,
    context_schema=Context
)

# 새 사용자 정보 저장
result = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice Park and my email is alice@example.com"}]},
    context=Context(user_id="user_789")
)

print("Save result:", result["messages"][-1].content)

# 저장된 정보 확인
saved_info = store.get(("users",), "user_789")
print("\nStored data:", saved_info.value if saved_info else "None")

# %% [markdown]
# ## 실용적인 예제: 사용자 선호도 관리
#
# 사용자 선호도를 저장하고 활용하는 완전한 예제입니다.

# %%
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# Store 초기화
preference_store = InMemoryStore()

# 샘플 사용자 선호도
preference_store.put(
    ("preferences",),
    "user_001",
    {
        "communication_style": "formal",
        "response_length": "detailed",
        "topics_of_interest": ["technology", "science"],
        "language": "Korean"
    }
)

@tool
def get_preferences(runtime: ToolRuntime[Context]) -> str:
    """Get user preferences."""
    store = runtime.store
    user_id = runtime.context.user_id
    
    prefs = store.get(("preferences",), user_id)
    if prefs:
        return str(prefs.value)
    else:
        return "No preferences found"

@tool
def update_preference(preference_key: str, preference_value: str, runtime: ToolRuntime[Context]) -> str:
    """Update a specific user preference."""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 기존 선호도 가져오기
    existing = store.get(("preferences",), user_id)
    
    if existing:
        prefs = existing.value
        prefs[preference_key] = preference_value
        store.put(("preferences",), user_id, prefs)
        return f"Updated {preference_key} to {preference_value}"
    else:
        # 새 선호도 생성
        store.put(("preferences",), user_id, {preference_key: preference_value})
        return f"Created new preference: {preference_key} = {preference_value}"

# 선호도 기반 동적 프롬프트
@dynamic_prompt
def preference_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    
    prefs = store.get(("preferences",), user_id)
    
    base = "You are a helpful assistant."
    
    if prefs:
        pref_data = prefs.value
        style = pref_data.get("communication_style", "casual")
        length = pref_data.get("response_length", "moderate")
        language = pref_data.get("language", "English")
        
        base += f"\nCommunication style: {style}"
        base += f"\nResponse length: {length}"
        base += f"\nPreferred language: {language}"
    
    return base

# 선호도 인식 에이전트
preference_agent = create_agent(
    model=model,
    tools=[get_preferences, update_preference],
    middleware=[preference_aware_prompt],
    store=preference_store,
    context_schema=Context
)

print("=== Test 1: Using existing preferences ===")
result = preference_agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about AI"}]},
    context=Context(user_id="user_001")
)
print(result["messages"][-1].content)

print("\n=== Test 2: Update preference ===")
result = preference_agent.invoke(
    {"messages": [{"role": "user", "content": "Change my communication style to casual"}]},
    context=Context(user_id="user_001")
)
print(result["messages"][-1].content)

print("\n=== Test 3: Check updated preferences ===")
updated_prefs = preference_store.get(("preferences",), "user_001")
print(f"Updated preferences: {updated_prefs.value}")

# %% [markdown]
# ## 계층적 네임스페이스
#
# 네임스페이스는 계층 구조를 가질 수 있어 데이터를 체계적으로 구성할 수 있습니다.

# %%
# 계층적 네임스페이스 예제
hierarchical_store = InMemoryStore()

# 사용자 기본 정보
hierarchical_store.put(
    ("users", "user_001", "profile"),
    "basic",
    {"name": "John Doe", "age": 30}
)

# 사용자 설정
hierarchical_store.put(
    ("users", "user_001", "settings"),
    "preferences",
    {"theme": "dark", "notifications": True}
)

# 사용자 활동
hierarchical_store.put(
    ("users", "user_001", "activity"),
    "recent",
    {"last_login": "2024-01-01", "page_views": 42}
)

# 다른 네임스페이스 조회
profile = hierarchical_store.get(("users", "user_001", "profile"), "basic")
settings = hierarchical_store.get(("users", "user_001", "settings"), "preferences")
activity = hierarchical_store.get(("users", "user_001", "activity"), "recent")

print("Profile:", profile.value)
print("Settings:", settings.value)
print("Activity:", activity.value)

# %% [markdown]
# ## 고급 예제: 학습하는 에이전트
#
# 사용자와의 상호작용에서 학습하고 저장하는 에이전트입니다.

# %%
learning_store = InMemoryStore()

@tool
def learn_from_interaction(fact: str, category: str, runtime: ToolRuntime[Context]) -> str:
    """Learn and store information from user interactions."""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 카테고리별 학습 내용 저장
    namespace = ("learned", user_id, category)
    
    # 기존 학습 내용 가져오기
    existing = store.get(namespace, "facts")
    
    if existing:
        facts = existing.value.get("facts", [])
        facts.append(fact)
        store.put(namespace, "facts", {"facts": facts})
    else:
        store.put(namespace, "facts", {"facts": [fact]})
    
    return f"Learned: {fact} (category: {category})"

@tool
def recall_learned_info(category: str, runtime: ToolRuntime[Context]) -> str:
    """Recall previously learned information."""
    store = runtime.store
    user_id = runtime.context.user_id
    
    namespace = ("learned", user_id, category)
    learned = store.get(namespace, "facts")
    
    if learned:
        facts = learned.value.get("facts", [])
        return f"I remember these facts about {category}: {', '.join(facts)}"
    else:
        return f"I don't have any learned information about {category}"

# 학습하는 에이전트
learning_agent = create_agent(
    model=model,
    tools=[learn_from_interaction, recall_learned_info],
    store=learning_store,
    context_schema=Context,
    system_prompt="""You are a learning assistant.
    
    When users tell you facts about themselves, use learn_from_interaction to remember them.
    When users ask what you know about a topic, use recall_learned_info.
    
    Categories: personal, work, hobbies, preferences"""
)

# 학습 단계
print("=== Learning Phase ===")
interactions = [
    "I work as a software engineer at a tech company",
    "My hobby is playing guitar",
    "I prefer working in the morning",
]

for interaction in interactions:
    result = learning_agent.invoke(
        {"messages": [{"role": "user", "content": interaction}]},
        context=Context(user_id="user_learning")
    )
    print(f"User: {interaction}")
    print(f"Agent: {result['messages'][-1].content}\n")

# 회상 단계
print("\n=== Recall Phase ===")
result = learning_agent.invoke(
    {"messages": [{"role": "user", "content": "What do you know about my work?"}]},
    context=Context(user_id="user_learning")
)
print(f"Agent: {result['messages'][-1].content}")

result = learning_agent.invoke(
    {"messages": [{"role": "user", "content": "What are my hobbies?"}]},
    context=Context(user_id="user_learning")
)
print(f"Agent: {result['messages'][-1].content}")

# %% [markdown]
# ## 프로덕션 사용: PostgreSQL Store
#
# 프로덕션 환경에서는 데이터베이스 기반 저장소를 사용해야 합니다.

# %%
# PostgreSQL Store 예제 (설치 필요: pip install langgraph-checkpoint-postgres)

# from langgraph.store.postgres import PostgresStore

# DB_URI = "postgresql://user:password@localhost:5432/dbname"

# async with PostgresStore.from_conn_string(DB_URI) as store:
#     # Store 초기화
#     await store.setup()
#     
#     # 데이터 저장
#     await store.aput(
#         ("users",),
#         "user_123",
#         {"name": "John", "email": "john@example.com"}
#     )
#     
#     # 데이터 조회
#     item = await store.aget(("users",), "user_123")
#     print(item.value)

print("PostgreSQL Store example (commented out - requires database setup)")

# %% [markdown]
# ## 모범 사례
#
# ### 1. 적절한 네임스페이스 구조
#
# 데이터를 논리적으로 구성하세요.

# %%
# 좋은 예: 계층적 구조
namespaces = {
    "users": ("users",),
    "user_profile": ("users", "user_id", "profile"),
    "user_preferences": ("users", "user_id", "preferences"),
    "user_activity": ("users", "user_id", "activity"),
}

# 나쁜 예: 평면 구조
# ("all_data",)

# %% [markdown]
# ### 2. 키 명명 규칙
#
# 일관된 명명 규칙을 사용하세요.

# %%
# 좋은 예: 설명적이고 일관된 키
keys = {
    "basic_info": "user_basic_info",
    "preferences": "user_preferences",
    "last_activity": "user_last_activity"
}

# 나쁜 예: 불명확한 키
# "data1", "info", "x"

# %% [markdown]
# ### 3. 데이터 검증
#
# 저장하기 전에 데이터를 검증하세요.

# %%
@tool
def save_validated_data(data: dict, runtime: ToolRuntime[Context]) -> str:
    """Save data with validation."""
    # 데이터 검증
    required_fields = ["name", "email"]
    
    for field in required_fields:
        if field not in data:
            return f"Error: Missing required field: {field}"
    
    # 이메일 형식 검증 (간단한 예)
    if "@" not in data["email"]:
        return "Error: Invalid email format"
    
    # 검증 통과 후 저장
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, data)
    
    return "Data saved successfully"


# %% [markdown]
# ### 4. 에러 처리
#
# Store 작업 시 적절한 에러 처리를 구현하세요.

# %%
@tool
def safe_get_data(key: str, runtime: ToolRuntime[Context]) -> str:
    """Safely retrieve data from store."""
    try:
        store = runtime.store
        user_id = runtime.context.user_id
        
        item = store.get(("users",), user_id)
        
        if item:
            return str(item.value)
        else:
            return "No data found for this user"
            
    except Exception as e:
        return f"Error retrieving data: {str(e)}"


# %% [markdown]
# ### 5. 정기적인 정리
#
# 오래된 데이터를 정기적으로 정리하는 전략을 수립하세요.

# %%
from datetime import datetime, timedelta

@tool
def cleanup_old_data(days: int, runtime: ToolRuntime[Context]) -> str:
    """Clean up data older than specified days."""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 활동 데이터에 타임스탬프 포함
    activity = store.get(("users", user_id, "activity"), "history")
    
    if activity:
        cutoff_date = datetime.now() - timedelta(days=days)
        # 오래된 항목 필터링 로직
        return f"Cleaned up data older than {days} days"
    
    return "No data to clean up"

# %% [markdown]
# ## 요약
#
# ### 장기 메모리의 핵심 개념
#
# 1. **Storage**: LangGraph Store를 사용하여 JSON 문서로 저장
# 2. **Organization**: 네임스페이스와 키로 계층적 구조 구성
# 3. **Access**: 도구에서 `runtime.store`를 통해 읽기/쓰기
# 4. **Persistence**: 세션 간 데이터 유지
#
# ### 주요 사용 사례
#
# - 사용자 프로필 및 선호도 저장
# - 과거 상호작용에서 학습
# - 개인화된 경험 제공
# - 컨텍스트 유지
#
# ### 프로덕션 체크리스트
#
# - ✅ DB 기반 Store 사용 (PostgreSQL 등)
# - ✅ 적절한 네임스페이스 구조 설계
# - ✅ 데이터 검증 구현
# - ✅ 에러 처리 추가
# - ✅ 정기적인 데이터 정리 전략 수립
#
# 장기 메모리는 에이전트가 사용자를 기억하고 시간이 지남에 따라 더 나은 서비스를 제공할 수 있게 합니다.
