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
# # 🧠 LangGraph 메모리 시스템 완벽 가이드
#
# ## 📚 개요
#
# AI 애플리케이션이 진정한 가치를 제공하려면 **메모리(Memory)**가 필수적입니다. LangGraph는 두 가지 강력한 메모리 시스템을 제공합니다:
#
# 1. **단기 메모리(Short-term Memory)**: 대화 세션 내에서 컨텍스트 유지
# 2. **장기 메모리(Long-term Memory)**: 세션을 넘어 사용자별 정보 저장
#
# ## 🎯 학습 목표
#
# 이 튜토리얼을 완료하면 다음을 마스터하게 됩니다:
#
# 1. **단기 메모리** 구현 - Checkpointer를 활용한 대화 지속성
# 2. **장기 메모리** 구축 - Store를 활용한 영구 데이터 저장
# 3. **메모리 관리** 전략 - 메시지 트리밍, 요약, 삭제
# 4. **시맨틱 검색** - 임베딩 기반 메모리 검색
# 5. **프로덕션 배포** - PostgreSQL, Redis 등 실제 환경 적용
#
# ## 🔑 핵심 개념 미리보기
#
# ```
# ┌─────────────────────────────────────────────────────────┐
# │                  LangGraph Memory System                  │
# ├──────────────────────────┬──────────────────────────────┤
# │    Short-term Memory      │      Long-term Memory         │
# ├──────────────────────────┼──────────────────────────────┤
# │ • Thread-level            │ • User-level                  │
# │ • Checkpointer            │ • Store                       │
# │ • Multi-turn chats        │ • Persistent data             │
# │ • Session context         │ • Cross-session               │
# └──────────────────────────┴──────────────────────────────┘
# ```
#
# ## 💡 중요 원칙
#
# > **"메모리는 AI 에이전트를 단순한 도구에서 지능적인 파트너로 변화시킵니다"**
# > 
# > _Memory transforms AI agents from simple tools to intelligent partners_

# %% [markdown]
# ## 환경 설정

# %%
# 필요한 패키지 임포트
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key")

print("✅ 환경 설정 완료!")

# %% [markdown]
# ---
#
# # Part 1: 단기 메모리 (Short-term Memory) 🎯
#
# ## 1.1 단기 메모리란?
#
# 단기 메모리는 **대화 세션 내에서** 컨텍스트를 유지하는 메커니즘입니다. 이는 다음과 같은 특징을 가집니다:
#
# - **Thread 기반**: 각 대화는 고유한 thread_id로 식별
# - **Checkpointer 사용**: 상태를 저장하고 복원
# - **Multi-turn 대화**: 이전 대화 내용을 기억
#
# ### 핵심 컴포넌트: Checkpointer
#
# Checkpointer는 그래프의 상태를 저장하고 복원하는 역할을 합니다. 개발 환경에서는 `InMemorySaver`를, 프로덕션에서는 데이터베이스 기반 Checkpointer를 사용합니다.

# %%
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Checkpointer 생성 - 메모리에 상태를 저장
checkpointer = InMemorySaver()


# 간단한 챗봇 그래프 생성
def call_model(state: MessagesState):
    """Call the LLM with the current messages"""
    # 현재 메시지로 LLM 호출
    response = llm.invoke(state["messages"])
    # 응답을 메시지 리스트에 추가
    return {"messages": response}


# 그래프 구성
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Checkpointer와 함께 컴파일 - 핵심!
graph = builder.compile(checkpointer=checkpointer)

print("✅ 단기 메모리가 활성화된 그래프 생성 완료!")

# %% [markdown]
# ## 1.2 단기 메모리 실습: Multi-turn 대화
#
# 이제 같은 thread에서 여러 번 대화를 나누며 봇이 이전 대화를 기억하는지 확인해봅시다.

# %%
# Thread ID 설정 - 대화 세션 식별자
config = {"configurable": {"thread_id": "conversation_1"}}  # 고유한 대화 식별자

# 첫 번째 메시지 - 자기소개
print("👤 User: 안녕! 나는 철수야")
result = graph.invoke(
    {"messages": [{"role": "user", "content": "안녕! 나는 철수야"}]},
    config,  # thread_id 전달
)
print(f"🤖 Bot: {result['messages'][-1].content}\n")

# 두 번째 메시지 - 이름 확인
print("👤 User: 내 이름이 뭐라고 했지?")
result = graph.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐라고 했지?"}]},
    config,  # 같은 thread_id 사용
)
print(f"🤖 Bot: {result['messages'][-1].content}\n")

# 다른 thread로 테스트 - 메모리 분리 확인
config_2 = {"configurable": {"thread_id": "conversation_2"}}

print("--- 새로운 대화 세션 ---")
print("👤 User: 내 이름이 뭐야?")
result = graph.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]},
    config_2,  # 다른 thread_id
)
print(f"🤖 Bot: {result['messages'][-1].content}")
print("\n💡 다른 thread에서는 이전 대화를 기억하지 못합니다!")

# %% [markdown]
# ## 1.3 프로덕션 환경: PostgreSQL Checkpointer
#
# 실제 서비스에서는 데이터베이스 기반 Checkpointer를 사용해야 합니다. 여기서는 PostgreSQL 예제를 보여드립니다.

# %%
# PostgreSQL Checkpointer 예제 (실제 실행 시 DB 연결 필요)
from typing import Dict, Any


def create_production_graph():
    """Create a production-ready graph with PostgreSQL checkpointer"""

    # 실제 환경에서는 아래 주석을 해제하고 사용
    from langgraph.checkpoint.postgres import PostgresSaver

    DB_URI = "postgresql://postgres:postgres@localhost:5432/mydb"

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # 첫 실행 시 테이블 생성
        # checkpointer.setup()

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)

        graph = builder.compile(checkpointer=checkpointer)
        return graph

    print("📝 프로덕션 환경 코드 예제:")
    print(
        """
    DB_URI = "postgresql://user:pass@host:port/db"
    
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
    """
    )
    return None


create_production_graph()

# %% [markdown]
# ## 1.4 Subgraph에서의 메모리
#
# 서브그래프를 사용할 때는 부모 그래프에만 Checkpointer를 설정하면 자동으로 전파됩니다.

# %%
from typing_extensions import TypedDict


class SubgraphState(TypedDict):
    """State for subgraph example"""

    message: str
    counter: int


# 서브그래프 생성
def create_subgraph():
    """Create a subgraph"""

    def subgraph_node(state: SubgraphState):
        # 카운터 증가
        return {
            "message": state["message"] + " (서브그래프 처리)",
            "counter": state["counter"] + 1,
        }

    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node("process", subgraph_node)
    subgraph_builder.add_edge(START, "process")
    subgraph_builder.add_edge("process", END)

    # 서브그래프는 checkpointer 없이 컴파일
    return subgraph_builder.compile()


# 부모 그래프 생성
def main_node(state: SubgraphState):
    """Main graph node"""
    return {
        "message": state["message"] + " (메인 처리)",
        "counter": state["counter"] + 10,
    }


# 서브그래프 생성
subgraph = create_subgraph()

# 메인 그래프 구성
main_builder = StateGraph(SubgraphState)
main_builder.add_node("main", main_node)
main_builder.add_node("subgraph", subgraph)  # 서브그래프를 노드로 추가

main_builder.add_edge(START, "main")
main_builder.add_edge("main", "subgraph")
main_builder.add_edge("subgraph", END)

# 부모 그래프만 checkpointer와 함께 컴파일
parent_checkpointer = InMemorySaver()
parent_graph = main_builder.compile(checkpointer=parent_checkpointer)

# 실행
result = parent_graph.invoke(
    {"message": "시작", "counter": 0}, {"configurable": {"thread_id": "sub_test"}}
)

print(f"✅ 서브그래프 실행 결과:")
print(f"  메시지: {result['message']}")
print(f"  카운터: {result['counter']}")
print("\n💡 부모 그래프의 checkpointer가 서브그래프에도 자동 적용됩니다!")

# %% [markdown]
# ---
#
# # Part 2: 장기 메모리 (Long-term Memory) 💾
#
# ## 2.1 장기 메모리란?
#
# 장기 메모리는 **세션을 넘어서** 사용자별 또는 애플리케이션 레벨의 데이터를 저장합니다:
#
# - **사용자 프로필**: 선호도, 설정, 개인 정보
# - **대화 히스토리**: 과거 상호작용 기록
# - **학습된 정보**: 시스템이 시간이 지남에 따라 학습한 내용
#
# ### 핵심 컴포넌트: Store
#
# Store는 key-value 형식으로 데이터를 영구 저장하는 시스템입니다.

# %%
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import uuid

# 메모리 스토어 생성
store = InMemoryStore()


# 사용자 정보를 저장하는 그래프
class UserState(MessagesState):
    """State with user context"""

    user_id: str


def chat_with_memory(
    state: UserState, config: RunnableConfig, *, store: BaseStore  # store 주입
):
    """Chat function with long-term memory"""

    # 사용자 ID 가져오기
    user_id = config["configurable"].get("user_id", "default_user")

    # 네임스페이스 정의 - 사용자별 메모리 분리
    namespace = ("users", user_id)

    # 마지막 메시지 확인
    last_message = state["messages"][-1]

    # "기억해" 키워드가 있으면 저장
    if "기억해" in last_message.content:
        # 메모리에 저장할 내용 추출
        memory_content = last_message.content.replace("기억해:", "").strip()
        # Store에 저장
        store.put(namespace, str(uuid.uuid4()), {"memory": memory_content})

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"알겠습니다. '{memory_content}'를 기억하겠습니다.",
                }
            ]
        }

    # "뭐 기억하고 있어?" 키워드가 있으면 조회
    elif "뭐 기억하고 있어" in last_message.content:
        # 저장된 메모리 검색
        memories = store.search(namespace, query="*")  # 모든 메모리 조회

        if memories:
            memory_list = [item.value["memory"] for item in memories]
            response = "제가 기억하고 있는 내용:\n" + "\n".join(
                f"• {m}" for m in memory_list
            )
        else:
            response = "아직 기억하고 있는 내용이 없습니다."

        return {"messages": [{"role": "assistant", "content": response}]}

    # 일반 대화
    else:
        # 기존 메모리를 컨텍스트로 사용
        memories = store.search(namespace, query="*")
        context = ""
        if memories:
            context = "\n".join([item.value["memory"] for item in memories])
            system_prompt = f"당신은 도움이 되는 어시스턴트입니다. 사용자에 대해 알고 있는 정보: {context}"
        else:
            system_prompt = "당신은 도움이 되는 어시스턴트입니다."

        # LLM 호출
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.invoke(messages)

        return {"messages": response}


# 그래프 구성
memory_builder = StateGraph(UserState)
memory_builder.add_node("chat", chat_with_memory)
memory_builder.add_edge(START, "chat")
memory_builder.add_edge("chat", END)

# Store와 Checkpointer 모두 사용
memory_graph = memory_builder.compile(
    checkpointer=InMemorySaver(), store=store  # 단기 메모리  # 장기 메모리
)

print("✅ 장기 메모리가 활성화된 그래프 생성 완료!")

# %% [markdown]
# ## 2.2 장기 메모리 실습: 세션 간 정보 유지

# %%
# 첫 번째 세션 - 정보 저장
config_session1 = {"configurable": {"thread_id": "session_1", "user_id": "user_123"}}

print("=== 세션 1: 정보 저장 ===")
print("\n👤 User: 기억해: 내 이름은 김철수이고 개발자야")
result = memory_graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "기억해: 내 이름은 김철수이고 개발자야"}
        ]
    },
    config_session1,
)
print(f"🤖 Bot: {result['messages'][-1].content}")

print("\n👤 User: 기억해: 나는 파이썬을 좋아해")
result = memory_graph.invoke(
    {"messages": [{"role": "user", "content": "기억해: 나는 파이썬을 좋아해"}]},
    config_session1,
)
print(f"🤖 Bot: {result['messages'][-1].content}")

# 두 번째 세션 - 다른 thread_id지만 같은 user_id
config_session2 = {
    "configurable": {
        "thread_id": "session_2",  # 다른 세션
        "user_id": "user_123",  # 같은 사용자
    }
}

print("\n=== 세션 2: 새로운 대화 (다른 thread_id) ===")
print("\n👤 User: 뭐 기억하고 있어?")
result = memory_graph.invoke(
    {"messages": [{"role": "user", "content": "뭐 기억하고 있어?"}]}, config_session2
)
print(f"🤖 Bot: {result['messages'][-1].content}")

print("\n💡 다른 세션에서도 사용자 정보를 기억합니다!")

# %% [markdown]
# ## 2.3 Tool에서 메모리 접근
#
# 에이전트의 도구(Tool)에서도 메모리에 접근할 수 있습니다.

# %%
from typing import Annotated
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool


# Tool에서 State 접근 예제
class AgentState(MessagesState):
    """State for agent with tools"""

    user_preference: str


@tool
def get_user_preference(
    state: Annotated[AgentState, InjectedState],  # State 주입
) -> str:
    """Get user preference from state"""
    preference = state.get("user_preference", "없음")
    return f"사용자 선호도: {preference}"


@tool
def update_user_preference(
    new_preference: str, state: Annotated[AgentState, InjectedState]
) -> str:
    """Update user preference in state"""
    # Tool에서 state 업데이트는 Command를 통해 수행
    return f"선호도를 '{new_preference}'로 업데이트했습니다."


# 도구 사용 예제
def agent_with_tools(state: AgentState):
    """Agent that can use tools"""
    # 여기서는 간단한 예제만 보여줌
    last_message = state["messages"][-1].content

    if "선호도" in last_message:
        # Tool 호출 시뮬레이션
        tool_result = get_user_preference.invoke({"state": state})
        return {"messages": [{"role": "assistant", "content": tool_result}]}

    return {"messages": [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]}


print("✅ Tool에서 메모리 접근 패턴 정의 완료!")
print("\n💡 InjectedState를 사용하여 Tool에서 state에 접근할 수 있습니다.")

# %% [markdown]
# ---
#
# # Part 3: 메모리 관리 전략 🔧
#
# ## 3.1 메시지 트리밍 (Trimming)
#
# 긴 대화에서 LLM의 컨텍스트 윈도우 제한을 관리하기 위해 메시지를 트리밍합니다.

# %%
from langchain_core.messages import trim_messages, HumanMessage, AIMessage


# 메시지 트리밍 예제
def demonstrate_trimming():
    """Demonstrate message trimming strategies"""

    # 긴 대화 히스토리 시뮬레이션
    messages = [
        HumanMessage(content="안녕하세요"),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
        HumanMessage(content="날씨가 어때요?"),
        AIMessage(content="오늘은 맑은 날씨입니다."),
        HumanMessage(content="추천 음식이 있나요?"),
        AIMessage(content="파스타를 추천합니다."),
        HumanMessage(content="레시피를 알려주세요"),
        AIMessage(content="토마토 파스타 레시피입니다..."),
        HumanMessage(content="감사합니다"),
        AIMessage(content="천만에요!"),
    ]

    print(f"원본 메시지 수: {len(messages)}\n")

    # 전략 1: 최근 N개 메시지만 유지
    trimmed_last = trim_messages(
        messages,
        strategy="last",
        max_tokens=100,  # 대략 100 토큰만 유지
        start_on="human",  # 사람 메시지로 시작
        end_on=("human", "ai"),  # 사람 또는 AI 메시지로 끝
    )

    print("전략 1 - 최근 메시지 유지:")
    for msg in trimmed_last:
        role = "User" if isinstance(msg, HumanMessage) else "Bot"
        print(f"  {role}: {msg.content[:30]}...")

    # 전략 2: 첫 메시지와 최근 메시지 유지
    trimmed_mixed = trim_messages(
        messages,
        strategy="first",
        max_tokens=100,
        include_system=False,
    )

    print("\n전략 2 - 첫 메시지 유지:")
    for msg in trimmed_mixed:
        role = "User" if isinstance(msg, HumanMessage) else "Bot"
        print(f"  {role}: {msg.content[:30]}...")

    return trimmed_last


trimmed = demonstrate_trimming()
print(f"\n✅ 트리밍 후 메시지 수: {len(trimmed)}")


# %%
# 그래프에서 트리밍 적용
class TrimmedState(MessagesState):
    """State with message trimming"""

    pass


def call_model_with_trimming(state: TrimmedState):
    """Call model with automatic message trimming"""

    # 메시지 트리밍 - 최대 500 토큰만 유지
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=500,
        start_on="human",
        end_on=("human", "ai"),
        include_system=True,  # 시스템 메시지 포함
    )

    # 트리밍된 메시지로 LLM 호출
    response = llm.invoke(trimmed_messages)

    return {"messages": [response]}


# 트리밍이 적용된 그래프 생성
trimming_builder = StateGraph(TrimmedState)
trimming_builder.add_node("chat", call_model_with_trimming)
trimming_builder.add_edge(START, "chat")
trimming_builder.add_edge("chat", END)

trimming_graph = trimming_builder.compile(checkpointer=InMemorySaver())

print("✅ 자동 트리밍이 적용된 그래프 생성 완료!")
print("\n💡 긴 대화에서도 컨텍스트 윈도우 제한을 자동으로 관리합니다.")

# %% [markdown]
# ## 3.2 메시지 삭제 (Deletion)
#
# 특정 메시지를 영구적으로 삭제하여 메모리를 관리할 수 있습니다.

# %%
from langchain_core.messages import RemoveMessage


class DeletionState(MessagesState):
    """State for message deletion example"""

    pass


def delete_old_messages(state: DeletionState):
    """Delete messages older than threshold"""
    messages = state["messages"]

    # 5개 이상의 메시지가 있으면 오래된 메시지 삭제
    if len(messages) > 5:
        # 처음 2개 메시지 삭제
        messages_to_delete = [RemoveMessage(id=msg.id) for msg in messages[:2]]
        return {"messages": messages_to_delete}

    return {}


def chat_and_cleanup(state: DeletionState):
    """Chat with automatic cleanup"""
    # 먼저 오래된 메시지 정리
    cleanup_result = delete_old_messages(state)

    # LLM 호출
    response = llm.invoke(state["messages"])

    # 응답과 정리 결과 병합
    messages_update = [response]
    if "messages" in cleanup_result:
        messages_update = cleanup_result["messages"] + messages_update

    return {"messages": messages_update}


# 삭제 로직이 포함된 그래프
deletion_builder = StateGraph(DeletionState)
deletion_builder.add_node("chat", chat_and_cleanup)
deletion_builder.add_edge(START, "chat")
deletion_builder.add_edge("chat", END)

deletion_graph = deletion_builder.compile(checkpointer=InMemorySaver())

print("✅ 자동 메시지 삭제가 적용된 그래프 생성 완료!")
print("\n💡 오래된 메시지를 자동으로 삭제하여 메모리를 효율적으로 관리합니다.")

# %% [markdown]
# ## 3.3 메시지 요약 (Summarization)
#
# 오래된 메시지를 요약하여 컨텍스트를 압축할 수 있습니다.

# %%
from langchain_core.messages import SystemMessage


class SummarizationState(MessagesState):
    """State with summarization support"""

    summary: str = ""  # 대화 요약 저장


def summarize_conversation(messages: list) -> str:
    """Summarize a list of messages"""
    # 요약을 위한 프롬프트
    summary_prompt = """
    Please summarize the following conversation in 2-3 sentences,
    focusing on key information and context:
    
    {conversation}
    """

    # 대화 내용 포맷팅
    conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

    # LLM으로 요약 생성 (실제로는 별도의 요약 모델 사용 권장)
    summary_response = llm.invoke(
        [
            {
                "role": "system",
                "content": summary_prompt.format(conversation=conversation),
            }
        ]
    )

    return summary_response.content


def chat_with_summarization(state: SummarizationState):
    """Chat with automatic summarization"""
    messages = state["messages"]

    # 10개 이상의 메시지가 있으면 요약
    if len(messages) > 10:
        # 처음 5개 메시지 요약
        messages_to_summarize = messages[:5]
        summary = summarize_conversation(messages_to_summarize)

        # 요약을 시스템 메시지로 추가하고 오래된 메시지 삭제
        new_messages = [
            SystemMessage(content=f"Previous conversation summary: {summary}")
        ] + messages[
            5:
        ]  # 요약된 메시지는 제거

        # 상태 업데이트
        return {"messages": new_messages, "summary": summary}

    # 일반 응답
    response = llm.invoke(messages)
    return {"messages": [response]}


print("✅ 대화 요약 기능 구현 완료!")
print("\n💡 긴 대화를 자동으로 요약하여 컨텍스트를 효율적으로 관리합니다.")

# %% [markdown]
# ---
#
# # Part 4: 시맨틱 검색 (Semantic Search) 🔍
#
# ## 4.1 임베딩 기반 메모리 검색
#
# Store에 시맨틱 검색을 활성화하면 의미적으로 유사한 메모리를 찾을 수 있습니다.

# %%
from langchain_openai import OpenAIEmbeddings

# 시맨틱 검색이 활성화된 Store 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_store = InMemoryStore(
    index={
        "embed": embeddings,  # 임베딩 모델
        "dims": 1536,  # 임베딩 차원
    }
)

# 메모리 저장
user_namespace = ("user_456", "memories")

# 다양한 메모리 저장
memories_to_store = [
    {"id": "1", "text": "나는 피자를 좋아해"},
    {"id": "2", "text": "내 직업은 데이터 사이언티스트야"},
    {"id": "3", "text": "파이썬과 머신러닝을 주로 다뤄"},
    {"id": "4", "text": "주말에는 등산을 즐겨해"},
    {"id": "5", "text": "커피보다 차를 선호해"},
]

for memory in memories_to_store:
    semantic_store.put(user_namespace, memory["id"], {"text": memory["text"]})

print("✅ 시맨틱 메모리 저장 완료!\n")

# 시맨틱 검색 테스트
queries = ["음식 취향", "프로그래밍", "여가 활동"]

print("🔍 시맨틱 검색 결과:\n")
for query in queries:
    # 쿼리와 유사한 메모리 검색
    results = semantic_store.search(
        user_namespace, query=query, limit=2  # 상위 2개 결과
    )

    print(f"Query: '{query}'")
    for item in results:
        print(f"  → {item.value['text']}")
    print()


# %% [markdown]
# ## 4.2 시맨틱 메모리를 활용한 대화 시스템

# %%
def chat_with_semantic_memory(state: MessagesState, *, store: BaseStore):
    """Chat function with semantic memory search"""

    # 사용자의 마지막 메시지
    user_message = state["messages"][-1].content

    # 관련 메모리 검색
    namespace = ("user_456", "memories")
    relevant_memories = store.search(namespace, query=user_message, limit=3)

    # 컨텍스트 구성
    context = ""
    if relevant_memories:
        context = "사용자에 대한 관련 정보:\n"
        context += "\n".join([f"• {item.value['text']}" for item in relevant_memories])

    # 시스템 프롬프트 구성
    system_prompt = f"""
    You are a helpful assistant with access to user's personal information.
    Use the following context to provide personalized responses:
    
    {context}
    
    Respond naturally and incorporate relevant information when appropriate.
    """

    # LLM 호출
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


# 시맨틱 메모리 그래프 생성
semantic_builder = StateGraph(MessagesState)
semantic_builder.add_node("chat", chat_with_semantic_memory)
semantic_builder.add_edge(START, "chat")
semantic_builder.add_edge("chat", END)

semantic_graph = semantic_builder.compile(store=semantic_store)

# 테스트
print("💬 시맨틱 메모리 기반 대화 테스트:\n")

test_messages = [
    "오늘 점심 뭐 먹을까?",
    "새로운 프로젝트를 시작하려고 하는데 조언 좀 해줘",
    "주말 계획이 있어?",
]

for msg in test_messages:
    print(f"👤 User: {msg}")
    result = semantic_graph.invoke({"messages": [{"role": "user", "content": msg}]})
    print(f"🤖 Bot: {result['messages'][-1].content}\n")

# %% [markdown]
# ---
#
# # Part 5: 프로덕션 배포 🚀
#
# ## 5.1 데이터베이스 Checkpointer 비교

# %%
# 프로덕션 Checkpointer 옵션
production_options = {
    "PostgreSQL": {
        "package": "langgraph-checkpoint-postgres",
        "class": "PostgresSaver",
        "connection": "postgresql://user:pass@host:port/db",
        "features": ["ACID 준수", "복잡한 쿼리 지원", "엔터프라이즈 표준"],
        "setup": "checkpointer.setup()  # 첫 실행 시",
    },
    "MongoDB": {
        "package": "langgraph-checkpoint-mongodb",
        "class": "MongoDBSaver",
        "connection": "mongodb://localhost:27017",
        "features": ["NoSQL 유연성", "수평 확장성", "JSON 네이티브"],
        "setup": "클러스터 생성 필요",
    },
    "Redis": {
        "package": "langgraph-checkpoint-redis",
        "class": "RedisSaver",
        "connection": "redis://localhost:6379",
        "features": ["초고속 메모리 DB", "캐싱 최적화", "Pub/Sub 지원"],
        "setup": "checkpointer.setup()  # 첫 실행 시",
    },
}

print("📊 프로덕션 Checkpointer 비교:\n")
for db, info in production_options.items():
    print(f"### {db}")
    print(f"  패키지: {info['package']}")
    print(f"  클래스: {info['class']}")
    print(f"  연결: {info['connection']}")
    print(f"  특징: {', '.join(info['features'])}")
    print(f"  설정: {info['setup']}\n")


# %% [markdown]
# ## 5.2 프로덕션 Store 구현 예제

# %%
def create_production_setup():
    """Production setup example with both checkpointer and store"""

    print("🏭 프로덕션 환경 설정 예제:\n")

    # PostgreSQL 예제
    postgres_example = """
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://user:password@localhost:5432/langgraph_db"

# Context manager로 자동 리소스 관리
with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # 첫 실행 시 테이블 생성
    store.setup()
    checkpointer.setup()
    
    # 그래프 컴파일
    graph = builder.compile(
        checkpointer=checkpointer,  # 단기 메모리
        store=store  # 장기 메모리
    )
    
    # 그래프 실행
    result = graph.invoke(input_data, config)
    """

    print("### PostgreSQL 설정:")
    print(postgres_example)

    # Redis 예제
    redis_example = """
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

REDIS_URI = "redis://localhost:6379"

# Redis 연결
with (
    RedisStore.from_conn_string(REDIS_URI) as store,
    RedisSaver.from_conn_string(REDIS_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()
    
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store
    )
    """

    print("\n### Redis 설정:")
    print(redis_example)

    return "프로덕션 설정 예제 완료"


create_production_setup()

# %% [markdown]
# ## 5.3 베스트 프랙티스

# %%
# 메모리 관리 베스트 프랙티스
best_practices = {
    "아키텍처 설계": [
        "단기/장기 메모리 명확히 구분",
        "적절한 네임스페이스 전략 수립",
        "메모리 계층 구조 설계",
    ],
    "성능 최적화": [
        "메시지 트리밍으로 컨텍스트 관리",
        "캐싱 적극 활용",
        "인덱싱으로 검색 성능 향상",
    ],
    "데이터 관리": ["정기적인 메모리 정리", "백업 전략 수립", "민감 정보 암호화"],
    "모니터링": ["메모리 사용량 추적", "응답 시간 모니터링", "에러 로깅 및 알림"],
    "확장성": ["수평 확장 가능한 아키텍처", "로드 밸런싱 고려", "샤딩 전략 수립"],
}

print("📋 메모리 관리 베스트 프랙티스:\n")
for category, practices in best_practices.items():
    print(f"### {category}")
    for practice in practices:
        print(f"  ✓ {practice}")
    print()
