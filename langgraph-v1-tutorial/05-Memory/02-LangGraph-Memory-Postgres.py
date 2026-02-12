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

# %%
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from datetime import datetime
import uuid
import os

# 환경변수 설정
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# 데이터베이스 연결 문자열 생성
DB_URI = f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode=require"

# 쓰기 설정
write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": "test1"}}

# 읽기 설정
read_config = {"configurable": {"thread_id": "1"}}

# 데이터베이스 연결 설정
with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    checkpointer.setup()
    store.setup()

    checkpoint = {
        "v": 4,
        "ts": datetime.now().isoformat(),
        "id": str(uuid.uuid4()),
        "channel_values": {"my_key": "teddy", "node": "node"},
        "channel_versions": {"__start__": 2, "my_key": 3, "start:node": 3, "node": 3},
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
            "node": {"start:node": 2},
        },
    }

    checkpointer.put(
        write_config,
        checkpoint,
        {"step": -1, "source": "input", "parents": {}, "user_id": "1"},
        {},
    )

    # 목록 조회
    print(list(checkpointer.list(read_config)))

# %%
from langchain_teddynote.memory import create_memory_extractor

memory_extractor = create_memory_extractor(model="gpt-4.1")

# %%
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing import Any
import uuid
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1", temperature=0)


def call_model(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict[str, Any]:
    """Call the LLM model and manage user memory.

    Args:
        state (MessagesState): The current state containing messages.
        config (RunnableConfig): The runnable configuration.
        store (BaseStore): The memory store.
    """
    # 마지막 메시지에서 user_id 추출
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # 유저의 메모리 검색
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([f"{memory.key}: {memory.value}" for memory in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # 사용자가 기억 요청 시 메모리 저장
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        result = memory_extractor.invoke({"input": str(state["messages"][-1].content)})
        for memory in result.memories:
            print(memory)
            print("-" * 100)
            store.put(namespace, str(uuid.uuid4()), {memory.key: memory.value})

    # LLM 호출
    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


# %%
from langchain_teddynote.messages import stream_graph


with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # 그래프 생성
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")

    # 그래프 컴파일
    graph_with_memory = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    def run_graph(
        msg,
        thread_id,
        user_id,
    ):
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }
        }
        print(f"\n[User🙋] {msg}")
        stream_graph(
            graph_with_memory,
            inputs={"messages": [{"role": "user", "content": msg}]},
            config=config,
        )
        print()

    run_graph("내 이름이 뭐라고?", "1", "someone")

    run_graph("내 이름이 뭐라고?", "2", "someone")

    run_graph("내 이름은 테디야 remember", "3", "someone")

    run_graph("내 이름이 뭐라고?", "100", "someone")

# %%
