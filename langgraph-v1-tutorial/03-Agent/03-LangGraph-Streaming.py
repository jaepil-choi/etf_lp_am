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
# # 메시지 스트리밍
#
# 스트리밍을 사용하면 LLM의 출력이 생성되는 대로 점진적으로 표시할 수 있어 더 나은 사용자 경험을 제공합니다. 사용자는 전체 응답이 완료될 때까지 기다리지 않고 실시간으로 결과를 확인할 수 있습니다.
#
# LangChain 에이전트는 기본적으로 스트리밍 모드로 실행되며, 실시간으로 응답을 제공합니다.
#
# **스트리밍 모드 종류:**
#
# | 모드 | 설명 |
# |:---|:---|
# | **updates** | 에이전트의 진행 상황 (기본값) - 각 노드 실행 완료 시 업데이트 |
# | **messages** | LLM의 토큰 스트리밍 - 실시간 텍스트 출력 |
# | **custom** | 커스텀 업데이트 - 도구에서 전송하는 사용자 정의 데이터 |
#
# > 참고 문서: [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming.md)

# %% [markdown]
# ## 환경 설정
#
# 스트리밍 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드하고, `langchain_teddynote`의 로깅 기능을 활성화하여 LangSmith에서 스트리밍 과정을 추적할 수 있도록 합니다.
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
# ---
#
# ## 에이전트 진행 상황 스트리밍
#
# `stream_mode="updates"`는 에이전트의 진행 상황을 추적하는 기본 스트리밍 모드입니다. 각 노드가 실행을 완료할 때마다 업데이트를 생성하며, 노드 이름과 해당 노드의 출력 메시지가 포함됩니다.
#
# 이 모드는 에이전트가 어떤 단계를 거치고 있는지 모니터링하거나, 로깅 시스템에 진행 상황을 기록할 때 유용합니다.
#
# 아래 코드는 updates 모드로 에이전트를 스트리밍하는 예시입니다.

# %%
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    return f"The weather in {city} is sunny!"

model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(model=model, tools=[get_weather])

# 기본 스트리밍 (updates 모드)
for chunk in agent.stream({"messages": [{"role": "user", "content": "What's the weather in Seoul?"}]}):
    print(chunk)
    print("---")

# %% [markdown]
# ---
#
# ## LLM 토큰 스트리밍
#
# `stream_mode="messages"`를 사용하면 LLM이 생성하는 토큰을 실시간으로 스트리밍할 수 있습니다. 이는 사용자에게 즉각적인 피드백을 제공하는 데 유용하며, 채팅 인터페이스에서 타자기 효과를 구현할 때 자주 사용됩니다.
#
# 각 청크는 `AIMessageChunk` 객체로 전달되며, `content` 속성에 토큰 텍스트가 포함됩니다.
#
# 아래 코드는 messages 모드로 LLM 토큰을 스트리밍하는 예시입니다.

# %%
# messages 스트리밍 모드
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me a short story about a robot."}]},
    stream_mode="messages"  # LLM 토큰 스트리밍
):
    print(chunk, end="", flush=True)

# %% [markdown]
# ### 실용적인 예제: 타자기 효과
#
# LLM 토큰을 스트리밍하여 타자기처럼 텍스트를 출력하는 실용적인 예제입니다. `time.sleep()`을 사용하여 각 토큰 사이에 약간의 지연을 추가하면 더 자연스러운 타이핑 효과를 연출할 수 있습니다.
#
# 아래 코드는 타자기 효과를 구현하는 예시입니다.

# %%
import sys
import time

print("AI: ", end="", flush=True)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Write a haiku about technology."}]},
    stream_mode="messages"
):
    # AIMessageChunk에서 텍스트 추출
    if hasattr(chunk, 'content'):
        print(chunk.content, end="", flush=True)
        time.sleep(0.02)  # 타자기 효과를 위한 약간의 지연

print()  # 줄바꿈

# %% [markdown]
# ---
#
# ## 커스텀 업데이트
#
# `runtime.stream_writer` 또는 `runtime.get_stream_writer()`를 사용하면 에이전트 실행 중 커스텀 업데이트를 스트리밍할 수 있습니다. 이는 장시간 실행되는 도구에서 진행 상황, 중간 결과 또는 디버그 정보를 전송하는 데 유용합니다.
#
# 커스텀 업데이트는 `stream_mode="custom"`으로 수신할 수 있으며, 도구에서 전송한 데이터 형식 그대로 전달됩니다.
#
# 아래 코드는 커스텀 업데이트를 스트리밍하는 도구 예시입니다.

# %%
from langchain.tools import tool, ToolRuntime

@tool
def process_data(data_size: int, runtime: ToolRuntime) -> str:
    """Process data with progress updates."""
    writer = runtime.get_stream_writer()

    # 진행 상황을 커스텀 업데이트로 전송
    for i in range(0, data_size, 10):
        progress = min(i + 10, data_size)
        writer({"progress": progress, "total": data_size})

    return f"Processed {data_size} items successfully!"

agent = create_agent(model=model, tools=[process_data])

# 커스텀 스트리밍 모드로 진행 상황 추적
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Process 50 items of data"}]},
    stream_mode="custom"  # 커스텀 업데이트 수신
):
    if "progress" in chunk:
        percentage = (chunk["progress"] / chunk["total"]) * 100
        print(f"Progress: {chunk['progress']}/{chunk['total']} ({percentage:.0f}%)")

# %% [markdown]
# ---
#
# ## 다중 스트리밍 모드
#
# 여러 스트리밍 모드를 동시에 사용할 수 있습니다. `stream_mode`에 리스트로 전달하면 각 모드의 업데이트를 모두 받을 수 있으며, 반환되는 청크는 `(stream_mode, data)` 튜플 형태입니다.
#
# 이 방식은 진행 상황(updates)과 세부 작업 정보(custom)를 동시에 추적해야 할 때 유용합니다.
#
# 아래 코드는 여러 스트리밍 모드를 동시에 사용하는 예시입니다.

# %%
# 여러 스트리밍 모드 동시 사용
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Process 30 items"}]},
    stream_mode=["updates", "custom"]  # 여러 모드 동시 사용
):
    # chunk는 (stream_mode, data) 튜플 형태
    mode, data = chunk

    if mode == "updates":
        print(f"[UPDATE] Node completed: {list(data.keys())}")
    elif mode == "custom":
        if "progress" in data:
            print(f"[PROGRESS] {data['progress']}/{data['total']}")

# %% [markdown]
# ---
#
# ## 스트리밍 비활성화
#
# 개별 모델 또는 도구에 대해 스트리밍을 비활성화하려면 해당 객체를 생성할 때 `streaming=False`를 설정합니다. 이 경우 `messages` 모드를 사용해도 토큰이 실시간으로 스트리밍되지 않고 전체 응답이 한 번에 전달됩니다.
#
# 스트리밍 비활성화는 응답 전체가 필요한 후처리 작업이나, 네트워크 오버헤드를 줄이고 싶을 때 유용합니다.
#
# 아래 코드는 스트리밍을 비활성화한 모델 예시입니다.

# %%
from langchain_openai import ChatOpenAI

# 스트리밍 비활성화된 모델
non_streaming_model = ChatOpenAI(model="gpt-4.1-mini", streaming=False)

agent = create_agent(model=non_streaming_model, tools=[get_weather])

# messages 모드를 사용해도 토큰이 스트리밍되지 않음
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]},
    stream_mode="messages"
):
    # 전체 응답이 한 번에 전달됨
    print(chunk)

# %% [markdown]
# ---
#
# ## 종합 예제: 진행률 바가 있는 데이터 처리
#
# 여러 스트리밍 기능을 결합한 실용적인 예제입니다. 데이터 분석 도구에서 단계별 진행 상황을 커스텀 업데이트로 전송하고, 다중 스트리밍 모드로 진행 상황과 최종 결과를 모두 수신합니다.
#
# 이 패턴은 대시보드나 모니터링 시스템에서 장시간 실행 작업의 상태를 실시간으로 표시할 때 유용합니다.
#
# 아래 코드는 진행률 보고 기능이 포함된 데이터 분석 에이전트 예시입니다.

# %%
import time
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

@tool
def analyze_data(dataset_name: str, num_records: int, runtime: ToolRuntime) -> str:
    """Analyze a dataset with detailed progress reporting."""
    writer = runtime.get_stream_writer()

    # 단계별 분석 프로세스
    steps = [
        ("loading", "Loading data", 0.2),
        ("cleaning", "Cleaning data", 0.3),
        ("processing", "Processing data", 0.3),
        ("finalizing", "Finalizing results", 0.2)
    ]

    for step_name, step_desc, duration in steps:
        writer({
            "step": step_name,
            "description": step_desc,
            "status": "started"
        })

        time.sleep(duration)  # 작업 시뮬레이션

        writer({
            "step": step_name,
            "description": step_desc,
            "status": "completed"
        })

    return f"Successfully analyzed {num_records} records from {dataset_name}!"

@tool
def get_summary(analysis_result: str, runtime: ToolRuntime) -> str:
    """Generate a summary of the analysis."""
    return f"Analysis complete. {analysis_result}"

model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(model=model, tools=[analyze_data, get_summary])

print("Starting data analysis...\n")

# 다중 스트리밍 모드로 실행
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Analyze the sales dataset with 1000 records"}]},
    stream_mode=["custom", "updates"]
):
    mode, data = chunk

    if mode == "custom":
        # 커스텀 진행 상황 표시
        if "step" in data:
            status_icon = "✓" if data["status"] == "completed" else "→"
            print(f"{status_icon} {data['description']}... {data['status']}")

    elif mode == "updates":
        # 노드 완료 정보 (선택적으로 표시)
        if "messages" in data:
            last_msg = data["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"\n[Result] {last_msg.content}")

print("\nAnalysis finished!")


# %% [markdown]
# ---
#
# ## 실전 팁
#
# ### 적절한 스트리밍 모드 선택
#
# 상황에 따라 적절한 스트리밍 모드를 선택하면 더 나은 사용자 경험과 성능을 얻을 수 있습니다.
#
# | 상황 | 권장 모드 | 설명 |
# |:---|:---|:---|
# | 채팅 인터페이스 | messages | 실시간 텍스트 출력으로 자연스러운 대화 경험 |
# | 백엔드 작업 모니터링 | updates + custom | 노드 진행과 세부 작업 상태 동시 추적 |
# | 디버깅/로깅 | updates | 에이전트 실행 흐름 분석 |
#
# 아래 코드는 각 상황에 적합한 스트리밍 설정 예시입니다.

# %%
# 채팅 인터페이스에 적합한 설정
def chat_interface():
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Hello!"}]},
        stream_mode="messages"  # 실시간 응답 표시
    ):
        if hasattr(chunk, 'content'):
            yield chunk.content

# 백그라운드 작업 모니터링에 적합한 설정
def background_task():
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Process data"}]},
        stream_mode=["updates", "custom"]  # 진행 상황 추적
    ):
        mode, data = chunk
        # 진행 상황을 데이터베이스나 로그에 기록
        pass


# %% [markdown]
# ### 에러 처리
#
# 스트리밍 중 네트워크 오류, API 제한 초과 등 예외가 발생할 수 있으므로 적절한 에러 처리가 중요합니다. try-except 블록으로 스트리밍 루프를 감싸면 오류 발생 시에도 애플리케이션이 안정적으로 동작합니다.
#
# 아래 코드는 스트리밍 에러 처리 예시입니다.

# %%
try:
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Test query"}]},
        stream_mode="messages"
    ):
        print(chunk, end="", flush=True)
except Exception as e:
    print(f"\nError during streaming: {e}")

# %% [markdown]
# ### 성능 최적화
#
# 불필요한 스트리밍 모드를 사용하지 않으면 네트워크 오버헤드가 줄어들고 성능이 향상됩니다. 필요한 모드만 선택적으로 사용하는 것이 좋습니다.
#
# 아래 코드는 성능 최적화를 위한 스트리밍 설정 예시입니다.

# %%
# 좋은 예: 필요한 모드만 사용
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Query"}]},
    stream_mode="messages"  # 필요한 모드만
):
    pass

# 나쁜 예: 모든 모드 사용 (불필요한 오버헤드)
# for chunk in agent.stream(
#     {"messages": [{"role": "user", "content": "Query"}]},
#     stream_mode=["updates", "messages", "custom"]
# ):
#     pass

# %% [markdown]
# ---
#
# ## 정리
#
# 이 튜토리얼에서는 LangGraph 에이전트의 스트리밍 기능을 학습했습니다.
#
# **핵심 개념 요약:**
#
# | 스트리밍 모드 | 사용 시점 | 반환 데이터 |
# |:---|:---|:---|
# | **updates** | 노드 실행 진행 상황 추적 | 노드별 출력 메시지 |
# | **messages** | 실시간 텍스트 출력 (채팅 UI) | AIMessageChunk 토큰 |
# | **custom** | 도구의 세부 진행 상황 | 사용자 정의 데이터 |
#
# **주요 기능:**
# - `stream_mode` 매개변수로 스트리밍 모드 선택
# - 리스트로 다중 모드 동시 사용 가능
# - `runtime.stream_writer`로 커스텀 업데이트 전송
# - `streaming=False`로 개별 모델의 스트리밍 비활성화
#
# **다음 단계:**
# - Runtime 컨텍스트를 활용한 고급 도구 구현 학습
# - 구조화된 출력(Structured Output) 사용법 학습
