# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: langchain-kr-lwwSZlnu-py3.11
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 대화 기록 요약을 추가하는 방법
#
# ![](image/langgraph-10.jpeg)
#
# 대화 기록을 유지하는 것은 **지속성**의 가장 일반적인 사용 사례 중 하나입니다. 이는 대화를 지속하기 쉽게 만들어주는 장점이 있습니다. 
#
# 하지만 대화가 길어질수록 대화 기록이 누적되어 `context window`를 더 많이 차지하게 됩니다. 이는 `LLM` 호출이 더 비싸고 길어지며, 잠재적으로 오류가 발생할 수 있어 바람직하지 않을 수 있습니다. 이를 해결하기 위한 한 가지 방법은 현재까지의 대화 요약본을 생성하고, 이를 최근 `N` 개의 메시지와 함께 사용하는 것입니다. 
#
# 이 가이드에서는 이를 구현하는 방법의 예시를 살펴보겠습니다.
#
# 다음과 같은 단계가 필요합니다.
#
# - 대화가 너무 긴지 확인 (메시지 수나 메시지 길이로 확인 가능)
# - 너무 길다면 요약본 생성 (이를 위한 프롬프트 필요)
# - 마지막 `N` 개의 메시지를 제외한 나머지 삭제
#
# 이 과정에서 중요한 부분은 오래된 메시지를 삭제(`DeleteMessage`) 하는 것입니다. 
#

# %% [markdown]
# ## 환경 설정
#
# 대화 요약 기능을 구현하기 위한 환경을 설정합니다. 환경 변수를 로드하고 LangSmith 추적을 활성화합니다.
#
# 아래 코드에서는 환경 설정을 수행합니다.

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# # !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-V1-Tutorial")

# %% [markdown]
# ## State 정의 및 기본 설정
#
# 긴 대화에 대하여 요약본을 생성한 뒤, 기존의 대화를 삭제하고 요약본을 대화로 저장하는 로직을 구현합니다. State에 `summary` 필드를 추가하여 대화 요약을 저장합니다.
#
# **조건**: 대화의 길이가 6개 초과일 경우 요약본을 생성합니다.
#
# 아래 코드에서는 State 클래스와 LLM 모델을 정의합니다.

# %%
from typing import Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.graph.message import add_messages

# 메모리 저장소 설정
memory = MemorySaver()


# 메시지 상태와 요약 정보를 포함하는 상태 클래스
class State(MessagesState):
    messages: Annotated[list, add_messages]
    summary: str


# 대화 및 요약을 위한 모델 초기화
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# %% [markdown]
# ### ask_llm 노드 정의
#
# `ask_llm` 노드는 `messages`를 LLM에 주입하여 답변을 생성합니다. 이전의 대화 요약본(`summary`)이 존재한다면 시스템 메시지로 추가하여 LLM이 이전 대화 맥락을 이해할 수 있도록 합니다. 요약본이 없다면 현재 메시지만 사용합니다.
#
# 아래 코드에서는 `ask_llm` 노드 함수를 정의합니다.

# %%
def ask_llm(state: State):
    # 이전 요약 정보 확인
    summary = state.get("summary", "")

    # 이전 요약 정보가 있다면 시스템 메시지로 추가
    if summary:
        # 시스템 메시지 생성
        system_message = f"Summary of conversation earlier: {summary}"
        # 시스템 메시지와 이전 메시지 결합
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        # 이전 메시지만 사용
        messages = state["messages"]

    # 모델 호출
    response = model.invoke(messages)

    # 응답 반환
    return {"messages": [response]}


# %% [markdown]
# ### should_continue 조건 함수 정의
#
# `should_continue` 함수는 대화의 길이를 확인하여 다음 노드를 결정합니다. 메시지가 6개 초과일 경우 `summarize_conversation` 노드로 이동하여 요약을 생성하고, 그렇지 않으면 `END` 노드로 이동하여 대화를 종료합니다.
#
# 아래 코드에서는 조건부 라우팅 함수를 정의합니다.

# %%
from langgraph.graph import END


# 대화 종료 또는 요약 결정 로직
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    # 메시지 목록 확인
    messages = state["messages"]

    # 메시지 수가 6개 초과라면 요약 노드로 이동
    if len(messages) > 6:
        return "summarize_conversation"
    return END


# %% [markdown]
# ### summarize_conversation 노드 정의
#
# `summarize_conversation` 노드는 현재까지의 대화 내용을 LLM을 사용하여 요약합니다. 기존 요약이 있다면 새로운 메시지를 반영하여 확장하고, 없다면 새로운 요약을 생성합니다. 요약 후에는 `RemoveMessage`를 사용하여 마지막 2개를 제외한 오래된 메시지를 삭제합니다.
#
# 아래 코드에서는 대화 요약 노드 함수를 정의합니다.

# %%
# 대화 내용 요약 및 메시지 정리 로직
def summarize_conversation(state: State):
    # 이전 요약 정보 확인
    summary = state.get("summary", "")

    # 이전 요약 정보가 있다면 요약 메시지 생성
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above in Korean:"
        )
    else:
        # 요약 메시지 생성
        summary_message = "Create a summary of the conversation above in Korean:"

    # 요약 메시지와 이전 메시지 결합
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    # 모델 호출
    response = model.invoke(messages)
    # 오래된 메시지 삭제
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    # 요약 정보 반환
    return {"summary": response.content, "messages": delete_messages}


# %%
# 워크플로우 그래프 초기화
workflow = StateGraph(State)

# 대화 및 요약 노드 추가
workflow.add_node("conversation", ask_llm)
workflow.add_node(summarize_conversation)

# 시작점을 대화 노드로 설정
workflow.add_edge(START, "conversation")

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "conversation",
    should_continue,
)

# 요약 노드에서 종료 노드로의 엣지 추가
workflow.add_edge("summarize_conversation", END)

# 워크플로우 컴파일 및 메모리 체크포인터 설정
app = workflow.compile(checkpointer=memory)

# %% [markdown]
# ### 그래프 시각화
#
# 컴파일된 그래프를 시각화하여 `conversation` 노드에서 조건에 따라 `summarize_conversation` 또는 `END`로 분기하는 구조를 확인합니다.
#
# 아래 코드는 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(app)


# %% [markdown]
# ## 그래프 실행 테스트
#
# 이제 구현한 그래프를 테스트합니다. 여러 메시지를 전송하여 대화가 6개를 초과할 때 요약이 생성되는지 확인합니다. 먼저 업데이트 정보를 출력하는 헬퍼 함수를 정의합니다.
#
# 아래 코드에서는 출력 헬퍼 함수와 테스트 대화를 실행합니다.

# %%
# 업데이트 정보 출력 함수
def print_update(update):
    # 업데이트 딕셔너리 순회
    for k, v in update.items():
        # 메시지 목록 출력
        for m in v["messages"]:
            m.pretty_print()
        # 요약 정보 존재 시 출력
        if "summary" in v:
            print(v["summary"])


# %%
# 메시지 핸들링을 위한 HumanMessage 클래스 임포트
from langchain_core.messages import HumanMessage

# 스레드 ID가 포함된 설정 객체 초기화
config = {"configurable": {"thread_id": "1"}}

# 첫 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="안녕하세요? 반갑습니다. 제 이름은 테디입니다.")
input_message.pretty_print()

# 스트림 모드에서 첫 번째 메시지 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 두 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="제 이름이 뭔지 기억하세요?")
input_message.pretty_print()

# 스트림 모드에서 두 번째 메시지 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 세 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="제 직업은 AI 연구원이에요")
input_message.pretty_print()

# 스트림 모드에서 세 번째 메시지 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# %% [markdown]
# ### 상태 확인 (요약 전)
#
# 현재 상태를 확인해보면 요약이 아직 이루어지지 않은 것을 볼 수 있습니다. 메시지가 6개(HumanMessage 3개 + AIMessage 3개)뿐이므로 요약 조건(6개 초과)을 충족하지 않기 때문입니다.
#
# 아래 코드에서는 현재 상태를 조회합니다.

# %%
# 상태 구성 값 검색
values = app.get_state(config).values
values

# %% [markdown]
# ### 추가 메시지 전송 (요약 트리거)
#
# 메시지 수가 6개를 초과하도록 새로운 메시지를 전송합니다. 이번에는 요약 조건이 충족되어 `summarize_conversation` 노드가 실행됩니다.
#
# 아래 코드에서는 네 번째 메시지를 전송합니다.

# %%
# 사용자 입력 메시지 객체 생성
input_message = HumanMessage(
    content="최근 LLM 에 대해 좀 더 알아보고 있어요. LLM 에 대한 최근 논문을 읽고 있습니다."
)

# 메시지 내용 출력
input_message.pretty_print()

# 스트림 이벤트 실시간 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# %% [markdown]
# ### 상태 확인 (요약 후)
#
# 요약이 완료된 후 상태를 확인하면 `summary` 필드에 대화 요약이 저장되어 있고, `messages`에는 마지막 두 개의 메시지만 남아있는 것을 볼 수 있습니다. 오래된 메시지는 `RemoveMessage`로 삭제되었습니다.
#
# 아래 코드에서는 요약 후 상태를 조회합니다.

# %%
# 상태 구성 값 검색
values = app.get_state(config).values
values

# %%
messages = values["messages"]
messages

# %% [markdown]
# ### 요약 기반 대화 재개
#
# 이제 대화를 재개하여 요약 기능이 제대로 작동하는지 확인합니다. 메시지 목록에는 마지막 두 개의 메시지만 있지만, 이전 대화 내용에 대해 질문해도 정확히 응답할 수 있습니다. 이는 요약된 내용이 시스템 메시지로 LLM에 전달되기 때문입니다.
#
# 아래 코드에서는 이전 대화 내용에 대해 질문합니다.

# %%
# 사용자 메시지 객체 생성
input_message = HumanMessage(content="제 이름이 무엇인지 기억하세요?")

# 메시지 내용 출력
input_message.pretty_print()

# 스트림 이벤트 실시간 처리 및 업데이트
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# %%
# 사용자 메시지 객체 생성
input_message = HumanMessage(content="제 직업도 혹시 기억하고 계세요?")

# 메시지 내용 출력
input_message.pretty_print()

# 스트림 이벤트 실시간 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
