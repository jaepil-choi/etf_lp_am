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
# # Agent 에 메모리(memory) 추가
#
# 현재 챗봇은 과거 상호작용을 스스로 기억할 수 없어 일관된 다중 턴 대화를 진행하는 데 제한이 있습니다. 
#
# 이번 튜토리얼에서는 이를 해결하기 위해 **memory** 를 추가합니다.
#
# **참고**
#
# 이번에는 pre-built 되어있는 `ToolNode` 와 `tools_condition` 을 활용합니다.
#
# 1. [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode): 도구 호출을 위한 노드
# 2. [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.tools_condition): 도구 호출 여부에 따른 조건 분기

# %% [markdown]
#
# 우리의 챗봇은 이제 도구를 사용하여 사용자 질문에 답할 수 있지만, 이전 상호작용의 **context**를 기억하지 못합니다. 이는 멀티턴(multi-turn) 대화를 진행하는 능력을 제한합니다.
#
# `LangGraph`는 **persistent checkpointing** 을 통해 이 문제를 해결합니다. 
#
# 그래프를 컴파일할 때 `checkpointer`를 제공하고 그래프를 호출할 때 `thread_id`를 제공하면, `LangGraph`는 각 단계 후 **상태를 자동으로 저장** 합니다. 동일한 `thread_id`를 사용하여 그래프를 다시 호출하면, 그래프는 저장된 상태를 로드하여 챗봇이 이전에 중단한 지점에서 대화를 이어갈 수 있게 합니다.
#
# **checkpointing** 는 LangChain 의 메모리 기능보다 훨씬 강력합니다. (아마 이 튜토리얼을 완수하면 자연스럽게 이를 확인할 수 있습니다.)

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
# ## MemorySaver 체크포인터 생성
#
# 멀티턴(multi-turn) 대화를 가능하게 하기 위해 **checkpointing**을 추가합니다. 체크포인터는 그래프의 각 단계에서 상태를 저장하여, 이후 동일한 대화를 이어서 진행할 수 있게 합니다.
#
# `MemorySaver`는 인메모리 체크포인터로, 개발 및 테스트 환경에서 사용하기 적합합니다. 프로덕션 환경에서는 `SqliteSaver`나 `PostgresSaver` 등의 영구 저장소를 사용하는 것이 좋습니다.
#
# > 참고 문서: [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
#
# 아래 코드에서는 `MemorySaver` 체크포인터를 생성합니다.

# %%
from langgraph.checkpoint.memory import MemorySaver

# 메모리 저장소 생성
memory = MemorySaver()

# %% [markdown]
# **참고**
#
# 이번 튜토리얼에서는 `in-memory checkpointer` 를 사용합니다. 
#
# 하지만, 프로덕션 단계에서는 이를 `SqliteSaver` 또는 `PostgresSaver` 로 변경하고 자체 DB에 연결할 수 있습니다. 

# %%
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools.tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


########## 1. 상태 정의 ##########
# 상태 정의
class State(TypedDict):
    # 메시지 목록 주석 추가
    messages: Annotated[list, add_messages]


########## 2. 도구 정의 및 바인딩 ##########
# 도구 초기화
tool = TavilySearch(max_results=3)
tools = [tool]

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# 도구와 LLM 결합
llm_with_tools = llm.bind_tools(tools)


########## 3. 노드 추가 ##########
# 챗봇 함수 정의
def chatbot(state: State):
    # 메시지 호출 및 반환
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 상태 그래프 생성
graph_builder = StateGraph(State)

# 챗봇 노드 추가
graph_builder.add_node("chatbot", chatbot)

# 도구 노드 생성 및 추가
tool_node = ToolNode(tools=[tool])

# 도구 노드 추가
graph_builder.add_node("tools", tool_node)

# 조건부 엣지
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

########## 4. 엣지 추가 ##########

# tools > chatbot
graph_builder.add_edge("tools", "chatbot")

# START > chatbot
graph_builder.add_edge(START, "chatbot")

# chatbot > END
graph_builder.add_edge("chatbot", END)

# %% [markdown]
# ## 그래프 컴파일 (체크포인터 적용)
#
# 그래프를 컴파일할 때 `checkpointer` 파라미터에 생성한 `MemorySaver`를 전달합니다. 이렇게 하면 그래프가 각 노드를 실행할 때마다 자동으로 상태가 체크포인트에 저장됩니다.
#
# 동일한 `thread_id`로 그래프를 다시 호출하면, 저장된 상태를 불러와 이전 대화를 이어서 진행할 수 있습니다.
#
# 아래 코드에서는 체크포인터를 적용하여 그래프를 컴파일합니다.

# %%
# 그래프 빌더 컴파일
graph = graph_builder.compile(checkpointer=memory)

# %% [markdown]
# ## 그래프 시각화
#
# 그래프의 연결성은 이전 `LangGraph-Agent` 튜토리얼과 동일합니다. 차이점은 이번에 추가된 체크포인터가 그래프의 각 노드를 처리하면서 `State`를 체크포인트하여 저장한다는 것입니다.
#
# `langchain_teddynote.graphs` 모듈의 `visualize_graph()` 함수를 사용하여 그래프 구조를 확인합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

# 그래프 시각화
visualize_graph(graph)

# %% [markdown]
# ## RunnableConfig 설정
#
# `RunnableConfig` 을 정의하고 `recursion_limit` 과 `thread_id` 를 설정합니다.
#
# - `recursion_limit`: 최대 방문할 노드 수. 그 이상은 RecursionError 발생
# - `thread_id`: 스레드 ID 설정
#
# `thread_id` 는 대화 세션을 구분하는 데 사용됩니다. 즉, 메모리의 저장은 `thread_id` 에 따라 개별적으로 이루어집니다.

# %%
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
    configurable={"thread_id": "1"},  # 스레드 ID 설정
)

# %%
# 첫 질문
question = (
    "내 이름은 `테디노트` 입니다. YouTube 채널을 운영하고 있어요. 만나서 반가워요"
)

for event in graph.stream({"messages": [("user", question)]}, config=config):
    for value in event.values():
        value["messages"][-1].pretty_print()

# %%
# 이어지는 질문
question = "내 이름이 뭐라고 했지?"

for event in graph.stream({"messages": [("user", question)]}, config=config):
    for value in event.values():
        value["messages"][-1].pretty_print()

# %% [markdown]
# ## 다른 thread_id로 테스트
#
# 이번에는 `RunnableConfig`의 `thread_id`를 변경한 뒤, 이전 대화 내용을 기억하고 있는지 물어보겠습니다. `thread_id`가 다르면 별도의 대화 세션으로 취급되므로, 이전 대화 내용을 기억하지 못합니다.
#
# 이를 통해 `thread_id`가 대화 세션을 구분하는 역할을 함을 확인할 수 있습니다.

# %%
from langchain_core.runnables import RunnableConfig

question = "내 이름이 뭐라고 했지?"

config = RunnableConfig(
    recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
    configurable={"thread_id": "2"},  # 스레드 ID 설정
)

for event in graph.stream({"messages": [("user", question)]}, config=config):
    for value in event.values():
        value["messages"][-1].pretty_print()

# %% [markdown]
# ## 스냅샷: 저장된 State 확인
#
# 지금까지 두 개의 다른 스레드에서 몇 개의 체크포인트를 만들었습니다. 
#
# `Checkpoint` 에는 현재 상태 값, 해당 구성, 그리고 처리할 `next` 노드가 포함되어 있습니다.
#
# 주어진 설정에서 그래프의 `state`를 검사하려면 언제든지 `get_state(config)`를 호출하세요.

# %%
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    configurable={"thread_id": "1"},  # 스레드 ID 설정
)
# 그래프 상태 스냅샷 생성
snapshot = graph.get_state(config)
snapshot.values["messages"]

# %% [markdown]
# ### 설정 정보 확인 (config)
#
# `snapshot.config`를 통해 현재 스냅샷의 설정 정보를 확인할 수 있습니다. 여기에는 `thread_id`, `checkpoint_ns`, `checkpoint_id` 등의 정보가 포함되어 있습니다.
#
# 아래 코드에서는 스냅샷의 config 정보를 출력합니다.

# %%
# 설정된 config 정보
snapshot.config

# %% [markdown]
# ### 저장된 상태 값 확인 (values)
#
# `snapshot.values`를 통해 현재 체크포인트에 저장된 상태 값을 확인할 수 있습니다. 여기에는 `messages` 키 아래에 지금까지의 대화 내용이 저장되어 있습니다.
#
# 아래 코드에서는 저장된 상태 값을 출력합니다.

# %%
# 저장된 값(values)
snapshot.values

# %% [markdown]
# ### 다음 노드 확인 (next)
#
# `snapshot.next`를 통해 현재 시점에서 앞으로 방문할 다음 노드를 확인할 수 있습니다. 그래프가 `__END__`에 도달하여 실행이 완료된 경우, 다음 노드는 빈 값(`()`)이 출력됩니다.
#
# 아래 코드에서는 다음에 방문할 노드를 확인합니다.

# %%
# 다음 노드
snapshot.next

# %%
snapshot.metadata["writes"]["chatbot"]["messages"][0]

# %% [markdown]
# ### 메타데이터 시각화
#
# 복잡한 구조의 메타데이터를 시각화하기 위해 `langchain_teddynote.messages` 모듈의 `display_message_tree` 함수를 사용합니다. 이 함수는 중첩된 딕셔너리 구조를 트리 형태로 보기 쉽게 출력해줍니다.
#
# 아래 코드에서는 스냅샷의 메타데이터를 트리 형태로 시각화합니다.

# %%
from langchain_teddynote.messages import display_message_tree

# 메타데이터(tree 형태로 출력)
display_message_tree(snapshot.metadata)
