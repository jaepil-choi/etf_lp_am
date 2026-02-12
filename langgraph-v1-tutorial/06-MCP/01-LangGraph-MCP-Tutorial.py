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
# # LangGraph MCP (Model Context Protocol) 튜토리얼
#
# 이 튜토리얼에서는 LangGraph와 MCP(Model Context Protocol)를 통합하여 강력한 AI 에이전트를 구축하는 방법을 배웁니다. MCP는 AI 애플리케이션에서 도구(Tool)와 컨텍스트를 표준화된 방식으로 제공하는 오픈 프로토콜입니다.
#
# > 참고 문서: [Model Context Protocol 공식 문서](https://modelcontextprotocol.io/introduction)
#
# ## 학습 목표
#
# - MCP의 개념과 아키텍처를 이해합니다
# - MultiServerMCPClient를 사용하여 다중 서버를 관리하는 방법을 학습합니다
# - React Agent 및 ToolNode와 MCP를 통합하는 방법을 익힙니다
# - 실전 예제를 통해 복잡한 에이전트를 구축합니다
#
# ## 목차
#
# 1. MCP 개요 및 설치
# 2. 기본 MCP 서버 생성
# 3. MultiServerMCPClient 설정
# 4. React Agent와 MCP 통합
# 5. ToolNode와 MCP 통합
# 6. 다중 MCP 서버 관리
# 7. 실전 예제 - 복잡한 에이전트 구축

# %% [markdown]
# ## Part 1: MCP 기본 개념
#
# ### MCP(Model Context Protocol)란?
#
# MCP는 애플리케이션이 언어 모델에 도구와 컨텍스트를 제공하는 방법을 표준화한 오픈 프로토콜입니다. 이 프로토콜을 사용하면 다양한 서비스와 도구를 일관된 방식으로 LLM에 연결할 수 있습니다.
#
# ### 주요 특징
#
# - **표준화된 도구 인터페이스**: 일관된 방식으로 도구를 정의하고 사용할 수 있습니다
# - **다양한 전송 메커니즘**: stdio, HTTP, WebSocket 등 여러 통신 방식을 지원합니다
# - **동적 도구 검색**: 런타임에 도구를 자동으로 검색하고 로드할 수 있습니다
# - **확장 가능한 아키텍처**: 여러 서버를 동시에 연결하여 사용할 수 있습니다
#
# ### 설치
#
# MCP를 사용하기 위해 필요한 패키지를 설치합니다. `langchain-mcp-adapters` 패키지는 LangChain 에이전트가 MCP 서버에 정의된 도구를 사용할 수 있도록 해줍니다.
#
# 아래 코드는 튜토리얼에서 사용할 주요 패키지들을 import합니다.

# %%
import nest_asyncio
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# MCP 클라이언트: 여러 MCP 서버에 연결하여 도구를 가져옵니다
from langchain_mcp_adapters.client import MultiServerMCPClient

# %%
# 환경 변수 설정
from dotenv import load_dotenv

load_dotenv(override=True)

# 비동기 호출을 활성화합니다 (Jupyter 환경에서 필요)
nest_asyncio.apply()

# %%
# LangSmith 추적을 설정합니다
# https://smith.langchain.com 에서 프로젝트를 확인할 수 있습니다
from langchain_teddynote import logging

logging.langsmith("LangGraph-MCP")


# %% [markdown]
# ## Part 2: 기본 MCP 서버 생성
#
# MCP 서버는 도구를 제공하는 독립적인 프로세스입니다. FastMCP를 사용하면 Python으로 간단하게 MCP 서버를 만들 수 있습니다. 이 튜토리얼에서는 미리 준비된 MCP 서버들을 사용합니다.
#
# ### 제공되는 MCP 서버
#
# 이 튜토리얼에서 사용하는 MCP 서버 파일들은 `server/` 디렉토리에 위치해 있습니다:
#
# | 파일명 | 설명 | 전송 방식 |
# |--------|------|-----------|
# | `mcp_server_local.py` | 날씨 정보를 제공하는 로컬 서버 | stdio |
# | `mcp_server_remote.py` | 현재 시간을 제공하는 원격 서버 | HTTP |
# | `mcp_server_rag.py` | PDF 문서 검색 기능을 제공하는 RAG 서버 | stdio |
#
# 각 서버는 `FastMCP`를 사용하여 구현되어 있으며, 도구(Tool)를 정의하고 클라이언트 요청에 응답합니다.

# %% [markdown]
# ## Part 3: MultiServerMCPClient 설정
#
# `MultiServerMCPClient`는 여러 MCP 서버를 동시에 관리하고 연결할 수 있는 클라이언트입니다. 각 서버에서 제공하는 도구들을 통합하여 하나의 도구 목록으로 사용할 수 있습니다.
#
# ### 지원하는 전송 방식
#
# - **stdio**: 클라이언트가 서버를 서브프로세스로 실행하고 표준 입출력을 통해 통신합니다. 로컬 개발에 적합합니다.
# - **streamable_http**: 서버가 독립적인 프로세스로 실행되어 HTTP 요청을 처리합니다. 원격 연결에 적합합니다.
#
# 아래 코드는 MCP 클라이언트를 설정하고 서버에서 도구를 가져오는 헬퍼 함수를 정의합니다.

# %%
async def setup_mcp_client(server_configs: dict):
    """MCP 클라이언트를 설정하고 도구를 가져옵니다.
    
    Args:
        server_configs: 서버 구성 딕셔너리. 각 서버의 이름을 키로,
                       연결 정보(command, args, transport 또는 url)를 값으로 가집니다.
    
    Returns:
        tuple: (MCP 클라이언트, 로드된 도구 목록)
    """
    # MCP 클라이언트 생성
    client = MultiServerMCPClient(server_configs)

    # 서버에서 도구 가져오기
    tools = await client.get_tools()

    # 로드된 도구 목록 출력
    print(f"[MCP] {len(tools)}개의 도구가 로드되었습니다:")
    for tool in tools:
        print(f"  - {tool.name}")

    return client, tools


# %%
# 날씨 서버 구성 정의 (stdio 전송 방식)
server_configs = {
    "weather": {
        "command": "uv",  # uv 패키지 매니저 사용
        "args": ["run", "python", "server/mcp_server_local.py"],
        "transport": "stdio",  # 표준 입출력을 통한 통신
    },
}

# MCP 클라이언트 생성 및 도구 로드
client, tools = await setup_mcp_client(server_configs=server_configs)

# %%
# LLM 설정
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# React Agent 생성: MCP 도구를 사용하는 ReAct 패턴 에이전트
agent = create_react_agent(
    llm, 
    tools, 
    checkpointer=InMemorySaver()  # 대화 상태를 메모리에 저장
)

# %%
# 스트리밍 헬퍼 함수와 UUID 생성 함수를 import합니다
from langchain_teddynote.messages import astream_graph, random_uuid
from langchain_core.runnables import RunnableConfig

# 대화 스레드 ID를 설정합니다
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# 에이전트 실행: 날씨 정보 요청
response = await astream_graph(
    agent,
    inputs={"messages": [("human", "안녕하세요. 서울의 날씨를 알려주세요.")]},
    config=config,
)

# %% [markdown]
# ### HTTP 전송 방식 사용
#
# 원격 서버나 HTTP 엔드포인트를 사용하는 경우 `streamable_http` 전송 방식을 사용합니다. 이 방식은 서버가 별도의 프로세스로 실행 중이어야 합니다.
#
# **사전 준비**: 아래 코드를 실행하기 전에 별도의 터미널에서 Remote MCP 서버를 먼저 구동해야 합니다.
#
# ```bash
# uv run python server/mcp_server_remote.py
# ```
#
# 아래 코드는 HTTP 기반 MCP 서버에 연결하는 예제입니다.

# %%
# HTTP 기반 MCP 서버 설정
http_server_config = {
    "current_time": {
        "url": "http://127.0.0.1:8002/mcp",  # HTTP 엔드포인트 URL
        "transport": "streamable_http",  # HTTP 스트리밍 전송 방식
    },
}

# MCP 클라이언트 생성 및 HTTP 서버 도구 로드
client, http_tools = await setup_mcp_client(server_configs=http_server_config)

# %%
# LLM 설정 (경량 모델 사용)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# HTTP 도구를 사용하는 React Agent 생성
agent = create_react_agent(
    llm, 
    http_tools, 
    checkpointer=InMemorySaver()
)

# %%
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# 에이전트 실행: 현재 시간 요청
response = await astream_graph(
    agent,
    inputs={"messages": [("human", "안녕하세요. 현재 시간을 알려주세요.")]},
    config=config,
)

# %% [markdown]
# ### MCP Inspector 사용
#
# MCP Inspector는 MCP 서버를 테스트하고 디버깅할 수 있는 웹 기반 도구입니다. 브라우저에서 서버의 도구 목록을 확인하고, 직접 도구를 호출하여 결과를 확인할 수 있습니다.
#
# 다음 명령어를 터미널에서 실행하면 MCP Inspector가 시작됩니다:
#
# ```bash
# npx @modelcontextprotocol/inspector
# ```
#
# 아래 이미지는 MCP Inspector의 인터페이스 예시입니다.

# %% [markdown]
# ![mcp_inspector](./assets/mcp-inspector.png)

# %%
# RAG(검색 증강 생성) MCP 서버 설정
# PDF 문서에서 정보를 검색하는 기능을 제공합니다
rag_server_config = {
    "rag": {
        "command": "uv",
        "args": ["run", "python", "server/mcp_server_rag.py"],
        "transport": "stdio",
    },
}

# MCP 클라이언트 생성 및 RAG 도구 로드
client, rag_tools = await setup_mcp_client(server_configs=rag_server_config)

# %%
# LLM 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# RAG 도구를 사용하는 React Agent 생성
rag_agent = create_react_agent(
    llm, 
    rag_tools, 
    checkpointer=InMemorySaver()
)

# %%
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# RAG 에이전트 실행: PDF 문서에서 정보 검색
_ = await astream_graph(
    rag_agent,
    inputs={
        "messages": [
            (
                "human",
                "삼성전자가 개발한 생성형 AI 의 이름은? mcp 서버를 사용해서 검색해주세요.",
            )
        ]
    },
    config=config,
)

# %%
# 다른 질문으로 RAG 에이전트 테스트
_ = await astream_graph(
    rag_agent,
    inputs={
        "messages": [
            (
                "human",
                "구글이 Anthropic 에 투자하기로 한 금액을 검색해줘",
            )
        ]
    },
    config=config,
)


# %% [markdown]
# ## Part 4: React Agent와 MCP 통합
#
# React Agent는 추론(Reason)과 행동(Act)을 반복하는 ReAct 패턴을 구현합니다. LLM이 상황을 분석하고, 필요한 도구를 선택하여 호출하고, 결과를 바탕으로 다음 행동을 결정하는 과정을 자동으로 수행합니다.
#
# MCP 도구와 함께 사용하면 다양한 외부 서비스에 접근할 수 있는 강력한 에이전트를 만들 수 있습니다.
#
# > 참고 문서: [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
#
# 아래 코드는 MCP 도구를 사용하는 React Agent를 생성하는 함수를 정의합니다.

# %%
async def create_mcp_react_agent(server_configs: dict):
    """MCP 도구를 사용하는 React Agent를 생성합니다.
    
    이 함수는 주어진 서버 구성으로 MCP 클라이언트를 생성하고,
    해당 도구들을 사용하는 React Agent를 반환합니다.
    
    Args:
        server_configs: MCP 서버 구성 딕셔너리
    
    Returns:
        CompiledStateGraph: 컴파일된 React Agent
    """
    # MCP 클라이언트 생성 및 도구 가져오기
    client, tools = await setup_mcp_client(server_configs=server_configs)

    # LLM 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # React Agent 생성
    agent = create_react_agent(
        llm, 
        tools, 
        checkpointer=InMemorySaver()
    )

    return agent


# %%
# 다중 MCP 서버 구성: 날씨(stdio) + 시간(HTTP)
server_configs = {
    "weather": {
        "command": "uv",
        "args": ["run", "python", "server/mcp_server_local.py"],
        "transport": "stdio",
    },
    "current_time": {
        "url": "http://127.0.0.1:8002/mcp",
        "transport": "streamable_http",
    },
}

# 다중 MCP 서버를 사용하는 React Agent 생성
agent = await create_mcp_react_agent(server_configs)

# %%
# 대화 스레드 설정 (상태 유지를 위해 동일한 thread_id 사용)
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# 첫 번째 질문: 현재 시간
await astream_graph(
    agent, 
    inputs={"messages": [("human", "현재 시간을 알려주세요")]}, 
    config=config
)

# 두 번째 질문: 날씨 (같은 대화 스레드에서 연속 질문)
await astream_graph(
    agent,
    inputs={"messages": [("human", "현재 서울의 날씨도 알려주세요")]},
    config=config,
)

# %% [markdown]
# ## Part 5: ToolNode와 MCP 통합
#
# `ToolNode`를 사용하면 LangGraph에서 더 세밀한 제어가 가능한 커스텀 워크플로우를 만들 수 있습니다. React Agent와 달리, 그래프의 각 노드를 직접 정의하고 연결할 수 있어 복잡한 로직을 구현하기에 적합합니다.
#
# ### ToolNode의 특징
#
# - **세밀한 제어**: 각 노드의 동작을 직접 정의할 수 있습니다
# - **유연한 워크플로우**: 조건부 분기, 병렬 처리 등 복잡한 흐름을 구현할 수 있습니다
# - **확장성**: 추가 도구(예: Tavily 검색)를 쉽게 통합할 수 있습니다
#
# 아래 코드는 MCP 도구와 추가 도구를 결합한 커스텀 워크플로우를 생성합니다.

# %%
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_teddynote.graphs import visualize_graph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Dict, Any
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langchain_tavily import TavilySearch


class AgentState(TypedDict):
    """에이전트 상태 정의
    
    Attributes:
        messages: 대화 메시지 목록. add_messages 리듀서로 메시지가 누적됩니다.
        context: 추가 컨텍스트 정보를 저장하는 딕셔너리 (선택적)
    """
    messages: Annotated[List[BaseMessage], add_messages]
    context: Dict[str, Any]


async def create_mcp_workflow(server_configs: dict):
    """MCP 도구를 사용하는 커스텀 워크플로우를 생성합니다.
    
    이 함수는 MCP 도구와 Tavily 검색 도구를 결합하여
    에이전트-도구 루프를 구현하는 그래프를 생성합니다.
    
    Args:
        server_configs: MCP 서버 구성 딕셔너리
    
    Returns:
        CompiledStateGraph: 컴파일된 워크플로우 그래프
    """
    # MCP 클라이언트 생성 및 도구 로드
    client, tools = await setup_mcp_client(server_configs=server_configs)

    # Tavily 웹 검색 도구 추가
    tavily_tool = TavilySearch(max_results=2)
    tools.append(tavily_tool)

    # LLM 설정 및 도구 바인딩
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # 워크플로우 그래프 생성
    workflow = StateGraph(AgentState)

    async def agent_node(state: AgentState):
        """에이전트 노드: LLM을 호출하여 응답을 생성합니다"""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    # ToolNode 생성: 도구 호출을 처리합니다
    tool_node = ToolNode(tools)

    # 그래프에 노드 추가
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 엣지 정의: 시작 -> 에이전트
    workflow.add_edge(START, "agent")
    
    # 조건부 엣지: 에이전트 -> (도구 or 종료)
    # tools_condition은 도구 호출이 필요하면 "tools"로, 아니면 END로 라우팅합니다
    workflow.add_conditional_edges("agent", tools_condition)
    
    # 도구 -> 에이전트 (도구 실행 후 다시 에이전트로)
    workflow.add_edge("tools", "agent")

    # 그래프 컴파일
    app = workflow.compile(checkpointer=InMemorySaver())

    # 그래프 시각화
    visualize_graph(app)

    return app


# %%
# MCP 서버 구성 정의
server_configs = {
    "weather": {
        "command": "uv",
        "args": ["run", "python", "server/mcp_server_local.py"],
        "transport": "stdio",
    },
    "current_time": {
        "url": "http://127.0.0.1:8002/mcp",
        "transport": "streamable_http",
    },
}

# %%
# MCP 워크플로우 생성 및 그래프 시각화
mcp_app = await create_mcp_workflow(server_configs)

# %%
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# MCP 워크플로우 실행: 현재 시간 조회
_ = await astream_graph(
    mcp_app, 
    inputs={"messages": [("human", "현재 시간을 알려주세요")]}, 
    config=config
)

# %%
# 복합 작업: 시간 조회 후 뉴스 검색 (Tavily 도구 사용)
_ = await astream_graph(
    mcp_app,
    inputs={
        "messages": [
            ("human", "오늘 뉴스를 검색해주세요. 검색시 시간을 조회한 뒤 처리하세요.")
        ]
    },
    config=config,
)

# %% [markdown]
# ## Part 6: 외부 MCP 서버에서 3rd Party 도구 사용하기
#
# ### Smithery AI란?
#
# [Smithery AI](https://smithery.ai/)는 AI 에이전트 서비스의 허브 역할을 하는 플랫폼입니다. 에이전트형 AI(예: 대형 언어 모델)가 외부 도구나 정보와 효율적으로 연결될 수 있도록 설계된 MCP 서버들을 검색하고 배포하는 역할을 수행합니다.
#
# ### 주요 기능
#
# - **MCP 서버 레지스트리**: 다양한 용도의 MCP 서버를 검색하고 사용할 수 있습니다
# - **표준 프로토콜**: MCP(Model Context Protocol) 표준을 따르는 서버들을 제공합니다
# - **손쉬운 통합**: `npx`를 통해 간편하게 MCP 서버에 연결할 수 있습니다
#
# ### Context7 MCP 서버
#
# 이 예제에서는 Smithery AI에서 제공하는 `context7-mcp` 서버를 사용합니다. 이 서버는 최신 프로그래밍 언어 및 프레임워크 문서를 검색하고 제공하는 기능을 가지고 있습니다.
#
# 아래 코드는 Context7 MCP 서버를 포함한 다중 서버 구성을 설정합니다.

# %%
# 다중 MCP 서버 구성 (로컬 + HTTP + Smithery AI)
server_configs = {
    # 로컬 날씨 서버 (stdio)
    "weather": {
        "command": "uv",
        "args": ["run", "python", "server/mcp_server_local.py"],
        "transport": "stdio",
    },
    # 원격 시간 서버 (HTTP)
    "current_time": {
        "url": "http://127.0.0.1:8002/mcp",
        "transport": "streamable_http",
    },
    # Smithery AI의 Context7 MCP 서버: 최신 문서 검색
    "context7-mcp": {
        "command": "npx",
        "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@upstash/context7-mcp",
            "--key",
            "7c5b4b8f-cb2a-4c2b-b0e0-d3cc49fc7a85",
        ],
        "transport": "stdio",
    },
}

# 다중 서버를 사용하는 MCP 워크플로우 생성
mcp_app = await create_mcp_workflow(server_configs)

# %%
# 새로운 대화 스레드 설정
config = RunnableConfig(configurable={"thread_id": random_uuid()})

# Context7 서버를 활용한 복합 작업:
# 1. 최신 LangGraph 문서에서 ReAct Agent 관련 내용 검색
# 2. 검색된 정보를 바탕으로 코드 생성
await astream_graph(
    mcp_app,
    inputs={
        "messages": [
            (
                "human",
                "최신 LangGraph 도큐먼트에서 ReAct Agent 관련 내용을 검색하세요. 그런 다음 Tavily 검색을 수행하는 ReAct Agent를 생성하세요. 사용 LLM=gpt-4.1-mini",
            )
        ]
    },
    config=config,
)
