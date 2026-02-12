# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: langchain-kr-CdOel15G-py3.11
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 계층적 에이전트 팀 (Hierarchical Agent Team)
#
# 이 튜토리얼에서는 **계층적 에이전트 팀**을 구성하는 방법을 살펴봅니다.
#
# 단일 에이전트나 단일 수준의 감독자(Supervisor)로는 대응하기 힘든 복잡한 작업을 **계층적 구조**를 통해 분할하고, 각각의 하위 수준 감독자(Sub-Supervisor)가 해당 영역에 특화된 작업자(Worker) 에이전트를 관리하는 방식을 구현합니다.
#
# 이러한 계층적 접근 방식은 작업자가 너무 많아질 경우나, 단일 작업자가 처리하기 힘든 복잡한 작업을 효율적으로 해결하는 데 도움이 됩니다.  
#
# 본 예제는 [AutoGen 논문](https://arxiv.org/abs/2308.08155)의 아이디어를 LangGraph를 통해 구현한 사례로, 웹 연구와 문서 작성이라는 두 가지 하위 작업을 서로 다른 팀으로 구성하고, 상위 및 중간 수준의 감독자를 통해 전체 프로세스를 관리하는 방법을 제시합니다.
#
# ![](./assets/langgraph-multi-agent-team-supervisor.png)
#
# ---
#
# ## 왜 계층적 에이전트 팀인가?
#
# 이전 Supervisor 예제에서는 하나의 Supervisor 노드가 여러 작업자 노드에게 작업을 할당하고 결과를 취합하는 과정을 살펴보았습니다. 이 방식은 간단한 경우에 효율적입니다. 그러나 다음과 같은 상황에서는 계층적 구조가 필요할 수 있습니다.
#
# - **작업 복잡성 증가**: 단일 Supervisor로는 한 번에 처리할 수 없는 다양한 하위 영역의 전문 지식이 필요할 수 있습니다.
# - **작업자 수 증가**: 많은 수의 작업자를 관리할 때, 단일 Supervisor가 모든 작업자에게 직접 명령을 내리면 관리 부담이 커집니다.
#
# 이러한 상황에서 상위 수준의 Supervisor는 하위 수준의 **Sub-Supervisor**들에게 작업을 할당하고, 각 **Sub-Supervisor**는 해당 작업을 전문화된 작업자 팀에 재할당하는 계층적 구조를 구성할 수 있습니다.
#
# ---
#
# ## 이 튜토리얼에서 다룰 내용
#
# 1. **도구 생성**: 웹 연구(Web Research) 및 문서 작성(Documentation)을 위한 에이전트 도구 정의    
# 2. **에이전트 팀 정의**: 연구 팀 및 문서 작성 팀을 계층적으로 정의하고 구성  
# 3. **계층 추가**: 상위 수준 그래프와 중간 수준 감독자를 통해 전체 작업을 계층적으로 조정  
# 4. **결합**: 모든 요소를 통합하여 최종적인 계층적 에이전트 팀 구축
#
# > **참고 문서**
# > - [AutoGen 논문: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Wu et al.)](https://arxiv.org/abs/2308.08155)
# > - [LangGraph Multi-Agent 개념](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

# %% [markdown]
# ## 환경 설정
#
# 필요한 패키지를 설치하고 API 키를 설정합니다. `dotenv`를 사용하여 환경 변수를 로드하고, LangSmith 추적을 활성화합니다.
#
# 아래 코드는 환경 변수를 로드합니다.

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드 (override=True: 기존 환경변수 덮어쓰기)
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-Tutorial")

# %%
from langchain_teddynote.models import get_model_name, LLMs

# 최신 버전의 모델명을 가져옵니다.
MODEL_NAME = get_model_name(LLMs.GPT4)
print(f"사용 모델: {MODEL_NAME}")

# %% [markdown]
# ## 도구 생성
#
# 각 팀은 하나 이상의 에이전트로 구성되며, 각 에이전트는 하나 이상의 도구를 갖추게 됩니다. 아래에서는 다양한 팀에서 사용할 모든 도구를 정의합니다.
#
# ### Research Team 도구
#
# Research Team은 웹에서 정보를 찾기 위해 검색 엔진과 URL 스크래퍼를 사용할 수 있습니다. 
#
# - **TavilySearch**: 웹 검색을 수행하는 도구입니다.
# - **scrape_webpages**: 특정 URL에서 상세 정보를 스크래핑하는 도구입니다.
#
# 아래 코드는 Research Team에서 사용할 도구들을 정의합니다.

# %%
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_core.tools import tool

# TavilySearch 도구 정의: 최대 5개의 검색 결과 반환
tavily_tool = TavilySearch(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """웹 페이지 스크래핑 도구

    주어진 URL 목록에서 웹 페이지 내용을 스크래핑합니다.
    requests와 BeautifulSoup을 사용하여 상세 정보를 추출합니다.
    """
    # 웹 페이지 로더 초기화: User-Agent 헤더 설정
    loader = WebBaseLoader(
        web_path=urls,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
        },
    )
    # 문서 로드
    docs = loader.load()

    # 로드된 문서의 제목과 내용을 XML 형식으로 변환하여 반환
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


# %% [markdown]
# ### Doc Writing Team 도구
#
# 다음으로, 문서 작성 팀(Doc Writing Team)이 사용할 도구들을 정의합니다. 이 도구들은 에이전트가 파일 시스템에 접근할 수 있도록 합니다.
#
# - **create_outline**: 문서의 개요를 생성하고 파일로 저장합니다.
# - **read_document**: 저장된 문서를 읽습니다.
# - **write_document**: 새 문서를 작성하고 저장합니다.
# - **edit_document**: 기존 문서를 편집합니다.
#
# > **주의**: 파일 시스템 접근은 보안상 위험할 수 있으므로 사용에 주의가 필요합니다.
#
# 아래 코드는 Doc Writing Team에서 사용할 파일 접근 도구들을 정의합니다.

# %%
from pathlib import Path
from typing import Dict, Optional, List
from typing_extensions import Annotated

# 작업 디렉토리 설정
WORKING_DIRECTORY = Path("./tmp")
WORKING_DIRECTORY.mkdir(exist_ok=True)  # tmp 폴더가 없으면 생성


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """아웃라인 생성 도구

    주어진 포인트 목록으로 아웃라인을 생성하고 파일로 저장합니다.
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """문서 읽기 도구

    지정된 파일에서 문서 내용을 읽어 반환합니다.
    시작/종료 줄을 지정하여 부분 읽기도 가능합니다.
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    # 시작 줄이 지정되지 않은 경우 기본값 0 설정
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """문서 작성 도구

    텍스트 내용을 받아 파일로 저장합니다.
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "File path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "File path of the edited document."]:
    """문서 편집 도구

    지정된 줄 번호에 텍스트를 삽입하여 문서를 편집합니다.
    줄 번호는 1부터 시작합니다.
    """
    # 문서 읽기
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    # 삽입할 텍스트를 줄 번호 순으로 정렬하여 처리
    sorted_inserts = sorted(inserts.items())

    # 지정된 줄 번호에 텍스트 삽입
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    # 편집된 문서 저장
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# %% [markdown]
# ### Python REPL 도구
#
# 다음은 코드 실행 도구인 `PythonREPLTool`을 정의합니다. 이 도구는 에이전트가 Python 코드를 실행할 수 있게 해주며, 차트 생성이나 데이터 처리에 활용됩니다.
#
# 아래 코드는 PythonREPL 도구를 초기화합니다.

# %%
from langchain_experimental.tools import PythonREPLTool

# Python REPL 도구 초기화
python_repl_tool = PythonREPLTool()

# %% [markdown]
# ## 유틸리티 함수 정의
#
# 다중 에이전트 시스템을 효율적으로 구축하기 위해 유틸리티 함수와 클래스를 정의합니다. 이를 통해 에이전트 생성 및 관리 코드를 재사용할 수 있습니다.
#
# `AgentFactory` 클래스는 다음 기능을 제공합니다.
#
# 1. **Worker Agent 생성**: ReAct 패턴의 에이전트를 생성합니다.
# 2. **에이전트 노드 래퍼**: 에이전트를 그래프 노드 함수로 변환합니다.
#
# 아래 코드는 AgentFactory 클래스를 정의합니다.

# %%
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent


class AgentFactory:
    """에이전트 팩토리 클래스

    에이전트 생성 및 노드 래핑을 담당합니다.
    동일한 LLM 설정으로 여러 에이전트를 생성할 때 유용합니다.
    """
    
    def __init__(self, model_name):
        """초기화

        Args:
            model_name: 사용할 LLM 모델명
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def create_agent_node(self, agent, name: str):
        """에이전트를 그래프 노드 함수로 변환

        Args:
            agent: ReAct 에이전트 인스턴스
            name: 에이전트 이름 (메시지 출처 식별용)

        Returns:
            그래프에서 사용할 수 있는 노드 함수
        """
        def agent_node(state):
            # 에이전트 실행
            result = agent.invoke(state)
            # 결과를 HumanMessage로 변환하여 반환
            return {
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name=name)
                ]
            }
        return agent_node


# LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# AgentFactory 인스턴스 생성
agent_factory = AgentFactory(MODEL_NAME)

# %% [markdown]
# ### AgentFactory 사용 예시
#
# `AgentFactory`를 사용하여 에이전트 노드를 생성하는 방법을 살펴보겠습니다. 아래 예시에서는 검색 에이전트를 생성합니다.
#
# 아래 코드는 검색 에이전트와 해당 노드를 생성합니다.

# %%
# ReAct 패턴의 검색 에이전트 생성
search_agent = create_react_agent(llm, tools=[tavily_tool])

# 에이전트를 그래프 노드로 변환
search_node = agent_factory.create_agent_node(search_agent, name="Searcher")

# %% [markdown]
# ### Team Supervisor 생성 함수
#
# 다음은 팀 감독자(Team Supervisor)를 생성하는 함수입니다. 이 함수는 지정된 멤버들 중에서 다음 작업자를 선택하거나, 작업 완료 시 `FINISH`를 반환하는 Supervisor 체인을 생성합니다.
#
# 아래 코드는 Team Supervisor 생성 함수를 정의합니다.

# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal


def create_team_supervisor(model_name, system_prompt, members) -> str:
    """Team Supervisor 생성 함수

    팀 멤버를 관리하고 다음 작업자를 결정하는 Supervisor 체인을 생성합니다.

    Args:
        model_name: 사용할 LLM 모델명
        system_prompt: Supervisor의 시스템 프롬프트
        members: 관리할 팀 멤버 목록

    Returns:
        Supervisor 체인 (prompt | llm with structured output)
    """
    # 다음 작업자 선택 옵션: FINISH + 멤버 목록
    options_for_next = ["FINISH"] + members

    # 라우팅 응답 모델 정의
    class RouteResponse(BaseModel):
        next: Literal[*options_for_next]

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options_for_next))

    # LLM 초기화
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Supervisor 체인 구성: 프롬프트 -> LLM (구조화된 출력)
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    return supervisor_chain


# %% [markdown]
# ## 에이전트 팀 정의
#
# 이제 Research Team과 Doc Writing Team을 정의합니다. 각 팀은 자체 Sub-Supervisor를 가지며, 해당 영역에 특화된 작업자 에이전트들로 구성됩니다.
#
# ### Research Team
#
# Research Team은 웹 검색과 스크래핑을 담당하는 두 개의 작업자 노드를 가집니다.
#
# - **Searcher**: TavilySearch를 사용하여 웹 검색을 수행합니다.
# - **WebScraper**: 특정 URL에서 상세 정보를 스크래핑합니다.
#
# 아래 코드는 Research Team의 상태, 에이전트, Supervisor를 정의합니다.

# %%
import operator
from typing import List, TypedDict
from typing_extensions import Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent


class ResearchState(TypedDict):
    """Research Team 상태 정의

    Attributes:
        messages: 에이전트 간 공유하는 메시지 목록
        team_members: 팀 멤버 에이전트 목록
        next: 다음으로 실행할 에이전트
    """
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str


# LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# Searcher 에이전트 및 노드 생성
search_agent = create_react_agent(llm, tools=[tavily_tool])
search_node = agent_factory.create_agent_node(search_agent, name="Searcher")

# WebScraper 에이전트 및 노드 생성
web_scraping_agent = create_react_agent(llm, tools=[scrape_webpages])
web_scraping_node = agent_factory.create_agent_node(
    web_scraping_agent, name="WebScraper"
)

# Research Team Supervisor 생성
supervisor_agent = create_team_supervisor(
    MODEL_NAME,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: Search, WebScraper. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Searcher", "WebScraper"],
)


# %% [markdown]
# ### 라우팅 함수 정의
#
# Supervisor의 결정에 따라 다음 노드를 선택하는 라우팅 함수를 정의합니다. 이 함수는 상태에서 `next` 값을 추출하여 반환합니다.
#
# 아래 코드는 라우팅 함수를 정의합니다.

# %%
def get_next_node(x):
    """상태에서 다음 노드를 반환하는 함수"""
    return x["next"]


# %% [markdown]
# ### Research Team 그래프 생성
#
# Research Team의 StateGraph를 생성합니다. 노드와 엣지를 연결하여 워크플로우를 구성하고, Supervisor의 결정에 따라 작업자를 동적으로 선택합니다.
#
# 아래 코드는 Research Team 그래프를 생성하고 컴파일합니다.

# %%
from langchain_teddynote.graphs import visualize_graph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Research Team 그래프 생성
web_research_graph = StateGraph(ResearchState)

# 노드 추가
web_research_graph.add_node("Searcher", search_node)
web_research_graph.add_node("WebScraper", web_scraping_node)
web_research_graph.add_node("Supervisor", supervisor_agent)

# 작업자 -> Supervisor 엣지 추가
web_research_graph.add_edge("Searcher", "Supervisor")
web_research_graph.add_edge("WebScraper", "Supervisor")

# 조건부 엣지: Supervisor의 결정에 따라 다음 노드로 이동
web_research_graph.add_conditional_edges(
    "Supervisor",
    get_next_node,
    {"Searcher": "Searcher", "WebScraper": "WebScraper", "FINISH": END},
)

# 시작 노드 설정
web_research_graph.set_entry_point("Supervisor")

# 그래프 컴파일
web_research_app = web_research_graph.compile(checkpointer=MemorySaver())

# 그래프 시각화
visualize_graph(web_research_app, xray=True)

# %% [markdown]
# ### Research Team 실행
#
# 생성된 `web_research_app`을 실행하여 Research Team의 동작을 확인합니다. 실행 결과는 팀 멤버 간의 협업 과정을 보여줍니다.
#
# 아래 코드는 그래프 실행을 위한 헬퍼 함수를 정의합니다.

# %%
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid, invoke_graph


def run_graph(app, message: str, recursive_limit: int = 50):
    """그래프 실행 헬퍼 함수

    Args:
        app: 컴파일된 그래프 앱
        message: 사용자 입력 메시지
        recursive_limit: 최대 재귀 횟수

    Returns:
        그래프 실행 후 최종 상태 값
    """
    # 설정: 재귀 제한과 스레드 ID 지정
    config = RunnableConfig(
        recursion_limit=recursive_limit, 
        configurable={"thread_id": random_uuid()}
    )

    # 입력 메시지 구성
    inputs = {
        "messages": [HumanMessage(content=message)],
    }

    # 그래프 실행 및 결과 스트리밍
    invoke_graph(app, inputs, config)

    # 최종 상태 반환
    return app.get_state(config).values


# %%
# Research Team 실행: 네이버 금융 뉴스 정리 요청
output = run_graph(
    web_research_app,
    "https://finance.naver.com/news 의 주요 뉴스 정리해서 출력해줘. 출처(URL) 도 함께 출력해줘.",
)

# %%
# 최종 결과 출력
print(output["messages"][-1].content)

# %% [markdown]
# ### Doc Writing Team
#
# 이번에는 문서 작성 팀(Doc Writing Team)을 생성합니다. 이 팀은 세 개의 작업자 에이전트로 구성됩니다.
#
# - **DocWriter**: 문서를 작성하고 편집합니다.
# - **NoteTaker**: 연구 자료를 바탕으로 개요를 작성합니다.
# - **ChartGenerator**: 데이터를 시각화하는 차트를 생성합니다.
#
# 각 에이전트에게는 서로 다른 파일 접근 도구가 제공됩니다. 또한, 상태 전처리 노드를 통해 각 에이전트가 현재 작업 디렉토리의 파일 목록을 인식할 수 있도록 합니다.
#
# 아래 코드는 Doc Writing Team의 상태, 에이전트, Supervisor를 정의합니다.

# %%
import operator
from typing import List, TypedDict, Annotated
from pathlib import Path

# 작업 디렉토리 설정
WORKING_DIRECTORY = Path("./tmp")
WORKING_DIRECTORY.mkdir(exist_ok=True)


class DocWritingState(TypedDict):
    """Doc Writing Team 상태 정의

    Attributes:
        messages: 에이전트 간 공유하는 메시지 목록
        team_members: 팀 멤버 목록
        next: 다음으로 실행할 에이전트
        current_files: 현재 작업 디렉토리의 파일 목록
    """
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str


def preprocess(state):
    """상태 전처리 함수

    작업 디렉토리의 파일 목록을 상태에 추가하여
    에이전트가 현재 파일 상태를 인식할 수 있도록 합니다.
    """
    written_files = []

    try:
        # 작업 디렉토리 내의 모든 파일 검색
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except Exception:
        pass

    # 파일이 없으면 해당 메시지 반환
    if not written_files:
        return {**state, "current_files": "No files written."}

    # 파일 목록을 상태에 추가
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


# LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME)

# DocWriter 에이전트 생성
doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    prompt="You are a arxiv researcher. Your mission is to write arxiv style paper on given topic/resources.",
)
context_aware_doc_writer_agent = preprocess | doc_writer_agent
doc_writing_node = agent_factory.create_agent_node(
    context_aware_doc_writer_agent, name="DocWriter"
)

# NoteTaker 에이전트 생성
note_taking_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt="You are an expert in creating outlines for research papers. Your mission is to create an outline for a given topic/resources or documents.",
)
context_aware_note_taking_agent = preprocess | note_taking_agent
note_taking_node = agent_factory.create_agent_node(
    context_aware_note_taking_agent, name="NoteTaker"
)

# ChartGenerator 에이전트 생성
chart_generating_agent = create_react_agent(
    llm, tools=[read_document, python_repl_tool]
)
context_aware_chart_generating_agent = preprocess | chart_generating_agent
chart_generating_node = agent_factory.create_agent_node(
    context_aware_chart_generating_agent, name="ChartGenerator"
)

# Doc Writing Team Supervisor 생성
doc_writing_supervisor = create_team_supervisor(
    MODEL_NAME,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  ['DocWriter', 'NoteTaker', 'ChartGenerator']. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["DocWriter", "NoteTaker", "ChartGenerator"],
)

# %% [markdown]
# ### Doc Writing Team 그래프 생성
#
# Doc Writing Team의 StateGraph를 생성합니다. Research Team과 동일한 구조로 노드와 엣지를 연결합니다.
#
# 아래 코드는 Doc Writing Team 그래프를 생성하고 컴파일합니다.

# %%
# Doc Writing Team 그래프 생성
authoring_graph = StateGraph(DocWritingState)

# 노드 추가
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("ChartGenerator", chart_generating_node)
authoring_graph.add_node("Supervisor", doc_writing_supervisor)

# 작업자 -> Supervisor 엣지 추가
authoring_graph.add_edge("DocWriter", "Supervisor")
authoring_graph.add_edge("NoteTaker", "Supervisor")
authoring_graph.add_edge("ChartGenerator", "Supervisor")

# 조건부 엣지: Supervisor의 결정에 따라 다음 노드로 이동
authoring_graph.add_conditional_edges(
    "Supervisor",
    get_next_node,
    {
        "DocWriter": "DocWriter",
        "NoteTaker": "NoteTaker",
        "ChartGenerator": "ChartGenerator",
        "FINISH": END,
    },
)

# 시작 노드 설정
authoring_graph.set_entry_point("Supervisor")

# 그래프 컴파일
authoring_app = authoring_graph.compile(checkpointer=MemorySaver())

# %% [markdown]
# ### Doc Writing Team 시각화
#
# 생성된 Doc Writing Team 그래프를 시각화합니다.
#
# 아래 코드는 그래프 구조를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

# Doc Writing Team 그래프 시각화
visualize_graph(authoring_app, xray=True)

# %% [markdown]
# ### Doc Writing Team 실행
#
# Doc Writing Team 그래프를 실행하여 문서 작성 과정을 확인합니다. 이 예시에서는 Transformer 구조에 대한 논문 작성을 요청합니다.
#
# 아래 코드는 Doc Writing Team을 실행합니다.

# %%
# Doc Writing Team 실행: Transformer 구조 논문 작성 요청
output = run_graph(
    authoring_app,
    "Transformer 의 구조에 대해서 심층 파악해서 논문의 목차를 한글로 작성해줘. "
    "그 다음 각각의 목차에 대해서 5문장 이상 작성해줘. "
    "상세내용 작성시 만약 chart 가 필요하면 차트를 작성해줘. "
    "최종 결과를 저장해줘. ",
)

# %% [markdown]
# ## Super-Graph 생성
#
# 이 설계에서는 **상향식 계획 정책**을 적용하고 있습니다. 앞서 두 개의 팀 그래프(Research Team, Doc Writing Team)를 생성했습니다. 이제 이 두 팀을 조정하는 상위 수준의 **Super-Graph**를 정의합니다.
#
# Super-Graph는 다음 역할을 수행합니다.
#
# - **팀 간 작업 라우팅**: 사용자 요청에 따라 적절한 팀에 작업을 할당합니다.
# - **상태 공유 관리**: 서로 다른 그래프 간에 상태가 어떻게 공유되는지 정의합니다.
# - **최종 결과 조합**: 각 팀의 결과를 취합하여 최종 응답을 생성합니다.
#
# 아래 코드는 Super-Graph의 총 감독자 노드를 생성합니다.

# %% [markdown]
# ### 총 감독자 노드 생성
#
# 먼저 두 팀(ResearchTeam, PaperWritingTeam)을 관리하는 총 감독자 노드를 생성합니다.
#
# 아래 코드는 총 감독자를 정의합니다.

# %%
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI

# LLM 초기화
llm = ChatOpenAI(model=MODEL_NAME)

# 총 감독자 노드 생성: 두 팀을 관리
supervisor_node = create_team_supervisor(
    MODEL_NAME,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: ['ResearchTeam', 'PaperWritingTeam']. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["ResearchTeam", "PaperWritingTeam"],
)

# %% [markdown]
# ### Super-Graph 상태 및 노드 정의
#
# Super-Graph의 상태와 유틸리티 노드를 정의합니다. Super-Graph는 주로 팀 간 작업을 라우팅하는 역할을 수행합니다.
#
# - **get_last_message**: 마지막 메시지를 추출하여 하위 그래프에 전달합니다.
# - **join_graph**: 하위 그래프의 결과를 취합합니다.
#
# 아래 코드는 Super-Graph 상태와 유틸리티 노드를 정의합니다.

# %%
from typing import TypedDict, List, Annotated
import operator


class State(TypedDict):
    """Super-Graph 상태 정의

    Attributes:
        messages: 메시지 목록
        next: 다음으로 실행할 팀 또는 FINISH
    """
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    """마지막 메시지 추출 함수

    상태에서 마지막 메시지를 추출하여 하위 그래프에 전달합니다.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, str):
        return {"messages": [HumanMessage(content=last_message)]}
    else:
        return {"messages": [last_message.content]}


def join_graph(response: dict):
    """그래프 결과 취합 함수

    하위 그래프의 마지막 메시지를 추출하여 반환합니다.
    """
    return {"messages": [response["messages"][-1]]}


# %% [markdown]
# ### Super-Graph 구성
#
# 이제 두 팀을 연결하는 Super-Graph를 정의합니다. 각 팀은 서브그래프로 포함되며, 총 감독자가 팀 간 작업을 조정합니다.
#
# 아래 코드는 Super-Graph를 생성하고 컴파일합니다.

# %%
# Super-Graph 생성
super_graph = StateGraph(State)

# 노드 추가: 각 팀을 서브그래프로 연결
super_graph.add_node("ResearchTeam", get_last_message | web_research_app | join_graph)
super_graph.add_node("PaperWritingTeam", get_last_message | authoring_app | join_graph)
super_graph.add_node("Supervisor", supervisor_node)

# 팀 -> Supervisor 엣지 추가
super_graph.add_edge("ResearchTeam", "Supervisor")
super_graph.add_edge("PaperWritingTeam", "Supervisor")

# 조건부 엣지: Supervisor의 결정에 따라 다음 팀으로 이동
super_graph.add_conditional_edges(
    "Supervisor",
    get_next_node,
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END,
    },
)

# 시작 노드: Supervisor
super_graph.set_entry_point("Supervisor")

# 그래프 컴파일
super_graph = super_graph.compile(checkpointer=MemorySaver())

# %% [markdown]
# ### Super-Graph 시각화
#
# 완성된 Super-Graph 구조를 시각화합니다. 계층적 구조가 어떻게 구성되어 있는지 확인할 수 있습니다.
#
# 아래 코드는 Super-Graph를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

# Super-Graph 시각화
visualize_graph(super_graph)

# %%
# Super-Graph 실행: Multi-Agent 구조에 대한 논문 작성 요청
output = run_graph(
    super_graph,
    """주제: multi-agent 구조를 사용하여 복잡한 작업을 수행하는 방법

상세 가이드라인:  
- 주제에 대한 Arxiv 논문 형식의 리포트 생성
- Outline 생성
- 각각의 Outline 에 대해서 5문장 이상 작성
- 상세내용 작성시 만약 chart 가 필요하면 차트 생성 및 추가
- 한글로 리포트 작성
- 출처는 APA 형식으로 작성
- 최종 결과는 .md 파일로 저장""",
    recursive_limit=150,
)

# %% [markdown]
# ### 결과 출력
#
# 마크다운 형식으로 최종 결과물을 출력합니다.
#
# 아래 코드는 실행 결과를 마크다운으로 렌더링합니다.

# %%
from IPython.display import Markdown

# 마크다운 형식으로 결과 출력
if hasattr(output["messages"][-1], "content"):
    display(Markdown(output["messages"][-1].content))
else:
    display(Markdown(output["messages"][-1]))

# %%
# 원본 메시지 출력
print(output["messages"][-1])
