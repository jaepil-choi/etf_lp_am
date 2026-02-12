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
# # LangGraph V1.0 Quickstart
#
# 이 튜토리얼은 LangGraph V1.0과 Ollama를 사용하여 간단한 설정부터 완전히 작동하는 AI 에이전트까지 단계별로 안내합니다. Ollama는 로컬 환경에서 LLM을 실행할 수 있는 도구로, 클라우드 API 없이도 AI 애플리케이션을 개발할 수 있습니다.
#
# > 📖 **참고 문서**: [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api.md)

# %% [markdown]
# 환경 변수를 설정해야 원활하게 동작합니다.
# 이를 설정하기 위해서는 `.env` 에 키를 추가해야 합니다.

# %%
from dotenv import load_dotenv
from langchain_teddynote import logging

# 환경 변수 로드
load_dotenv(override=True)
# 추적을 위한 프로젝트 이름 설정
logging.langsmith("LangChain-V1-Tutorial")

# %% [markdown]
# ## 모델 이름 지정
#
# 모델 이름을 지정할 때 다음 형식을 사용할 수 있습니다:
#
# ### 기본 형식
#
# 단순히 모델 이름만 지정:
# * `'o3-mini'`
# * `'claude-sonnet-4-5'`
#
# ### 통합 형식
#
# 모델 제공자와 모델을 함께 지정할 수 있습니다:
#
# ```
# '{model_provider}:{model}'
# ```
#
# **예시:**
# * `'openai:o1'`
# * `'anthropic:claude-sonnet-4-5'`
#
# 이 형식을 사용하면 하나의 인자로 모델 제공자와 모델을 동시에 명시할 수 있습니다.
#
# **주요 파라미터**
#
# * **temperature**: 출력의 무작위성을 조절하는 모델 온도 값
# * **max_tokens**: 생성할 최대 토큰 수
# * **timeout**: 응답 대기 최대 시간 (초 단위)
# * **max_retries**: 요청 실패 시 최대 재시도 횟수
# * **base_url**: 커스텀 API endpoint URL
# * **rate_limiter**: 요청 속도를 제어하는 BaseRateLimiter 인스턴스
#
# ### 사용 예시
#
# ```python
# model_kwargs = {
#     "temperature": 0.7,
#     "max_tokens": 1000,
#     "timeout": 30
# }
# ```
#
# > **참고**: 사용 가능한 전체 파라미터 목록은 각 모델 제공자의 integration reference를 참조하세요.
#
# - [공식문서](https://reference.langchain.com/python/langchain/models/?_gl=1*kundig*_gcl_au*MjAwMTM0Mzc1Mi4xNzYxNDEwNDky*_ga*MTI0ODcwNDIuMTc2MTgwNjA5Mg..*_ga_47WX3HKKY2*czE3NjE4MDYwNzUkbzUkZzEkdDE3NjE4MDYxMjEkajE0JGwwJGgw#langchain.chat_models.init_chat_model)

# %%
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0,
)

# %%
from langchain_teddynote.messages import stream_response

result = llm.stream("반가워")
# 스트리밍 출력
stream_response(result)

# %% [markdown]
# ## 에이전트 생성
#
# LangGraph 기반의 에이전트를 사용합니다. 과거 `create_react_agent` 대신 `create_agent` 를 사용합니다.

# %%
from langchain.agents import create_agent

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0,
)

# 모델 식별자 문자열을 사용한 간단한 방법
agent = create_agent(llm, tools=[])

# %% [markdown]
# ### 그래프 시각화
#
# `langchain_teddynote` 패키지의 `visualize_graph` 함수를 사용하여 에이전트의 내부 구조를 시각화합니다. 이를 통해 노드와 엣지의 연결 상태를 한눈에 파악할 수 있습니다.
#
# 아래 코드는 생성된 에이전트 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(agent)

# %% [markdown]
# ### 메시지 출력
#
# `stream_graph` 함수를 사용하면 에이전트의 실행 결과를 스트리밍 방식으로 출력할 수 있습니다. 각 노드에서 처리된 결과가 실시간으로 표시됩니다.
#
# 아래 코드는 사용자 메시지를 입력하고 에이전트의 응답을 스트리밍합니다.

# %%
from langchain_teddynote.messages import stream_graph
from langchain_core.messages import HumanMessage

stream_graph(agent, inputs={"messages": [HumanMessage(content="안녕하세요?")]})

# %% [markdown]
# ## 기본 에이전트 구축
#
# 질문에 답하고 도구를 호출할 수 있는 간단한 에이전트를 만듭니다. 
#
# 기본 날씨 함수(실제로 기능이 있는 도구는 아닙니다!) 를 도구로 사용하며, 간단한 프롬프트로 동작을 안내합니다.

# %%
from langchain.tools import tool
from langchain.agents import create_agent


# 날씨 정보를 반환하는 간단한 함수
@tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 반환합니다."""
    return f"It's always sunny in {city}!"


# 에이전트 생성
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# %% [markdown]
# ### 그래프 시각화
#
# 도구가 연결된 에이전트의 구조를 시각화합니다. `model` 노드와 `tools` 노드 간의 연결 관계를 확인할 수 있습니다.
#
# 아래 코드는 도구가 연결된 에이전트 그래프를 시각화합니다.

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(agent)

# %% [markdown]
# ### 스트리밍 답변 출력
#
# 도구가 연결된 에이전트에 질문을 보내고 응답을 확인합니다. LLM이 도구 호출이 필요하다고 판단하면 자동으로 `tools` 노드를 거쳐 결과를 반환합니다.
#
# 아래 코드는 날씨 질문을 에이전트에 전달하고 스트리밍 응답을 출력합니다.

# %%
# 에이전트 실행
stream_graph(agent, inputs={"messages": [HumanMessage(content="서울 날씨가 어때?")]})

# %% [markdown]
# ## 도구(Tool)
#
# 도구를 사용하면 모델이 정의한 함수를 호출하여 외부 시스템과 상호작용할 수 있습니다. 
#
# 도구는 런타임 컨텍스트에 의존할 수 있으며 에이전트 메모리와 상호작용할 수도 있습니다.
#
# ### 컨택스트(Context)
#
# 컨택스트는 도구에 전달되는 추가 정보를 제공합니다. 
#
# `runtime.context` 를 통해 컨택스트에 접근할 수 있습니다.
#
# ```python
# runtime.context.user_id
# ```

# %%
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "teddy": {
        "name": "Teddy Lee",
        "account_type": "Premium",
        "balance": 5000,
        "email": "teddy@example.com",
    },
    "shirley": {
        "name": "Shirley Kim",
        "account_type": "Standard",
        "balance": 1200,
        "email": "shirley@example.com",
    },
}


@dataclass
class UserContext:
    user_id: str


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """현재 사용자의 계좌 정보를 조회합니다."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOllama(model="gpt-oss:120b-cloud")

agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

# %%
from langchain_teddynote.messages import stream_graph
from langchain_core.messages import HumanMessage

stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="내 계좌의 현재 잔고를 알려주세요.")]},
    context=UserContext(user_id="teddy"),
)

# %% [markdown]
# ## 응답 형식(Response Format)
#
# 에이전트 응답이 특정 스키마와 일치하도록 구조화된 응답 형식을 정의합니다.
#
# 참고: dataclass 또는 pydantic 모델을 사용하여 응답 형식을 정의할 수 있습니다.

# %%
from pydantic import BaseModel, Field


class ResponseFormat(BaseModel):
    """에이전트 응답 스키마"""

    email_sender: str = Field(description="이메일 발신자")
    email_sender_address: str = Field(description="발신자 주소")


# %%
# 모든 구성요소를 포함한 에이전트 생성
agent = create_agent(
    model=llm,
    system_prompt="Extract useful information from the email.",
    tools=[],
    response_format=ResponseFormat,
)

sample_input = """From: 김철수 (chulsoo.kim@bikecorporation.me)
Subject: "ZENESIS" 자전거 유통 협력 및 미팅 일정 제안

안녕하세요, 이은채 대리님,

저는 바이크코퍼레이션의 김철수 상무입니다. 최근 보도자료를 통해 귀사의 신규 자전거 "ZENESIS"에 대해 알게 되었습니다. 바이크코퍼레이션은 자전거 제조 및 유통 분야에서 혁신과 품질을 선도하는 기업으로, 이 분야에서의 장기적인 경험과 전문성을 가지고 있습니다.

ZENESIS 모델에 대한 상세한 브로슈어를 요청드립니다. 특히 기술 사양, 배터리 성능, 그리고 디자인 측면에 대한 정보가 필요합니다. 이를 통해 저희가 제안할 유통 전략과 마케팅 계획을 보다 구체화할 수 있을 것입니다.

또한, 협력 가능성을 더 깊이 논의하기 위해 다음 주 화요일(1월 15일) 오전 10시에 미팅을 제안합니다. 귀사 사무실에서 만나 이야기를 나눌 수 있을까요?

감사합니다.

김철수
상무이사
바이크코퍼레이션
"""

# 첫 번째 질문: 날씨 문의
response = agent.invoke(
    {"messages": [HumanMessage(content=sample_input)]},
)
print(response["messages"][-1].content)
print("===" * 10)
print(response["structured_response"])

# %% [markdown]
# ## 단기 메모리 추가
#
# 에이전트에 메모리를 추가하여 상호작용 간에 상태를 유지합니다. 이를 통해 에이전트는 이전 대화와 컨텍스트를 기억할 수 있습니다.
#
# 단기 기억의 유지의 범위는 `thread_id` 로 관리 합니다. 즉, 동일한 `thread_id` 는 동일한 메모리를 공유합니다.
#
# 참고: 프로덕션 환경에서는 데이터베이스에 저장하는 영구 체크포인터를 사용하세요.

# %%
from langgraph.checkpoint.memory import InMemorySaver

# 메모리 체크포인터 생성
checkpointer = InMemorySaver()

# %%
# 모든 구성요소를 포함한 에이전트 생성
agent = create_agent(
    model=llm,
    checkpointer=checkpointer,
)

# thread_id는 특정 대화의 고유 식별자입니다.
config = {"configurable": {"thread_id": "1"}}


stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="안녕, 내 이름은 테디야")]},
    config=config,
)

# %%
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="내 이름이 뭔지 기억나?")]},
    config=config,
)

# %%
stream_graph(
    agent,
    inputs={"messages": [HumanMessage(content="내 이름이 뭔지 기억나?")]},
    config={"configurable": {"thread_id": "2"}},
)

# %% [markdown]
# ## 미들웨어(Middleware)
#
# 미들웨어는 에이전트 실행의 모든 단계를 제어하고 커스터마이징하는 방법을 제공합니다.
#
# 핵심 에이전트 루프는 모델을 호출하고, 모델이 실행할 도구를 선택하도록 한 다음, 더 이상 도구를 호출하지 않으면 종료하는 것을 포함합니다.
#
# ![](./assets/langgraph-middleware.avif)
#
# 미들웨어는 각 단계 전후에 후크를 노출합니다.
#
# - 에이전트 시작 전/후
# - 모델 호출 전/후
# - 도구 실행 전/후

# %% [markdown]
# ## Human in the Loop Middleware
#
# ### 개요
#
# Human in the Loop Middleware는 AI 시스템의 의사결정 과정에 사람의 개입을 가능하게 하는 중간 계층입니다. 자동화된 프로세스 중 특정 시점에서 사람의 검토, 승인 또는 수정을 요구할 수 있습니다.
#
# ### 주요 특징
#
# * **검증 단계 추가**: AI의 출력을 사람이 검토하고 승인하는 단계 삽입
# * **오류 방지**: 중요한 결정에 대한 사람의 최종 확인으로 오류 최소화
# * **유연한 개입**: 필요에 따라 자동/수동 모드 전환 가능
# * **피드백 루프**: 사람의 수정 사항을 학습 데이터로 활용
#
# ### Parameters
#
# **`timeout`**
# * **타입**: `int` 또는 `float`
# * **기본값**: `None`
# * **설명**: 사람의 응답을 기다리는 최대 시간(초)
# * **사용법**: timeout 초과 시 기본 동작 실행 또는 예외 발생
# ```python
# middleware = HumanInTheLoopMiddleware(timeout=300)  # 5분
# ```
#
# **`approval_required`**
# * **타입**: `bool`
# * **기본값**: `True`
# * **설명**: 사람의 명시적 승인이 필요한지 여부
# * **사용법**: `False`로 설정 시 검토만 하고 자동 진행
# ```python
# middleware = HumanInTheLoopMiddleware(approval_required=True)
# ```
#
# **`callback_function`**
# * **타입**: `callable`
# * **기본값**: `None`
# * **설명**: 사람의 개입이 필요할 때 호출되는 함수
# * **사용법**: 알림, 로깅, UI 표시 등의 커스텀 동작 정의
# ```python
# def notify_user(data):
#     print(f"Review needed: {data}")
#
# middleware = HumanInTheLoopMiddleware(callback_function=notify_user)
# ```
#
# **`intervention_condition`**
# * **타입**: `callable` 또는 `str`
# * **기본값**: `"always"`
# * **설명**: 사람 개입이 필요한 조건 정의
# * **사용법**: 함수 또는 조건 문자열로 지정
# ```python
# # 함수로 조건 정의
# def check_confidence(result):
#     return result.confidence < 0.8
#
# middleware = HumanInTheLoopMiddleware(intervention_condition=check_confidence)
#
# # 문자열로 조건 정의
# middleware = HumanInTheLoopMiddleware(intervention_condition="low_confidence")
# ```
#
# **`retry_limit`**
# * **타입**: `int`
# * **기본값**: `3`
# * **설명**: 사람의 응답을 요청하는 최대 재시도 횟수
# * **사용법**: 응답이 없을 때 재시도 횟수 제한
# ```python
# middleware = HumanInTheLoopMiddleware(retry_limit=5)
# ```
#
# **`fallback_action`**
# * **타입**: `str` 또는 `callable`
# * **기본값**: `"reject"`
# * **설명**: timeout 또는 응답 실패 시 수행할 동작
# * **옵션**: `"approve"`, `"reject"`, `"skip"`, 또는 커스텀 함수
# * **사용법**:
# ```python
# middleware = HumanInTheLoopMiddleware(fallback_action="approve")
#
# # 커스텀 fallback
# def custom_fallback(context):
#     return context.get("default_value")
#
# middleware = HumanInTheLoopMiddleware(fallback_action=custom_fallback)
# ```
#
# **`notification_channels`**
# * **타입**: `list`
# * **기본값**: `["console"]`
# * **설명**: 알림을 전송할 채널 목록
# * **옵션**: `"console"`, `"email"`, `"slack"`, `"webhook"` 등
# * **사용법**:
# ```python
# middleware = HumanInTheLoopMiddleware(
#     notification_channels=["email", "slack"]
# )
# ```
#
# **`store_feedback`**
# * **타입**: `bool`
# * **기본값**: `True`
# * **설명**: 사람의 피드백을 저장할지 여부
# * **사용법**: 학습 데이터로 활용하기 위해 피드백 저장
# ```python
# middleware = HumanInTheLoopMiddleware(store_feedback=True)
# ```
#
# **`priority_level`**
# * **타입**: `str` 또는 `int`
# * **기본값**: `"normal"`
# * **설명**: 개입 요청의 우선순위
# * **옵션**: `"low"`, `"normal"`, `"high"`, `"critical"` 또는 1-5
# * **사용법**:
# ```python
# middleware = HumanInTheLoopMiddleware(priority_level="high")
# ```

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.tools import tool


@tool
def search_tool(query: str) -> str:
    """정보를 검색합니다."""
    return f"Search results for: {query}"


@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """이메일을 전송합니다. 민감한 작업입니다."""
    return f"Email sent to {recipient}"


@tool
def delete_database_tool(database_name: str) -> str:
    """데이터베이스를 삭제합니다. 중요한 작업입니다."""
    return f"Database {database_name} deleted"


agent = create_agent(
    model=llm,
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 민감한 작업에 대해 승인 필요
                "send_email_tool": True,
                "delete_database_tool": True,
                # 안전한 작업은 자동 승인
                "search_tool": False,
            }
        ),
    ],
    checkpointer=InMemorySaver(),  # 상태 지속성 필요
)

# thread_id 필요
config = {"configurable": {"thread_id": "123"}}

# %%
from langchain_teddynote.messages import invoke_graph

invoke_graph(
    agent,
    inputs={
        "messages": [
            HumanMessage(
                content="teddy@example.com 에게 메일을 보내 주세요. 제목은 '테스트' 이고 내용은 '안녕하세요' 입니다."
            )
        ]
    },
    config=config,
)

# %%
# interrupt 확인
print(agent.get_state(config).interrupts[0].value["action_requests"][0]["description"])

# %%
# decisions: approve, reject, skip
stream_graph(
    agent, inputs=Command(resume={"decisions": [{"type": "approve"}]}), config=config
)
