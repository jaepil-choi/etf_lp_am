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
# # LangChain 가드레일
#
# 가드레일(Guardrails)은 에이전트 실행 중 주요 지점에서 콘텐츠를 검증하고 필터링하여 안전하고 규정을 준수하는 AI 애플리케이션을 구축할 수 있도록 도와줍니다. 가드레일을 통해 부적절한 입력을 차단하고, 민감한 정보를 보호하며, 출력 품질을 보장할 수 있습니다.
#
# ## 주요 사용 사례
#
# 가드레일은 다음과 같은 상황에서 사용됩니다:
#
# - 개인정보(PII) 유출 방지
# - 프롬프트 인젝션 공격 탐지 및 차단
# - 부적절하거나 유해한 콘텐츠 차단
# - 비즈니스 규칙 및 규정 준수 요구사항 강제
# - 출력 품질 및 정확성 검증
#
# 가드레일은 미들웨어를 사용하여 구현되며, 에이전트 시작 전, 완료 후, 또는 모델 및 도구 호출 주변에서 실행을 가로챌 수 있습니다. 이 튜토리얼에서는 다양한 가드레일 패턴과 구현 방법을 학습합니다.

# %% [markdown]
# ## 사전 준비
#
# 가드레일 기능을 사용하기 위해 환경 변수와 LangSmith 추적을 설정합니다. 환경 변수에는 LLM 서비스 인증 정보가 포함되며, LangSmith를 통해 가드레일이 트리거된 상황을 모니터링할 수 있습니다.
#
# 아래 코드는 `.env` 파일에서 환경 변수를 로드하고, LangSmith 추적을 활성화합니다.

# %%
from dotenv import load_dotenv
from langchain_teddynote import logging

# 환경 변수 로드
load_dotenv(override=True)

# LangSmith 추적 활성화
logging.langsmith("LangChain-V1-Tutorial")

# %% [markdown]
# ## 가드레일의 두 가지 접근 방식
#
# 가드레일은 구현 방식에 따라 크게 두 가지로 나눌 수 있습니다. 각 접근 방식은 장단점이 있으며, 실제 애플리케이션에서는 두 가지를 조합하여 사용하는 것이 일반적입니다.
#
# ### 1. 결정론적 가드레일 (Deterministic)
#
# 정규 표현식, 키워드 매칭, 명시적 검사 등 규칙 기반 로직을 사용합니다. 빠르고 예측 가능하며 비용 효율적이지만, 미묘한 위반 사항을 놓칠 수 있습니다. 금지어 필터링, PII 패턴 탐지 등에 적합합니다.
#
# ### 2. 모델 기반 가드레일 (Model-based)
#
# LLM 또는 분류기를 사용하여 의미론적 이해로 콘텐츠를 평가합니다. 규칙이 놓치는 미묘한 문제를 포착할 수 있지만, 느리고 비용이 더 많이 듭니다. 감정 분석, 안전성 평가, 품질 검증 등에 적합합니다.

# %% [markdown]
# ## 내장 가드레일
#
# LangChain은 자주 사용되는 가드레일 패턴을 내장 미들웨어로 제공합니다. 별도의 구현 없이 바로 사용할 수 있어 빠르게 보안 기능을 추가할 수 있습니다.
#
# ### PII 탐지
#
# 개인 식별 정보(PII)를 대화에서 탐지하고 처리하기 위한 `PIIMiddleware`를 제공합니다. 이메일, 신용카드 번호, 전화번호 등 다양한 PII 유형을 지원하며, 다음과 같은 처리 전략을 제공합니다:
#
# - **`redact`**: `[REDACTED_TYPE]`으로 대체하여 완전히 제거
# - **`mask`**: 부분적으로 가림 (예: 신용카드 마지막 4자리만 표시)
# - **`hash`**: 결정론적 해시로 대체하여 추적 가능성 유지
# - **`block`**: 탐지 시 예외를 발생시켜 처리 중단
#
# 아래 코드는 사용자 입력에서 이메일을 수정하고 신용카드를 마스킹하는 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def customer_service_tool(query: str) -> str:
    """Handle customer service queries."""
    return f"Processing customer query: {query}"

model = ChatOpenAI(model="gpt-4.1-mini")

agent = create_agent(
    model=model,
    tools=[customer_service_tool],
    middleware=[
        # 사용자 입력에서 이메일 수정
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # 사용자 입력에서 신용카드 마스킹
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
    ],
)

# PII가 포함된 메시지 테스트
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "My email is john.doe@example.com and card is 4532-1234-5678-9010"
    }]
})

print(result["messages"][-1].content)

# %% [markdown]
# ### 커스텀 PII 탐지기
#
# 내장 PII 타입 외에도 정규 표현식을 사용하여 커스텀 PII 패턴을 정의할 수 있습니다. API 키, 특정 형식의 ID, 지역별 전화번호 패턴 등 비즈니스 요구사항에 맞는 패턴을 추가할 수 있습니다.
#
# 아래 코드는 API 키 패턴과 한국 전화번호 패턴을 탐지하는 커스텀 PII 미들웨어를 생성합니다.

# %%
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=model,
    tools=[customer_service_tool],
    middleware=[
        # API 키 탐지 - 발견 시 에러 발생
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",  # 커스텀 정규 표현식
            strategy="block",
            apply_to_input=True,
        ),
        # 한국 전화번호 패턴 탐지
        PIIMiddleware(
            "phone_number",
            detector=r"010-\d{4}-\d{4}",
            strategy="redact",
            apply_to_input=True,
        ),
    ],
)

# 전화번호가 포함된 메시지 테스트
result = agent.invoke({
    "messages": [{"role": "user", "content": "My phone is 010-1234-5678"}]
})

print(result["messages"][-1].content)

# %% [markdown]
# ### 내장 PII 타입
#
# LangChain은 자주 사용되는 PII 패턴을 내장 타입으로 제공합니다. 이러한 타입은 검증된 패턴을 사용하므로 신뢰성이 높습니다:
#
# - `email`: 이메일 주소
# - `credit_card`: 신용카드 번호 (Luhn 알고리즘 검증 포함)
# - `ip`: IP 주소 (IPv4, IPv6)
# - `mac_address`: MAC 주소
# - `url`: URL
#
# 아래 코드는 여러 내장 PII 타입을 동시에 적용하는 에이전트를 생성합니다.

# %%
# 여러 PII 타입 동시 적용
agent = create_agent(
    model=model,
    tools=[customer_service_tool],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("ip", strategy="hash", apply_to_input=True),
        PIIMiddleware("url", strategy="redact", apply_to_input=True),
    ],
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Contact me at user@test.com from IP 192.168.1.1 or visit https://example.com"
    }]
})

print(result["messages"][-1].content)

# %% [markdown]
# ## Human-in-the-Loop
#
# 민감한 작업을 실행하기 전에 사람의 승인을 요구하는 것은 가장 효과적인 가드레일 중 하나입니다. `HumanInTheLoopMiddleware`를 사용하면 특정 도구 호출 전에 실행을 일시 중지하고 사람의 결정을 기다릴 수 있습니다. 이 기능에 대한 자세한 내용은 Human-in-the-Loop 튜토리얼을 참조하세요.
#
# **Human-in-the-Loop이 유용한 경우:**
# - 금융 거래 및 송금
# - 프로덕션 데이터 삭제 또는 수정
# - 외부 당사자에게 통신 전송
# - 비즈니스에 중요한 영향을 미치는 모든 작업
#
# 아래 코드는 이메일 전송과 데이터베이스 삭제 작업에 대해 사람의 승인을 요구하는 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email. This is a sensitive operation."""
    return f"Email sent to {recipient}"

@tool
def delete_database_tool(database_name: str) -> str:
    """Delete a database. This is a critical operation."""
    return f"Database {database_name} deleted"

agent = create_agent(
    model=model,
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
config = {"configurable": {"thread_id": "some_id"}}

# 이메일 전송 전 일시 중지됨
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config
)

print("Agent paused for approval. Approving...")

# 승인 후 재개
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # 동일한 thread_id로 재개
)

print(result["messages"][-1].content)

# %% [markdown]
# ## 커스텀 가드레일
#
# 내장 가드레일로 충분하지 않은 경우 에이전트 실행 전후에 실행되는 커스텀 미들웨어를 생성할 수 있습니다. `@before_agent`, `@after_agent` 데코레이터 또는 `AgentMiddleware` 클래스를 사용하여 비즈니스 요구사항에 맞는 가드레일을 구현합니다.

# %% [markdown]
# ### Before Agent 가드레일
#
# `@before_agent` 데코레이터를 사용하면 각 호출 시작 시 요청을 검증할 수 있습니다. 인증 확인, 속도 제한, 금지어 필터링 등 처리가 시작되기 전에 부적절한 요청을 차단하는 데 유용합니다. `can_jump_to=["end"]` 옵션을 사용하면 가드레일이 트리거될 때 에이전트 실행을 즉시 종료할 수 있습니다.
#
# 아래 코드는 금지된 키워드가 포함된 요청을 처리 전에 차단하는 결정론적 가드레일을 구현합니다.

# %%
from typing import Any
from langchain.agents.middleware import before_agent, AgentState, hook_config
from langgraph.runtime import Runtime

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """결정론적 가드레일: 금지된 키워드가 포함된 요청 차단"""
    banned_keywords = ["hack", "exploit", "malware", "해킹", "악성코드"]
    
    # 첫 번째 사용자 메시지 가져오기
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    # 금지된 키워드 확인
    for keyword in banned_keywords:
        if keyword in content:
            # 처리 전에 실행 차단
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "부적절한 콘텐츠가 포함된 요청은 처리할 수 없습니다. 요청을 다시 작성해주세요."
                }],
                "jump_to": "end"
            }

    return None

# 커스텀 가드레일 사용
agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[content_filter],
)

# 이 요청은 처리 전에 차단됨
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})

print(result["messages"][-1].content)

# %% [markdown]
# ### 클래스 기반 Before Agent 가드레일
#
# 더 복잡한 로직이 필요하거나 설정 가능한 매개변수가 있는 경우 `AgentMiddleware` 클래스를 상속받아 구현할 수 있습니다. 클래스 기반 접근 방식은 상태를 유지하거나 여러 후크를 함께 사용해야 할 때 유용합니다.
#
# 아래 코드는 금지어 목록을 매개변수로 받는 재사용 가능한 `ContentFilterMiddleware` 클래스를 구현합니다.

# %%
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from typing import Any

class ContentFilterMiddleware(AgentMiddleware):
    """결정론적 가드레일: 금지된 키워드가 포함된 요청 차단"""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "부적절한 콘텐츠가 포함된 요청은 처리할 수 없습니다."
                    }],
                    "jump_to": "end"
                }

        return None

agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware", "해킹"]
        ),
    ],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "데이터베이스를 해킹하는 방법은?"}]
})

print(result["messages"][-1].content)

# %% [markdown]
# ### After Agent 가드레일
#
# `@after_agent` 데코레이터를 사용하면 사용자에게 반환하기 전에 최종 출력을 검증할 수 있습니다. 모델 기반 안전성 검사, 품질 검증, 규정 준수 스캔 등 전체 에이전트 응답을 평가하는 데 유용합니다. 입력 검증보다 비용이 높을 수 있지만, 더 정교한 검사가 가능합니다.
#
# 아래 코드는 LLM을 사용하여 에이전트 응답의 안전성을 평가하는 모델 기반 가드레일을 구현합니다.

# %%
from langchain.agents.middleware import after_agent, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

safety_model = init_chat_model("openai:gpt-4.1-mini")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """모델 기반 가드레일: LLM을 사용하여 응답 안전성 평가"""
    # 최종 AI 응답 가져오기
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # 모델을 사용하여 안전성 평가
    safety_prompt = f"""이 응답이 안전하고 적절한지 평가하세요.
    'SAFE' 또는 'UNSAFE'로만 응답하세요.

    응답: {last_message.content}"""

    result = safety_model.invoke([{"role": "user", "content": safety_prompt}])

    if "UNSAFE" in result.content:
        return {
            "messages": [{
                "role": "assistant",
                "content": "해당 응답을 제공할 수 없습니다. 요청을 다시 작성해주세요."
            }],
            "jump_to": "end"
        }

    return None

# 안전 가드레일 사용
agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[safety_guardrail],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Tell me about Python programming"}]
})

print(result["messages"][-1].content)

# %% [markdown]
# ## 여러 가드레일 결합
#
# 미들웨어 배열에 여러 가드레일을 추가하여 계층화된 보호(Defense in Depth)를 구축할 수 있습니다. 가드레일은 배열 순서대로 실행되며, 빠른 결정론적 검사를 먼저, 느린 모델 기반 검사를 나중에 배치하면 불필요한 비용을 줄일 수 있습니다.
#
# 아래 코드는 입력 필터, PII 보호, Human-in-the-Loop, 안전성 검사를 계층화하여 적용하는 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[search_tool, send_email_tool],
    middleware=[
        # 계층 1: 결정론적 입력 필터 (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # 계층 2: PII 보호 (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # 계층 3: 민감한 도구에 대한 사람의 승인
        HumanInTheLoopMiddleware(interrupt_on={"send_email_tool": True}),

        # 계층 4: 모델 기반 안전성 검사 (after agent)
        safety_guardrail,
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "secure_thread"}}

# 모든 계층의 가드레일을 통과해야 함
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to team@example.com"}]},
    config=config
)

print(result["messages"][-1].content)

# %% [markdown]
# ## 종합 예제: 보안 고객 서비스 에이전트
#
# 이 섹션에서는 지금까지 학습한 가드레일 기술을 결합하여 실용적인 고객 서비스 에이전트를 구현합니다. 속도 제한, 콘텐츠 필터링, PII 보호, 출력 품질 검증을 모두 적용하여 안전하고 신뢰할 수 있는 에이전트를 구축합니다.
#
# 아래 코드는 고객 조회, 환불 처리, 알림 전송 기능을 가진 보안 고객 서비스 에이전트를 구현합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, AgentMiddleware, hook_config
from langchain.tools import tool
from typing import Any

@tool
def lookup_customer(customer_id: str) -> str:
    """Look up customer information."""
    return f"Customer {customer_id}: John Doe, Premium member"

@tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for an order."""
    return f"Refund of ${amount} processed for order {order_id}"

@tool
def send_notification(customer_id: str, message: str) -> str:
    """Send notification to customer."""
    return f"Notification sent to customer {customer_id}"

# 속도 제한 가드레일
class RateLimitMiddleware(AgentMiddleware):
    def __init__(self, max_requests: int = 10):
        super().__init__()
        self.request_count = 0
        self.max_requests = max_requests

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self.request_count += 1
        if self.request_count > self.max_requests:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
                }],
                "jump_to": "end"
            }
        return None

# 출력 품질 검증 가드레일
@after_agent
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """응답이 충분히 도움이 되는지 확인"""
    if not state["messages"]:
        return None
    
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        # 응답이 너무 짧으면 경고
        if len(last_message.content) < 20:
            print("Warning: Response may be too brief")
    
    return None

# 보안 고객 서비스 에이전트 생성
secure_agent = create_agent(
    model=model,
    tools=[lookup_customer, process_refund, send_notification],
    middleware=[
        # 1. 입력 검증
        RateLimitMiddleware(max_requests=100),
        ContentFilterMiddleware(banned_keywords=["hack", "fraud"]),
        
        # 2. PII 보호
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        
        # 3. 출력 검증
        validate_output,
    ],
)

# 테스트
print("=== Test 1: Normal query ===")
result = secure_agent.invoke({
    "messages": [{"role": "user", "content": "Look up customer CUST123"}]
})
print(result["messages"][-1].content)

print("\n=== Test 2: Query with PII ===")
result = secure_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Customer email is john@example.com and card is 4532-1234-5678-9010"
    }]
})
print(result["messages"][-1].content)

print("\n=== Test 3: Blocked content ===")
result = secure_agent.invoke({
    "messages": [{"role": "user", "content": "How to hack the system?"}]
})
print(result["messages"][-1].content)

# %% [markdown]
# ## 가드레일 설계 모범 사례
#
# 효과적인 가드레일 시스템을 구축하기 위한 핵심 원칙을 정리합니다. 이러한 원칙을 따르면 안전하면서도 사용자 경험을 해치지 않는 가드레일을 구현할 수 있습니다.
#
# ### 1. 계층화된 방어 (Defense in Depth)
#
# 여러 가드레일을 결합하여 다층 보호를 구현합니다. 한 계층이 실패해도 다른 계층이 보호할 수 있도록 설계합니다. 빠른 규칙 기반 필터를 먼저 배치하고, 느리지만 정확한 모델 기반 검증을 나중에 배치합니다.
#
# 아래 코드는 다층 보호를 적용하는 모범 사례 예제입니다.

# %%
# 좋은 예: 다층 보호
agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[
        # 입력 단계: 빠른 규칙 기반 필터
        ContentFilterMiddleware(banned_keywords=["hack"]),
        PIIMiddleware("email", strategy="redact"),
        
        # 출력 단계: 느리지만 정확한 모델 기반 검증
        safety_guardrail,
    ],
)

# %% [markdown]
# ### 2. 성능 고려사항
#
# 가드레일 순서는 성능에 큰 영향을 미칩니다. 빠른 검사(정규 표현식, 키워드 매칭)를 먼저 실행하여 명백한 위반을 조기에 차단하면 느린 검사(LLM 호출)를 실행할 필요가 없어 비용을 절감할 수 있습니다.
#
# 아래 코드는 올바른 가드레일 순서(빠른 검사 우선)와 잘못된 순서(느린 검사 우선)를 비교합니다.

# %%
# 좋은 예: 빠른 검사가 먼저
agent = create_agent(
    model=model,
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(banned_keywords=["hack"]),  # 빠름
        safety_guardrail,  # 느림 (LLM 호출)
    ],
)

# 나쁜 예: 느린 검사가 먼저
# agent = create_agent(
#     model=model,
#     tools=[search_tool],
#     middleware=[
#         safety_guardrail,  # 느림 - 모든 요청에 대해 실행됨
#         ContentFilterMiddleware(banned_keywords=["hack"]),  # 빠름
#     ],
# )

# %% [markdown]
# ### 3. 명확한 에러 메시지
#
# 사용자에게 요청이 차단된 이유를 명확하게 알려주세요. 모호한 에러 메시지는 사용자 경험을 저해하고, 같은 문제가 반복될 수 있습니다. 가능하면 대안이나 다음 단계도 안내합니다.
#
# 아래 코드는 구체적인 피드백을 제공하는 가드레일 예제입니다.

# %%
@before_agent(can_jump_to=["end"])
def clear_error_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """명확한 에러 메시지를 제공하는 가드레일"""
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    # 구체적인 피드백 제공
    if "password" in content or "비밀번호" in content:
        return {
            "messages": [{
                "role": "assistant",
                "content": "보안상의 이유로 비밀번호 관련 질문은 처리할 수 없습니다. 계정 설정에서 비밀번호를 재설정해주세요."
            }],
            "jump_to": "end"
        }

    return None


# %% [markdown]
# ### 4. 로깅 및 모니터링
#
# 가드레일이 트리거될 때 로그를 남겨 보안 위협을 추적하고 분석하세요. 로그에는 타임스탬프, 사용자 정보, 차단 사유, 원본 콘텐츠(민감 정보 제외) 등을 포함합니다. 이 데이터를 기반으로 가드레일 정책을 지속적으로 개선할 수 있습니다.
#
# 아래 코드는 보안 이벤트를 로깅하는 가드레일 예제입니다.

# %%
import logging

logger = logging.getLogger(__name__)

@before_agent(can_jump_to=["end"])
def monitored_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """로깅이 포함된 가드레일"""
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    if "hack" in content:
        # 보안 이벤트 로깅
        logger.warning(f"Security violation detected: {content[:50]}...")
        
        return {
            "messages": [{
                "role": "assistant",
                "content": "요청이 보안 정책에 위배됩니다."
            }],
            "jump_to": "end"
        }

    return None
