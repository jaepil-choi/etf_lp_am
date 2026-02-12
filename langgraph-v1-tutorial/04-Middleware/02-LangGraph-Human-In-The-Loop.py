# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: langgraph-v1-tutorial
#     language: python
#     name: langgraph-v1-tutorial
# ---

# %% [markdown]
# # Human-in-the-Loop
#
# Human-in-the-Loop(HITL)은 에이전트가 특정 작업을 수행하기 전에 사람의 승인을 받도록 하는 메커니즘입니다. 이를 통해 AI 에이전트의 자율성과 인간의 감독 사이에서 적절한 균형을 유지할 수 있습니다. 파일 작성, 데이터베이스 수정, 이메일 전송 등 민감한 작업에서 사람이 최종 결정권을 가지게 됩니다.
#
# ## 작동 원리
#
# HITL 미들웨어는 다음과 같은 흐름으로 동작합니다:
#
# 1. 모델이 도구 호출을 생성
# 2. 미들웨어가 도구 호출을 검사
# 3. 사람의 승인이 필요한 경우 **interrupt** 발생
# 4. 그래프 상태가 저장되고 실행 일시 중지
# 5. 사람의 결정을 받아 실행 재개
#
# ## Interrupt 결정 타입
#
# 미들웨어는 세 가지 내장 응답 방식을 정의합니다:
#
# | 결정 타입 | 설명 | 사용 사례 |
# |---------|------|----------|
# | `approve` | 작업을 그대로 승인하고 변경 없이 실행 | 작성된 이메일을 정확히 그대로 전송 |
# | `edit` | 도구 호출을 수정하여 실행 | 이메일 수신자를 변경한 후 전송 |
# | `reject` | 도구 호출을 거부하고 설명 추가 | 이메일 초안을 거부하고 재작성 방법 설명 |
#
# 이 튜토리얼에서는 HITL 미들웨어의 설정 방법과 각 결정 타입의 사용법을 학습합니다.
#
# ## 작동 흐름
# ![hitl-flow](../assets/hitl-flow.png)

# %% [markdown]
# ## 사전 준비
#
# Human-in-the-Loop 기능을 사용하기 위해서는 환경 변수 설정과 LangSmith 추적을 활성화해야 합니다. 환경 변수에는 LLM 서비스 인증 정보가 포함되며, LangSmith 추적을 통해 에이전트의 실행 과정을 상세히 모니터링할 수 있습니다.
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
# ## 기본 예제
#
# HITL 미들웨어를 사용하려면 `HumanInTheLoopMiddleware`를 에이전트에 추가하고, 어떤 도구가 승인을 필요로 하는지 `interrupt_on` 매개변수로 지정합니다. 또한 상태를 저장하고 복원하기 위해 **체크포인터(Checkpointer)**가 필수입니다. 체크포인터가 없으면 interrupt 후 실행을 재개할 수 없습니다.
#
# 아래 코드는 파일 작업 도구들을 정의하고, `write_file`과 `delete_file` 작업에 대해 사람의 승인을 요구하는 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# 도구 정의
@tool
def write_file(filename: str, content: str) -> str:
    """파일에 내용을 작성합니다."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"File {filename} written successfully"

@tool
def read_file(filename: str) -> str:
    """파일에서 내용을 읽어옵니다."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File {filename} not found"

@tool
def delete_file(filename: str) -> str:
    """파일을 삭제합니다."""
    import os
    try:
        os.remove(filename)
        return f"File {filename} deleted successfully"
    except FileNotFoundError:
        return f"File {filename} not found"

# 모델 초기화
# OpenAI 키 사용 시 gpt-4.1-mini, gpt-5.2 등으로 변경하세요.
model = init_chat_model("claude-sonnet-4-5")

# HITL 미들웨어와 함께 에이전트 생성
agent = create_agent(
    model=model,
    tools=[write_file, read_file, delete_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,    # 모든 결정(approve, edit, reject) 허용
                "delete_file": True,   # 모든 결정 허용
                "read_file": False,    # 안전한 작업, 승인 불필요
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),  # 체크포인터 필수
)

print("Agent created with HITL middleware")

# %% [markdown]
# ### Interrupt 발생 및 처리
#
# 에이전트가 승인이 필요한 도구를 호출하면 `__interrupt__` 키가 포함된 결과가 반환됩니다. 이 interrupt 데이터에는 대기 중인 작업 요청(`action_requests`)과 허용된 결정 타입(`review_configs`) 정보가 포함됩니다. 개발자는 이 정보를 사용자에게 표시하고 결정을 받을 수 있습니다.
#
# 아래 코드는 파일 쓰기 요청을 보내고, interrupt가 발생했을 때 작업 정보를 출력하는 예제입니다.

# %%
from langgraph.types import Command

# thread_id를 포함한 config 필수
config = {"configurable": {"thread_id": "thread_001"}}

# 파일 쓰기 요청 (interrupt 발생 예상)
result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Write 'Hello World' to a file called test.txt"
        }]
    },
    config=config
)

# Interrupt 확인
if "__interrupt__" in result:
    print("\n=== Interrupt Detected ===")
    interrupt_data = result["__interrupt__"][0].value
    
    print("\nAction Requests:")
    for action in interrupt_data["action_requests"]:
        print(f"  Tool: {action['name']}")
        print(f"  Args: {action['args']}")
        print(f"  Description: {action['description']}")
    
    print("\nReview Configs:")
    for config_item in interrupt_data["review_configs"]:
        print(f"  Tool: {config_item['action_name']}")
        print(f"  Allowed decisions: {config_item['allowed_decisions']}")
else:
    print("No interrupt occurred")

# %% [markdown]
# ## 결정 타입
#
# HITL에서 사용할 수 있는 세 가지 결정 타입을 실제 코드로 살펴보겠습니다. 각 결정 타입은 `Command(resume=...)` 형태로 에이전트에 전달되며, 동일한 `thread_id`를 사용해야 일시 중지된 실행을 이어서 진행할 수 있습니다.
#
# ### 1. Approve (승인)
#
# `approve` 결정은 에이전트가 제안한 작업을 수정 없이 그대로 실행합니다. 가장 간단한 형태의 승인으로, 도구 호출 인수를 변경할 필요가 없을 때 사용합니다.
#
# 아래 코드는 interrupt된 작업을 승인하고, 실제로 파일이 생성되었는지 확인합니다.

# %%
# 작업 승인
result = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config  # 동일한 thread_id
)

print("Result after approval:")
print(result["messages"][-1].content)

# 파일이 실제로 생성되었는지 확인
import os
if os.path.exists('test.txt'):
    with open('test.txt', 'r') as f:
        print(f"\nFile content: {f.read()}")

# %% [markdown]
# ### 2. Edit (수정)
#
# `edit` 결정은 도구 호출의 인수를 수정한 후 실행합니다. 에이전트가 올바른 도구를 선택했지만 파일명이나 내용 등 세부 사항을 조정해야 할 때 유용합니다. 수정된 인수는 `edited_action` 필드에 지정합니다.
#
# 아래 코드는 원래 `original.txt`로 저장하려던 파일을 `modified.txt`로 변경하고, 내용도 수정하여 실행합니다.

# %%
# 새로운 파일 쓰기 요청
config2 = {"configurable": {"thread_id": "thread_002"}}

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Write 'Original content' to original.txt"
        }]
    },
    config=config2
)

if "__interrupt__" in result:
    print("Interrupt detected - modifying the action")
    
    # 인수를 수정하여 승인
    result = agent.invoke(
        Command(
            resume={
                "decisions": [{
                    "type": "edit",
                    "edited_action": {
                        "name": "write_file",
                        "args": {
                            "filename": "modified.txt",  # 파일명 변경
                            "content": "Modified content"  # 내용 변경
                        }
                    }
                }]
            }
        ),
        config=config2
    )
    
    print("\nResult after edit:")
    print(result["messages"][-1].content)
    
    # 수정된 파일 확인
    if os.path.exists('modified.txt'):
        with open('modified.txt', 'r') as f:
            print(f"\nModified file content: {f.read()}")

# %% [markdown]
# ### 3. Reject (거부)
#
# `reject` 결정은 도구 호출을 거부하고 에이전트에게 피드백을 제공합니다. 거부 사유를 `message` 필드에 포함하면 에이전트가 이 피드백을 기반으로 다른 접근 방식을 시도하거나 사용자에게 설명할 수 있습니다.
#
# 아래 코드는 파일 삭제 요청을 거부하고, 중요한 데이터가 포함되어 있으므로 먼저 백업해야 한다는 피드백을 제공합니다.

# %%
# 파일 삭제 요청
config3 = {"configurable": {"thread_id": "thread_003"}}

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Delete the file test.txt"
        }]
    },
    config=config3
)

if "__interrupt__" in result:
    print("Interrupt detected - rejecting the action")
    
    # 작업 거부 및 피드백 제공
    result = agent.invoke(
        Command(
            resume={
                "decisions": [{
                    "type": "reject",
                    "message": "I cannot delete this file because it contains important data. Please back it up first."
                }]
            }
        ),
        config=config3
    )
    
    print("\nResult after rejection:")
    print(result["messages"][-1].content)


# %% [markdown]
# ## 실용적인 예제: 데이터베이스 관리 에이전트
#
# HITL의 실제 활용 사례로 데이터베이스 관리 에이전트를 구현해보겠습니다. SQL 실행과 같은 위험한 작업에는 승인을 요구하고, 읽기 작업은 자동으로 허용하는 정책을 적용합니다. 특히 SQL 실행은 수정이 불가능하게 설정하여 승인 또는 거부만 가능하도록 합니다.
#
# 아래 코드는 데이터베이스 작업 도구들을 정의하고, 작업별로 다른 승인 정책을 적용하는 에이전트를 생성합니다.

# %%
@tool
def execute_sql(query: str) -> str:
    """데이터베이스에서 SQL 쿼리를 실행합니다."""
    # 실제로는 데이터베이스에 연결하여 실행
    print(f"Executing SQL: {query}")
    return f"Query executed: {query}"

@tool
def read_table(table_name: str) -> str:
    """테이블에서 데이터를 읽어옵니다."""
    return f"Reading data from {table_name}"

@tool
def backup_database() -> str:
    """전체 데이터베이스를 백업합니다."""
    return "Database backup completed"

# 데이터베이스 에이전트 생성
db_agent = create_agent(
    model=model,
    tools=[execute_sql, read_table, backup_database],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # SQL 실행은 승인 또는 거부만 가능 (편집 불가)
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                # 백업은 모든 결정 허용
                "backup_database": True,
                # 읽기는 승인 불필요
                "read_table": False,
            },
            description_prefix="Database operation pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

print("Database agent created")

# %% [markdown]
# ### 안전한 작업 (Interrupt 없음)
#
# `interrupt_on`에서 `False`로 설정된 도구는 interrupt 없이 바로 실행됩니다. 읽기 전용 작업처럼 부작용이 없는 안전한 작업에 적합합니다. 이를 통해 사용자 경험을 방해하지 않으면서도 위험한 작업만 선별적으로 감독할 수 있습니다.
#
# 아래 코드는 테이블 읽기 작업을 요청하고, interrupt 없이 바로 실행되는지 확인합니다.

# %%
config_db1 = {"configurable": {"thread_id": "db_thread_001"}}

# 읽기 작업 - interrupt 발생하지 않음
result = db_agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Read data from users table"
        }]
    },
    config=config_db1
)

print("Result (no interrupt):")
print(result["messages"][-1].content)
print(f"\nInterrupt occurred: {'__interrupt__' in result}")

# %% [markdown]
# ### 위험한 작업 (Interrupt 발생)
#
# SQL 실행과 같은 위험한 작업은 interrupt가 발생하여 사람의 결정을 기다립니다. `allowed_decisions`가 `["approve", "reject"]`로 설정되어 있어 수정(edit)은 불가능합니다. 이는 SQL 쿼리를 임의로 수정하는 것이 더 위험할 수 있기 때문입니다.
#
# 아래 코드는 DELETE SQL 실행 요청을 보내고, 위험한 작업으로 판단하여 거부하는 예제입니다.

# %%
config_db2 = {"configurable": {"thread_id": "db_thread_002"}}

# SQL 실행 - interrupt 발생
result = db_agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Delete all records older than 30 days from the logs table"
        }]
    },
    config=config_db2
)

if "__interrupt__" in result:
    print("=== Interrupt for SQL Execution ===")
    interrupt_data = result["__interrupt__"][0].value
    
    action = interrupt_data["action_requests"][0]
    print(f"\nSQL Query: {action['args']['query']}")
    print(f"\nAllowed decisions: {interrupt_data['review_configs'][0]['allowed_decisions']}")
    
    # 위험한 작업 거부
    result = db_agent.invoke(
        Command(
            resume={
                "decisions": [{
                    "type": "reject",
                    "message": "This DELETE operation is too broad. Please add a LIMIT clause or be more specific about which records to delete."
                }]
            }
        ),
        config=config_db2
    )
    
    print("\nResult after rejection:")
    print(result["messages"][-1].content)


# %% [markdown]
# ## 다중 작업 승인
#
# 에이전트가 여러 도구를 동시에 호출할 경우, 모든 대기 중인 작업에 대해 개별적으로 결정을 제공해야 합니다. 결정 순서는 interrupt 데이터의 `action_requests` 순서와 일치해야 합니다. 각 작업에 대해 승인, 수정, 거부 중 하나를 선택할 수 있습니다.
#
# 아래 코드는 이메일 전송과 회의 예약을 동시에 요청하고, 첫 번째 작업은 승인, 두 번째 작업은 시간을 수정하여 승인하는 예제입니다.

# %%
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """이메일을 전송합니다."""
    return f"Email sent to {recipient} with subject '{subject}'"

@tool
def schedule_meeting(participants: list, time: str) -> str:
    """회의를 예약합니다."""
    return f"Meeting scheduled at {time} with {', '.join(participants)}"

@tool
def create_document(title: str, content: str) -> str:
    """새 문서를 생성합니다."""
    return f"Document '{title}' created"

# 모든 작업에 승인이 필요한 에이전트
multi_agent = create_agent(
    model=model,
    tools=[send_email, schedule_meeting, create_document],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
                "schedule_meeting": True,
                "create_document": True,
            },
        ),
    ],
    checkpointer=InMemorySaver(),
)

config_multi = {"configurable": {"thread_id": "multi_thread_001"}}

# 여러 작업을 동시에 요청
result = multi_agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Send an email using tools to john@example.com about the project update, then schedule a meeting with the team for tomorrow at 2pm"
        }]
    },
    config=config_multi
)

if "__interrupt__" in result:
    print("=== Multiple Actions Require Approval ===")
    interrupt_data = result["__interrupt__"][0].value
    
    print(f"\nNumber of actions: {len(interrupt_data['action_requests'])}")
    for i, action in enumerate(interrupt_data["action_requests"]):
        print(f"\nAction {i+1}:")
        print(f"  Tool: {action['name']}")
        print(f"  Args: {action['args']}")
    
    # 각 작업에 대한 결정 제공
    result = multi_agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"},  # 첫 번째 작업 승인
                    {
                        "type": "edit",  # 두 번째 작업 수정
                        "edited_action": {
                            "name": "schedule_meeting",
                            "args": {
                                "participants": ["john@example.com", "jane@example.com"],
                                "time": "tomorrow at 3pm"  # 시간 변경
                            }
                        }
                    }
                ]
            }
        ),
        config=config_multi
    )
    
    print("\n=== Result After Decisions ===")
    print(result["messages"][-1].content)


# %% [markdown]
# ## 실행 라이프사이클
#
# HITL 미들웨어는 모델이 응답을 생성한 후, 도구 호출이 실행되기 전에 실행되는 `after_model` 훅을 정의합니다. 내부적으로 다음과 같은 과정이 진행됩니다:
#
# 1. 에이전트가 모델을 호출하여 응답 생성
# 2. 미들웨어가 응답에서 도구 호출 검사
# 3. 사람의 입력이 필요한 호출이 있으면 `HITLRequest`를 빌드하고 `interrupt` 호출
# 4. 에이전트가 사람의 결정을 기다림
# 5. `HITLResponse` 결정에 따라 미들웨어가 승인/편집된 호출을 실행하거나 거부된 호출에 대한 `ToolMessage` 합성 후 실행 재개
#
# 이 라이프사이클을 이해하면 더 복잡한 HITL 시나리오를 구현할 수 있습니다.

# %% [markdown]
# ## 고급 예제: 조건부 승인 정책
#
# 실제 프로덕션 환경에서는 작업 내용에 따라 다른 승인 정책을 적용해야 할 수 있습니다. 예를 들어, 금액이 특정 임계값을 초과하는 금전 이체는 자동으로 거부하고, 그 이하는 승인하는 정책을 구현할 수 있습니다.
#
# 아래 코드는 금융 에이전트를 생성하고, 이체 금액에 따라 조건부로 승인 또는 거부하는 예제입니다.

# %%
@tool
def transfer_money(amount: float, recipient: str) -> str:
    """수신자에게 금액을 이체합니다."""
    return f"Transferred ${amount} to {recipient}"

@tool
def check_balance(account: str) -> str:
    """계좌 잔액을 조회합니다."""
    return f"Account {account} balance: $10,000"

# 금융 에이전트
finance_agent = create_agent(
    model=model,
    tools=[transfer_money, check_balance],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 금전 이체는 승인 또는 거부만 (편집 불가)
                "transfer_money": {"allowed_decisions": ["approve", "reject"]},
                # 잔액 조회는 승인 불필요
                "check_balance": False,
            },
            description_prefix="Financial operation pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

config_finance = {"configurable": {"thread_id": "finance_thread_001"}}

# 금전 이체 요청
result = finance_agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Transfer $500 to alice@example.com"
        }]
    },
    config=config_finance
)

if "__interrupt__" in result:
    print("=== Financial Operation Requires Approval ===")
    interrupt_data = result["__interrupt__"][0].value
    action = interrupt_data["action_requests"][0]
    
    amount = action["args"]["amount"]
    recipient = action["args"]["recipient"]
    
    print(f"\nAmount: ${amount}")
    print(f"Recipient: {recipient}")
    
    # 금액에 따른 조건부 승인
    if amount > 1000:
        decision = {
            "type": "reject",
            "message": f"Transfer amount ${amount} exceeds the $1,000 limit. Please contact a supervisor."
        }
        print("\n REJECTED: Amount exceeds limit")
    else:
        decision = {"type": "approve"}
        print("\n APPROVED: Amount within limit")
    
    result = finance_agent.invoke(
        Command(resume={"decisions": [decision]}),
        config=config_finance
    )
    
    print("\nFinal result:")
    print(result["messages"][-1].content)

# %% [markdown]
# ## [보너스] 정리 및 모범 사례
#
# HITL을 효과적으로 사용하기 위한 주요 가이드라인을 정리합니다. 이러한 모범 사례를 따르면 안정적이고 사용자 친화적인 HITL 시스템을 구축할 수 있습니다.
# 아래는 각 사례별로 `의사코드를 표시`했습니다. `패턴 예시`이기 때문에 상황에 맞게 적절히 변경해서 사용하세요.
#
# ### 1. 체크포인터 필수
#
# HITL을 사용하려면 반드시 체크포인터를 설정해야 합니다. 체크포인터가 없으면 interrupt 후 상태를 복원할 수 없어 실행을 재개할 수 없습니다. 개발/테스트 환경에서는 `InMemorySaver`를, 프로덕션에서는 `AsyncPostgresSaver`를 권장합니다.
#
# 아래 코드는 개발 환경과 프로덕션 환경에서의 체크포인터 설정 예제입니다.
#
# ```python
# # 개발/테스트
# from langgraph.checkpoint.memory import InMemorySaver
# checkpointer = InMemorySaver()
#
# # 프로덕션 (예제)
# # from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# # checkpointer = AsyncPostgresSaver.from_conn_string("postgresql://...")
# ```

# %% [markdown]
# ### 2. Thread ID 관리
#
# 동일한 `thread_id`를 사용해야 대화를 일시 중지하고 재개할 수 있습니다. 각 대화 세션에 고유한 `thread_id`를 할당하고, interrupt 후 재개 시 동일한 ID를 사용해야 합니다. UUID를 사용하면 충돌 없이 고유한 ID를 생성할 수 있습니다.
#
# 아래 코드는 동일한 `thread_id`로 초기 호출과 재개를 수행하는 패턴을 보여줍니다.
#
# ```python
# # 동일한 thread_id 사용
# config = {"configurable": {"thread_id": "unique_thread_id"}}
#
# # 초기 호출
# result = agent.invoke({"messages": [...]}, config=config)
#
# # 재개
# result = agent.invoke(Command(resume={...}), config=config)  # 동일한 config
# ```

# %% [markdown]
# ### 3. 결정 순서
#
# 여러 작업에 대한 결정을 제공할 때는 interrupt 데이터의 `action_requests` 순서와 정확히 일치해야 합니다. 순서가 맞지 않으면 잘못된 작업에 결정이 적용될 수 있습니다.
#
# 아래 코드는 세 개의 작업에 대해 순서대로 결정을 제공하는 예제입니다.
#
# ```python
# # 올바른 순서
# decisions = [
#     {"type": "approve"},     # 첫 번째 작업
#     {"type": "reject", ...}, # 두 번째 작업
#     {"type": "edit", ...}    # 세 번째 작업
# ]
# ```

# %% [markdown]
# ### 4. 편집 시 주의사항
#
# 도구 인수를 편집할 때는 신중하게 최소한의 변경만 수행하세요. 큰 수정은 에이전트가 접근 방식을 재평가하고 예기치 않은 동작을 일으킬 수 있습니다. 가능하면 수신자나 파일명 같은 단일 필드만 변경하고, 나머지는 원래 값을 유지하는 것이 좋습니다.
#
# 아래 코드는 좋은 편집 예제(최소 변경)와 나쁜 편집 예제(대폭 수정)를 비교합니다.
#
# ```python
# # 좋은 예: 소폭 수정
# {
#     "type": "edit",
#     "edited_action": {
#         "name": "send_email",
#         "args": {
#             "recipient": "new@example.com",  # 수신자만 변경
#             "subject": "...",  # 나머지는 유지
#             "body": "..."
#         }
#     }
# }
#
# # 나쁜 예: 대폭 수정
# # 도구 이름 변경, 모든 인수 변경 등
# ```

# %% [markdown]
# ### 5. 적절한 승인 정책
#
# 작업의 위험도에 따라 적절한 승인 정책을 설정하세요. 위험한 작업은 수정을 허용하지 않고 승인/거부만 가능하게 하고, 보통 위험한 작업은 모든 결정을 허용하며, 안전한 작업은 승인 없이 자동 실행되도록 설정합니다.
#
# 아래 코드는 작업 위험도에 따른 승인 정책 설정 예제입니다.
#
# ```python
# HumanInTheLoopMiddleware(
#     interrupt_on={
#         # 위험한 작업: 승인/거부만 (편집 불가)
#         "delete_database": {"allowed_decisions": ["approve", "reject"]},
#         
#         # 보통 위험: 모든 결정 허용
#         "send_email": True,
#         
#         # 안전한 작업: 승인 불필요
#         "read_data": False,
#     }
# )
# ```

# %% [markdown]
# ## 요약
#
# 이 튜토리얼에서는 Human-in-the-Loop 미들웨어를 사용하여 에이전트에 사람의 감독을 추가하는 방법을 학습했습니다. HITL을 사용하면 다음과 같은 이점을 얻을 수 있습니다:
#
# 1. **민감한 작업에 대한 사람의 감독**을 추가하여 안전성을 확보할 수 있습니다
# 2. **Approve, Edit, Reject** 세 가지 결정 타입으로 유연하게 대응할 수 있습니다
# 3. **체크포인터를 통한 상태 저장**으로 안전한 일시 중지/재개가 가능합니다
# 4. **작업별 승인 정책**을 유연하게 설정하여 위험도에 맞는 감독 수준을 적용할 수 있습니다
#
# HITL은 AI 에이전트가 안전하고 책임감 있게 작동하도록 보장하는 중요한 메커니즘입니다. 프로덕션 환경에서는 반드시 적절한 승인 정책과 체크포인터를 설정하여 사용하세요.
