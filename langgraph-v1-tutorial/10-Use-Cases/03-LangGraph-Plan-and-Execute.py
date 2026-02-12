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
# # Plan-and-Execute
#
# 이 튜토리얼에서는 "plan-and-execute" 스타일의 에이전트를 만드는 방법을 소개하고, 이를 [LangGraph](https://langchain-ai.github.io/langgraph/)를 활용하여 구현하는 과정을 단계별로 설명합니다.  
# "plan-and-execute" 전략은 복잡한 작업을 수행할 때 장기적인 계획을 먼저 수립한 뒤, 해당 계획을 단계별로 실행하며 필요에 따라 다시 계획을 재수정하는 접근법입니다.
#
# ![](./assets/langgraph-plan-and-execute.png)
#
# ---
#
# ## Plan-and-Execute란 무엇인가?
#
# "plan-and-execute"는 다음과 같은 특징을 갖는 접근 방식입니다.
#
# - **장기 계획 수립**: 복잡한 작업을 수행하기 전에 큰 그림을 그리는 장기 계획을 수립합니다.
# - **단계별 실행 및 재계획**: 세운 계획을 단계별로 실행하고, 각 단계가 완료될 때마다 계획이 여전히 유효한지 검토한 뒤 수정할 수 있습니다.
#   
# 이 방식은 [Plan-and-Solve 논문](https://arxiv.org/abs/2305.04091)과 [Baby-AGI 프로젝트](https://github.com/yoheinakajima/babyagi)에서 영감을 받았습니다. 전통적인 [ReAct 스타일](https://arxiv.org/abs/2210.03629)의 에이전트는 한 번에 한 단계씩 생각하는 반면, "plan-and-execute"는 명시적이고 장기적인 계획을 강조합니다.
#
# **장점**:
# 1. **명시적인 장기 계획**: 강력한 LLM조차도 한 번에 장기 계획을 처리하는 데 어려움을 겪을 수 있습니다. 명시적으로 장기 계획을 수립함으로써, 보다 안정적인 진행이 가능합니다.
# 2. **효율적인 모델 사용**: 계획 단계에서는 더 큰/강력한 모델을 사용하고, 실행 단계에서는 상대적으로 작은/약한 모델을 사용함으로써 자원 소비를 최적화할 수 있습니다.
#
# ---
#
# **주요 내용**
#
# - **도구 정의**: 사용할 도구 정의
# - **실행 에이전트 정의**: 실제 작업을 실행하는 에이전트 생성
# - **상태 정의**: 에이전트의 상태 정의
# - **계획 단계**: 장기 계획을 세우는 단계 생성
# - **재계획 단계**: 작업 진행 상황에 따라 계획을 재수정하는 단계 생성
# - **그래프 생성 및 실행**: 이러한 단계들을 연결하는 그래프 생성 및 실행
#
# ---
#
# **참고**
#
# - [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)  
# - [Plan-and-Solve 논문](https://arxiv.org/abs/2305.04091)  
# - [Baby-AGI 프로젝트](https://github.com/yoheinakajima/babyagi)  
# - [ReAct 논문](https://arxiv.org/abs/2210.03629)
#
# 지금부터는 각 단계를 따라가며 "plan-and-execute" 에이전트를 LangGraph로 구현하는 방법을 자세히 알아보겠습니다.

# %% [markdown]
# ## 환경 설정
#
# Plan-and-Execute 에이전트를 구현하기 위해 필요한 환경을 설정합니다. API 키를 환경변수로 관리하고, LangSmith를 통해 실행 과정을 추적할 수 있도록 설정합니다.
#
# 아래 코드는 `.env` 파일에서 API 키를 로드하고 LangSmith 추적을 활성화합니다.

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-Use-Cases")

# %% [markdown]
# 실습에 활용할 모델명을 정의합니다. `langchain_teddynote` 패키지의 `get_model_name` 함수를 사용하여 모델명을 가져옵니다.

# %%
from langchain_teddynote.models import get_model_name, LLMs

# 모델명 정의
MODEL_NAME = get_model_name(LLMs.GPT4o)
print(MODEL_NAME)

# %% [markdown]
# ## 도구 정의
#
# 사용할 도구를 먼저 정의합니다. 이 간단한 예제에서는 `Tavily`를 통해 제공되는 내장 검색 도구를 사용할 것입니다. 그러나 직접 도구를 만드는 것도 매우 쉽습니다. 
#
# 자세한 내용은 [도구(Tools)](https://wikidocs.net/262582) 문서를 참조하십시오.

# %%
from langchain_teddynote.tools import TavilySearch

# Tavily 검색 도구 초기화
tools = [TavilySearch(max_results=8)]

# %% [markdown]
# ## 작업 실행 에이전트 정의
#
# 이제 작업을 실행할 `execution agent`를 생성합니다. 이 예제에서는 각 작업에 동일한 `execution agent`를 사용하지만, 필요에 따라 작업별로 다른 에이전트를 사용할 수 있습니다.
#
# `create_react_agent` 함수는 LangGraph의 prebuilt 모듈에서 제공되며, ReAct(Reasoning + Acting) 패턴을 구현한 에이전트를 생성합니다. 이 에이전트는 도구를 사용하여 정보를 수집하고 추론을 수행합니다.
#
# 아래 코드는 Tavily 검색 도구를 사용하는 ReAct 에이전트를 생성합니다.

# %%
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

# 시스템 프롬프트 정의
system_prompt = "You are a helpful assistant. Answer in Korean."

# LLM 정의
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# ReAct 에이전트 생성
# state_modifier를 사용하여 시스템 프롬프트를 설정합니다.
agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

# %%
# 에이전트 실행
agent_executor.invoke(
    {"messages": [("user", "랭체인 한국어 튜토리얼에 대해서 설명해줘")]}
)

# %% [markdown]
# ## 상태 정의
#
# - `input`: 사용자의 입력
# - `plan`: 현재 계획
# - `past_steps`: 이전에 실행한 계획과 실행 결과
# - `response`: 최종 응답

# %%
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict


# 상태 정의
class PlanExecute(TypedDict):
    input: Annotated[str, "User's input"]
    plan: Annotated[List[str], "Current plan"]
    past_steps: Annotated[List[Tuple], operator.add]
    response: Annotated[str, "Final response"]


# %% [markdown]
# ## 계획(Plan) 단계
#
# 이제 **계획 단계**를 생성합니다. 이 단계에서는 `function calling`(도구 호출)을 사용하여 구조화된 계획을 수립합니다.
#
# LLM에 `with_structured_output` 메서드를 사용하여 Pydantic 모델 형식으로 출력을 강제합니다. 이를 통해 계획이 항상 일관된 형식으로 생성됩니다.
#
# 아래 코드는 Plan 모델과 planner 체인을 정의합니다.

# %%
from pydantic import BaseModel, Field
from typing import List


# Plan 모델 정의
class Plan(BaseModel):
    """Sorted steps to execute the plan"""

    steps: Annotated[List[str], "Different steps to follow, should be in sorted order"]


# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 계획 수립을 위한 프롬프트 템플릿 생성
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
Answer in Korean.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

planner = planner_prompt | ChatOpenAI(
    model=MODEL_NAME, temperature=0
).with_structured_output(Plan)

# %% [markdown]
# `planner`를 실행하여 계획을 수립한 결과를 확인합니다. 사용자의 질문에 대해 단계별 계획이 생성됩니다.

# %%
# Planner 실행
planner.invoke(
    {
        "messages": [
            (
                "user",
                """Slot Filling

사용자 질문의 의도를 정확하게 파악하고 답변을 하기 위해서는 꼭 필요한 정보들이 있을텐데, 해당 entity를 어떻게 확보해서 답변을 할 수 있을지 고민이 있습니다.  
예를들면 회의실을 예약하는 시나리오라고 하면, 시간, 장소, 인원 등의 필수 정보가 있어야할텐데 이런 항목을 multi-turn 활용하여 채울 수 있는 방법이 있겠으나  
범용적으로 사용하기에는 도메인별로 필요한 Entity가 달라서 어떻게 하면 좋을지 고민 입니다.  
외부에서 Slot Filling 방식으로 활용하고 있는지, 어떻게 활용하는지 알고 싶습니다.  """,
            )
        ]
    }
)

# %% [markdown]
# ## 재계획(Re-Plan) 단계
#
# 이제 이전 단계의 결과를 바탕으로 계획을 다시 수립하는 단계를 생성합니다. 재계획 단계에서는 현재까지 완료된 작업을 검토하고, 남은 작업을 업데이트합니다.
#
# `Act` 모델은 두 가지 유형의 행동을 정의합니다:
# - **Response**: 사용자에게 최종 응답을 제공할 때 사용
# - **Plan**: 추가 도구 사용이 필요할 때 사용
#
# 아래 코드는 재계획을 위한 프롬프트와 replanner 체인을 정의합니다.

# %%
from typing import Union


class Response(BaseModel):
    """Response to user."""

    # 사용자 응답
    response: str


class Act(BaseModel):
    """Action to perform."""

    # 수행할 작업: "Response", "Plan". 사용자에게 응답할 경우 Response 사용, 추가 도구 사용이 필요할 경우 Plan 사용
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# 계획을 재수립하기 위한 프롬프트 정의
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.

Answer in Korean."""
)


# Replanner 생성
replanner = replanner_prompt | ChatOpenAI(
    model=MODEL_NAME, temperature=0
).with_structured_output(Act)

# %% [markdown]
# ## 그래프 생성
#
# 이제 Plan-and-Execute 워크플로우를 구현하는 그래프를 생성합니다. 그래프는 다음과 같은 노드로 구성됩니다:
#
# - **planner**: 초기 계획을 수립하는 노드
# - **execute**: 계획의 각 단계를 실행하는 노드
# - **replan**: 실행 결과를 바탕으로 재계획하는 노드
# - **final_report**: 최종 보고서를 생성하는 노드
#
# 아래 코드는 각 노드의 함수와 최종 보고서 생성 로직을 정의합니다.

# %%
from langchain_core.output_parsers import StrOutputParser


# 사용자 입력을 기반으로 계획을 생성하고 반환
def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    # 생성된 계획의 단계 리스트 반환
    return {"plan": plan.steps}


# 에이전트 실행기를 사용하여 주어진 작업을 수행하고 결과를 반환
def execute_step(state: PlanExecute):
    plan = state["plan"]
    # 계획을 문자열로 변환하여 각 단계에 번호를 매김
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    # 현재 실행할 작업을 포맷팅하여 에이전트에 전달
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing [step 1. {task}]."""
    # 에이전트 실행기를 통해 작업 수행 및 결과 수신
    agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})
    # 이전 단계와 그 결과를 포함하는 딕셔너리 반환
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


# 이전 단계의 결과를 바탕으로 계획을 업데이트하거나 최종 응답을 반환
def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    # 응답이 사용자에게 반환될 경우
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    # 추가 단계가 필요할 경우 계획의 단계 리스트 반환
    else:
        next_plan = output.action.steps
        if len(next_plan) == 0:
            return {"response": "No more steps needed."}
        else:
            return {"plan": next_plan}


# 에이전트의 실행 종료 여부를 결정하는 함수
def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return "final_report"
    else:
        return "execute"


final_report_prompt = ChatPromptTemplate.from_template(
    """You are given the objective and the previously done steps. Your task is to generate a final report in markdown format.
Final report should be written in professional tone.

Your objective was this:

{input}

Your previously done steps(question and answer pairs):

{past_steps}

Generate a final report in markdown format. Write your response in Korean."""
)

final_report = (
    final_report_prompt
    | ChatOpenAI(model=MODEL_NAME, temperature=0)
    | StrOutputParser()
)


def generate_final_report(state: PlanExecute):
    past_steps = "\n\n".join(
        [
            f"Question: {past_step[0]}\n\nAnswer: {past_step[1]}\n\n####"
            for past_step in state["past_steps"]
        ]
    )
    response = final_report.invoke({"input": state["input"], "past_steps": past_steps})
    return {"response": response}


# %% [markdown]
# ### 그래프 조립
#
# 이제 지금까지 정의한 노드를 연결하여 그래프를 생성합니다. `add_conditional_edges`를 사용하여 `replan` 노드 이후의 분기를 처리합니다.
#
# `MemorySaver`를 사용하여 그래프 실행 상태를 저장하고, 이후에 상태를 복원할 수 있습니다. 아래 코드는 StateGraph를 조립하고 컴파일합니다.

# %%
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# 그래프 생성
workflow = StateGraph(PlanExecute)

# 노드 정의
workflow.add_node("planner", plan_step)
workflow.add_node("execute", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_node("final_report", generate_final_report)

# 엣지 정의
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "execute")
workflow.add_edge("execute", "replan")
workflow.add_edge("final_report", END)

# 조건부 엣지: replan 후 종료 여부를 결정하는 함수 사용
workflow.add_conditional_edges(
    "replan",
    should_end,
    {"execute": "execute", "final_report": "final_report"},
)

# 그래프 컴파일
app = workflow.compile(checkpointer=MemorySaver())

# %% [markdown]
# 그래프를 시각화하여 노드 간의 연결 관계를 확인합니다. `xray=True` 옵션을 사용하면 내부 구조를 더 자세히 볼 수 있습니다.

# %%
from langchain_teddynote.graphs import visualize_graph

visualize_graph(app, xray=True)

# %% [markdown]
# ## 그래프 실행
#
# 이제 생성한 Plan-and-Execute 에이전트를 실행합니다. `invoke_graph` 함수를 사용하여 그래프를 실행하고 결과를 확인합니다.
#
# `recursion_limit`을 설정하여 무한 루프를 방지하고, `thread_id`를 통해 세션을 구분합니다. 아래 코드는 AI Agent와 워크플로우의 차이점에 대한 질문을 처리합니다.

# %%
from langchain_teddynote.messages import invoke_graph, random_uuid
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(recursion_limit=50, configurable={"thread_id": random_uuid()})

inputs = {
    "input": """AI Agent 와 워크플로우의 차이에 대해서 설명하고, 각각의 장단점에 대해서 설명하세요"""
}

invoke_graph(app, inputs, config)

# %%
snapshot = app.get_state(config).values
print(snapshot["response"])

# %%
from IPython.display import Markdown

Markdown(snapshot["response"])

# %%
print(snapshot["response"])
