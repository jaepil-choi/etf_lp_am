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
# # 구조화된 출력 (Structured Output)
#
# 구조화된 출력을 사용하면 에이전트가 특정하고 예측 가능한 형식으로 데이터를 반환할 수 있습니다. 자연어 응답을 구문 분석하는 대신 애플리케이션에서 직접 사용할 수 있는 JSON 객체, Pydantic 모델 또는 데이터클래스 형태로 구조화된 데이터를 얻을 수 있습니다.
#
# **구조화된 출력 전략:**
#
# | 전략 | 설명 |
# |:---|:---|
# | **ProviderStrategy** | OpenAI, Grok 등 네이티브 구조화된 출력 지원 모델용 |
# | **ToolStrategy** | 도구 호출을 통한 구조화된 출력 (대부분의 모델 지원) |
# | **자동 선택** | 스키마 타입만 전달 시 모델에 따라 최적 전략 자동 선택 |
#
# LangChain의 `create_agent`는 `response_format` 매개변수로 구조화된 출력을 설정하며, 결과는 에이전트 상태의 `structured_response` 키에 반환됩니다.
#
# > 참고 문서: [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/structured_output.md)

# %% [markdown]
# ## 환경 설정
#
# 구조화된 출력 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드합니다.
#
# 아래 코드는 환경 변수를 로드합니다.

# %%
from dotenv import load_dotenv

load_dotenv(override=True)

# %% [markdown]
# ---
#
# ## Response Format
#
# 에이전트가 구조화된 데이터를 반환하는 방법을 `response_format` 매개변수로 제어합니다:
#
# | 설정 | 설명 |
# |:---|:---|
# | **ToolStrategy[T]** | 도구 호출을 통한 구조화된 출력 |
# | **ProviderStrategy[T]** | 제공자 네이티브 구조화된 출력 사용 |
# | **type[T]** | 스키마 타입 직접 전달 - 모델에 따라 최적 전략 자동 선택 |
# | **None** | 구조화된 출력 없음 |
#
# 스키마 타입이 직접 제공되면 LangChain이 자동으로 최적의 전략을 선택합니다:
# - 네이티브 구조화된 출력 지원 모델(OpenAI, Grok)에는 `ProviderStrategy`
# - 다른 모든 모델에는 `ToolStrategy`
#
# 구조화된 응답은 에이전트의 최종 상태의 `structured_response` 키에 반환됩니다.

# %% [markdown]
# ---
#
# ## Provider Strategy
#
# 일부 모델 제공자(현재 OpenAI 및 Grok만 해당)는 API를 통해 구조화된 출력을 네이티브로 지원합니다. 이 방법은 사용 가능한 경우 가장 신뢰할 수 있는 방법입니다.
#
# 스키마 타입을 `create_agent.response_format`에 직접 전달하면 LangChain이 지원 모델에 대해 자동으로 `ProviderStrategy`를 사용합니다.

# %% [markdown]
# ### Pydantic 모델
#
# Pydantic 모델을 사용하면 필드에 대한 상세한 설명과 검증 규칙을 정의할 수 있습니다. `Field`의 `description`은 모델이 각 필드의 용도를 이해하는 데 도움이 됩니다.
#
# 아래 코드는 Pydantic 모델로 연락처 정보를 추출하는 예시입니다.

# %%
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


class ContactInfo(BaseModel):
    """Contact information for a person."""

    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


# 모델 및 에이전트 생성
model = ChatOpenAI(model="gpt-4.1-mini")
agent = create_agent(
    model=model, tools=[], response_format=ContactInfo  # ProviderStrategy 자동 선택
)

# 에이전트 실행
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567",
            }
        ]
    }
)

# 구조화된 응답 확인
print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

# %% [markdown]
# ### 데이터클래스
#
# Python의 `@dataclass` 데코레이터를 사용하여 스키마를 정의할 수도 있습니다. 필드 설명은 주석으로 추가합니다.
#
# 아래 코드는 데이터클래스로 연락처 정보를 추출하는 예시입니다.

# %%
from dataclasses import dataclass
from langchain.agents import create_agent


@dataclass
class ContactInfo:
    """Contact information for a person."""

    name: str  # The name of the person
    email: str  # The email address of the person
    phone: str  # The phone number of the person


agent = create_agent(
    model=model, tools=[], response_format=ContactInfo  # ProviderStrategy 자동 선택
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: Jane Smith, jane@example.com, (555) 987-6543",
            }
        ]
    }
)

print(result["structured_response"])
# ContactInfo(name='Jane Smith', email='jane@example.com', phone='(555) 987-6543')

# %% [markdown]
# ### TypedDict
#
# `TypedDict`를 사용하면 딕셔너리 형태로 구조화된 출력을 받을 수 있습니다. 반환값이 딕셔너리이므로 JSON 직렬화가 용이합니다.
#
# 아래 코드는 TypedDict로 연락처 정보를 추출하는 예시입니다.

# %%
from typing_extensions import TypedDict
from langchain.agents import create_agent


class ContactInfo(TypedDict):
    """Contact information for a person."""

    name: str  # The name of the person
    email: str  # The email address of the person
    phone: str  # The phone number of the person


agent = create_agent(
    model=model, tools=[], response_format=ContactInfo  # ProviderStrategy 자동 선택
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Extract contact info from: Teddy Lee, teddy@example.com, (555) 111-2222",
            }
        ]
    }
)

print(result["structured_response"])
# {'name': 'Teddy Lee', 'email': 'teddy@example.com', 'phone': '(555) 111-2222'}

# %% [markdown]
# ---
#
# ## Tool Calling Strategy
#
# 네이티브 구조화된 출력을 지원하지 않는 모델의 경우 LangChain은 도구 호출을 사용하여 동일한 결과를 달성합니다. `ToolStrategy`는 도구 호출을 지원하는 대부분의 최신 모델에서 작동합니다.
#
# `ToolStrategy`를 명시적으로 사용하면 지원 여부와 관계없이 항상 도구 호출 방식을 사용합니다.

# %% [markdown]
# ### Pydantic 모델
#
# `ToolStrategy`를 사용할 때도 Pydantic 모델의 필드 설명과 검증 규칙이 동일하게 적용됩니다. `Literal` 타입으로 허용값을 제한하고, `ge`, `le` 등의 검증자로 범위를 지정할 수 있습니다.
#
# 아래 코드는 ToolStrategy로 제품 리뷰를 분석하는 예시입니다.

# %%
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )
    key_points: list[str] = Field(
        description="The key points of the review. Lowercase, 1-3 words each."
    )


agent = create_agent(model=model, tools=[], response_format=ToolStrategy(ProductReview))

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
            }
        ]
    }
)

print(result["structured_response"])
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])

# %% [markdown]
# ### Union 타입
#
# `Union` 타입을 사용하여 여러 스키마 옵션을 제공할 수 있습니다. 모델은 입력 컨텍스트에 따라 가장 적절한 스키마를 자동으로 선택합니다.
#
# 아래 코드는 리뷰와 불만을 구분하여 분석하는 예시입니다.

# %%
from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review"
    )
    key_points: list[str] = Field(
        description="The key points of the review. Lowercase, 1-3 words each."
    )


class CustomerComplaint(BaseModel):
    """A customer complaint about a product or service."""

    issue_type: Literal["product", "service", "shipping", "billing"] = Field(
        description="The type of issue"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="The severity of the complaint"
    )
    description: str = Field(description="Brief description of the complaint")


agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(Union[ProductReview, CustomerComplaint]),
)

# 리뷰 분석
result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'",
            }
        ]
    }
)
print("Review:", result1["structured_response"])

# 불만 처리
result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Customer complaint: Package arrived damaged and contents were broken",
            }
        ]
    }
)
print("Complaint:", result2["structured_response"])

# %% [markdown]
# ### 커스텀 도구 메시지 콘텐츠
#
# `tool_message_content` 매개변수를 사용하면 구조화된 출력이 생성될 때 대화 기록에 나타나는 메시지를 커스터마이징할 수 있습니다. 이는 후속 대화에서 컨텍스트를 제공하는 데 유용합니다.
#
# 아래 코드는 커스텀 도구 메시지를 설정하는 예시입니다.

# %%
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""

    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!",
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "From our meeting: Sarah needs to update the project timeline as soon as possible",
            }
        ]
    }
)

print(result["structured_response"])
# MeetingAction(task='Update the project timeline', assignee='Sarah', priority='high')

# %% [markdown]
# ---
#
# ## 오류 처리
#
# 모델은 도구 호출을 통해 구조화된 출력을 생성할 때 스키마와 일치하지 않는 값을 반환할 수 있습니다. LangChain은 이러한 오류를 자동으로 처리하는 지능형 재시도 메커니즘을 제공합니다.
#
# `handle_errors` 매개변수로 오류 처리 방법을 제어할 수 있으며, 기본값은 `True`입니다.

# %% [markdown]
# ### 스키마 검증 오류
#
# 구조화된 출력이 예상 스키마와 일치하지 않으면 에이전트는 오류 피드백을 제공하고 모델에게 재시도를 요청합니다. 예를 들어, rating이 1-5 범위인데 10이 입력되면 자동으로 수정을 시도합니다.
#
# 아래 코드는 스키마 검증 오류 처리 예시입니다.

# %%
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")


agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(ProductRating),  # 기본값: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]}
)

print(result["structured_response"])
# ProductRating(rating=5, comment='Amazing product')
# 모델이 자동으로 수정하여 10을 5로 변경

# %% [markdown]
# ### 오류 처리 전략
#
# `handle_errors` 매개변수를 사용하여 오류 처리 방법을 커스터마이징할 수 있습니다.
#
# | 설정 | 설명 |
# |:---|:---|
# | `True` | 모든 오류를 자동으로 처리하고 재시도 (기본값) |
# | `False` | 오류 발생 시 예외 발생 |
# | 문자열 | 커스텀 오류 메시지로 재시도 |
# | 예외 클래스 | 특정 예외만 처리 |
# | 콜러블 | 커스텀 오류 핸들러 함수 사용 |

# %% [markdown]
# ### 커스텀 오류 메시지
#
# 문자열을 전달하면 해당 메시지로 모델에게 재시도를 요청합니다.
#
# 아래 코드는 커스텀 오류 메시지 설정 예시입니다.

# %%
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors="Please provide a valid rating between 1-5 and include a comment.",
    ),
)

# %% [markdown]
# ### 특정 예외만 처리
#
# 예외 클래스를 전달하면 해당 예외만 처리하고 다른 예외는 그대로 발생합니다.
#
# 아래 코드는 특정 예외만 처리하는 예시입니다.

# %%
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=ValueError,  # ValueError만 재시도, 다른 예외는 발생
    ),
)

# %% [markdown]
# ### 여러 예외 유형 처리
#
# 튜플로 여러 예외 클래스를 전달하면 해당 예외들을 모두 처리합니다.
#
# 아래 코드는 여러 예외 유형을 처리하는 예시입니다.

# %%
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=(ValueError, TypeError),  # ValueError 및 TypeError 재시도
    ),
)

# %% [markdown]
# ### 커스텀 오류 핸들러 함수
#
# 함수를 전달하면 오류 발생 시 해당 함수가 호출되어 커스텀 오류 메시지를 생성합니다.
#
# 아래 코드는 커스텀 오류 핸들러 함수 예시입니다.

# %%
from langchain.agents.structured_output import (
    StructuredOutputValidationError,
    MultipleStructuredOutputsError,
)


def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"


agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating, handle_errors=custom_error_handler
    ),
)

# %% [markdown]
# ### 오류 처리 비활성화
#
# `handle_errors=False`를 설정하면 오류 발생 시 예외가 그대로 발생합니다.
#
# 아래 코드는 오류 처리 비활성화 예시입니다.

# %%
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating, handle_errors=False  # 모든 오류 발생
    ),
)

# %% [markdown]
# ---
#
# ## 종합 예제
#
# `Union` 타입과 오류 처리를 결합한 실용적인 예제입니다. 책과 영화 추천을 모두 처리하고, 오류 발생 시 자동으로 재시도합니다.
#
# 아래 코드는 여러 추천 유형을 처리하는 에이전트 예시입니다.

# %%
from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# 여러 응답 유형 정의
class BookRecommendation(BaseModel):
    """Book recommendation with details."""

    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    genre: Literal["fiction", "non-fiction", "science", "history", "biography"] = Field(
        description="Book genre"
    )
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    summary: str = Field(description="Brief summary of the book")


class MovieRecommendation(BaseModel):
    """Movie recommendation with details."""

    title: str = Field(description="Movie title")
    director: str = Field(description="Director name")
    year: int = Field(description="Release year")
    genre: Literal["action", "comedy", "drama", "horror", "sci-fi"] = Field(
        description="Movie genre"
    )
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)


# 에이전트 생성
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=Union[BookRecommendation, MovieRecommendation], handle_errors=True
    ),
    system_prompt="You are a helpful entertainment recommendation assistant.",
)

# 책 추천
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Recommend a good science fiction book"}]}
)
print("Book recommendation:")
print(result1["structured_response"])

# 영화 추천
result2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Recommend a comedy movie from the 2000s"}
        ]
    }
)
print("\nMovie recommendation:")
print(result2["structured_response"])

# %% [markdown]
# ---
#
# ## 정리
#
# 이 튜토리얼에서는 LangGraph 에이전트의 구조화된 출력 기능을 학습했습니다.
#
# **핵심 개념 요약:**
#
# | 개념 | 설명 |
# |:---|:---|
# | **ProviderStrategy** | OpenAI, Grok 등 네이티브 지원 모델용 (가장 신뢰성 높음) |
# | **ToolStrategy** | 도구 호출을 통한 구조화된 출력 (대부분의 모델 지원) |
# | **Union 타입** | 여러 스키마 중 자동 선택 |
# | **handle_errors** | 오류 발생 시 자동 재시도 및 수정 |
#
# **스키마 정의 방법:**
# - Pydantic 모델: 가장 풍부한 검증 기능 제공
# - 데이터클래스: 간단한 스키마 정의
# - TypedDict: 딕셔너리 형태로 반환
#
# **실전 팁:**
# - 필드에 명확한 description 제공
# - `Literal` 타입으로 허용값 제한
# - `ge`, `le` 등 검증자로 범위 지정
# - Union 타입으로 다양한 응답 유형 처리
