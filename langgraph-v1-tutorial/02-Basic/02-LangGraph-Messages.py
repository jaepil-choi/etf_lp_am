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
# # 메시지 (Messages)
#
# 메시지는 LangChain에서 모델과의 대화를 나타내는 기본 단위입니다. 모든 LLM 상호작용은 메시지를 통해 이루어지며, 각 메시지는 역할(role), 콘텐츠(content), 메타데이터를 포함합니다.
#
# LangGraph 애플리케이션에서 메시지는 상태(State)의 핵심 구성 요소로 사용됩니다. 대화 이력을 관리하고, 도구 호출 결과를 전달하며, 멀티모달 콘텐츠를 처리하는 데 활용됩니다.
#
# 메시지 객체는 다음을 포함합니다:
#
# - **역할(Role)**: 메시지 유형을 식별 (예: `system`, `user`, `assistant`)
# - **콘텐츠(Content)**: 메시지의 실제 내용 (텍스트, 이미지, 오디오, 문서 등)
# - **메타데이터(Metadata)**: 응답 정보, 메시지 ID, 토큰 사용량 등의 선택적 필드
#
# LangChain은 모델 제공자에 관계없이 일관된 동작을 보장하는 표준 메시지 유형을 제공합니다.

# %% [markdown]
# ## 환경 설정
#
# LangGraph 튜토리얼을 시작하기 전에 필요한 환경을 설정합니다. `dotenv`를 사용하여 API 키를 로드하고, `langchain_teddynote`의 로깅 기능을 활성화하여 LangSmith에서 실행 추적을 확인할 수 있도록 합니다.
#
# LangSmith 추적을 활성화하면 메시지 흐름을 시각적으로 디버깅할 수 있어, 대화 기반 애플리케이션 개발에 큰 도움이 됩니다.
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
# ## 기본 사용법
#
# 메시지를 사용하는 가장 간단한 방법은 메시지 객체를 생성하고 모델을 호출할 때 리스트로 전달하는 것입니다. `SystemMessage`는 모델의 행동을 지시하고, `HumanMessage`는 사용자 입력을 나타내며, `AIMessage`는 모델의 응답을 나타냅니다.
#
# 메시지 리스트를 모델에 전달하면 대화 컨텍스트가 유지되며, 모델은 이전 대화 내용을 참고하여 응답을 생성합니다.
#
# 아래 코드는 시스템 메시지와 사용자 메시지를 생성하고 모델을 호출하는 기본 예시입니다.

# %%
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 모델 초기화
model = init_chat_model("openai:gpt-4.1-mini")

# 메시지 객체 생성
system_msg = SystemMessage("당신은 친절한 Assistant입니다.")
human_msg = HumanMessage("안녕하세요. 반갑습니다.")

# 채팅 모델과 함께 사용
messages = [system_msg, human_msg]
response = model.invoke(messages)  # AIMessage 반환
response

# %% [markdown]
# ## 텍스트 프롬프트
#
# 단순한 텍스트 문자열만으로도 모델을 호출할 수 있습니다. 이 방식은 대화 기록이 필요 없는 간단한 생성 작업에 적합합니다. 문자열을 전달하면 내부적으로 `HumanMessage`로 변환되어 처리됩니다.
#
# 아래 코드는 단일 문자열로 모델을 호출하는 예시입니다.

# %%
# 단일 문자열로 간단한 요청
response = model.invoke("대한민국의 수도는?")
print(response.content)

# %% [markdown]
# ## 메시지 프롬프트
#
# 복잡한 대화나 멀티턴 상호작용에서는 메시지 객체 리스트를 사용합니다. 각 메시지 유형(`SystemMessage`, `HumanMessage`, `AIMessage`)을 조합하여 풍부한 대화 컨텍스트를 구성할 수 있습니다.
#
# **메시지 프롬프트를 사용하는 경우:**
# - 다중 턴 대화를 관리하는 경우
# - 멀티모달 콘텐츠(이미지, 오디오, 파일)를 작업하는 경우
# - 시스템 지침을 포함하는 경우
# - 이전 대화 맥락을 유지해야 하는 경우
#
# 아래 코드는 메시지 객체 리스트로 다중 턴 대화를 구성하는 예시입니다.

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# 메시지 객체를 사용한 대화
messages = [
    SystemMessage("당신은 친절한 어시스턴트입니다."),
    HumanMessage("대한민국의 수도는?"),
    AIMessage("대한민국의 수도는 서울입니다."),
    HumanMessage("영어로 작성해줘."),
]

response = model.invoke(messages)
print(response.content)

# %% [markdown]
# ### 딕셔너리 형식
#
# 메시지 객체 대신 딕셔너리 형식으로도 메시지를 지정할 수 있습니다. `role` 키에는 역할(`system`, `user`, `assistant`)을, `content` 키에는 메시지 내용을 지정합니다. 이 방식은 JSON 직렬화가 필요한 경우나 간결한 코드 작성에 유용합니다.
#
# 아래 코드는 딕셔너리 형식으로 대화를 구성하는 예시입니다.

# %%
# 딕셔너리 형식으로 메시지 지정
messages = [
    {"role": "system", "content": "당신은 친절한 어시스턴트입니다."},
    {"role": "user", "content": "대한민국의 수도는?"},
    {"role": "assistant", "content": "대한민국의 수도는 서울입니다."},
    {"role": "user", "content": "영어로 작성해줘."},
]

response = model.invoke(messages)
print(response.content)

# %% [markdown]
# ## 메시지 유형
#
# LangChain은 네 가지 핵심 메시지 유형을 제공합니다. 각 유형은 대화에서 서로 다른 역할을 담당하며, 올바른 유형을 사용하는 것이 효과적인 프롬프트 엔지니어링의 기본입니다.
#
# | 메시지 유형 | 설명 | 사용 시점 |
# |:---|:---|:---|
# | **SystemMessage** | 모델의 동작 방식과 역할을 정의 | 대화 시작 시 모델 행동 지침 설정 |
# | **HumanMessage** | 사용자 입력을 나타냄 | 사용자의 질문이나 요청 전달 |
# | **AIMessage** | 모델이 생성한 응답 | 이전 응답 기록 유지, 도구 호출 정보 포함 |
# | **ToolMessage** | 도구 호출의 결과 | 도구 실행 결과를 모델에 전달 |

# %% [markdown]
# ### SystemMessage (시스템 메시지)
#
# 시스템 메시지는 모델의 동작을 준비하는 초기 지침 세트입니다. 대화의 맨 처음에 위치하며, 모델의 역할, 톤, 응답 스타일, 제약 조건 등을 정의합니다.
#
# 효과적인 시스템 메시지는 일관된 모델 동작을 이끌어내며, 모델이 특정 도메인 전문가나 특정 성격의 어시스턴트처럼 행동하도록 지시할 수 있습니다.
#
# 아래 코드는 시스템 메시지로 모델의 역할을 정의하고 응답을 생성하는 예시입니다.

# %%
from langchain.messages import SystemMessage, HumanMessage

# 기본 지침
system_msg = SystemMessage("You are a helpful python coding assistant.")

messages = [system_msg, HumanMessage("로또 번호 추첨 코드를 작성해줘.")]

response = model.invoke(messages)
print(response.content)

# %% [markdown]
# ### HumanMessage (사용자 메시지)
#
# 사용자 메시지는 사용자의 입력과 요청을 나타냅니다. 텍스트뿐만 아니라 이미지, 오디오, 파일 등 다양한 형태의 멀티모달 콘텐츠를 포함할 수 있습니다.
#
# LangGraph 애플리케이션에서 사용자 메시지는 그래프 실행의 시작점이 되며, 상태(State)의 메시지 리스트에 추가되어 대화 이력을 형성합니다.
#
# 아래 코드는 HumanMessage 객체를 생성하고 모델을 호출하는 예시입니다.

# %%
from langchain.messages import HumanMessage

# 메시지 객체 사용
response = model.invoke([HumanMessage("What is machine learning?")])
print(response.content)

# %% [markdown]
# ### 메시지 메타데이터
#
# 메시지에는 `content` 외에도 다양한 메타데이터를 추가할 수 있습니다. `name`은 다중 사용자 환경에서 발신자를 식별하는 데 사용되고, `id`는 메시지를 고유하게 식별하여 추적 및 참조에 활용됩니다.
#
# 메타데이터는 대화 분석, 로깅, 사용자별 응답 개인화 등 다양한 용도로 활용할 수 있습니다.
#
# 아래 코드는 메타데이터가 포함된 HumanMessage를 생성하는 예시입니다.

# %%
# 메타데이터 추가
human_msg = HumanMessage(
    content="안녕하세요? 반가워요",
    name="teddy",  # 선택사항: 다른 사용자 식별
    id="abc123",  # 선택사항: 추적을 위한 고유 식별자
)

human_msg

# %%
response = model.invoke([human_msg])
response

# %% [markdown]
# ### AIMessage (AI 메시지)
#
# AI 메시지는 모델 호출의 출력을 나타냅니다. 텍스트 응답뿐만 아니라 도구 호출 정보(`tool_calls`), 토큰 사용량, 응답 메타데이터 등을 포함할 수 있습니다.
#
# 대화 이력을 유지할 때 이전 AI 응답을 `AIMessage`로 포함시키면, 모델이 자신의 이전 발언을 인식하고 일관된 대화를 이어갈 수 있습니다.
#
# 아래 코드는 모델 호출 후 반환되는 AIMessage의 구조를 확인하는 예시입니다.

# %%
# 모델 호출 시 AIMessage 반환
response = model.invoke("대한민국의 수도는?")
print(f"Type: {type(response)}")
print(f"Content: {response.content}")

# %% [markdown]
# ## ToolMessage (도구 메시지)
#
# 도구 호출을 지원하는 모델에서 AI 메시지에 도구 호출(`tool_calls`)이 포함될 수 있습니다. 도구 메시지는 도구 실행 결과를 모델에 다시 전달하는 데 사용됩니다.
#
# LangGraph 에이전트에서 도구 호출 흐름은 다음과 같습니다:
# 1. 사용자 질문 (HumanMessage)
# 2. 모델이 도구 호출 결정 (AIMessage with tool_calls)
# 3. 도구 실행 및 결과 반환 (ToolMessage)
# 4. 모델이 결과를 해석하여 최종 응답 (AIMessage)
#
# ![](assets/LangGraph-Messages-Tool-Message.png)
#
# 아래 코드는 도구 호출 정보가 포함된 AIMessage를 생성하는 예시입니다.

# %%
from langchain.messages import AIMessage, HumanMessage, ToolMessage

# 모델이 도구 호출을 수행한 후
ai_message = AIMessage(
    content="",
    tool_calls=[
        {"name": "get_weather", "args": {"location": "서울"}, "id": "call_123"}
    ],
)
ai_message

# %% [markdown]
# ### ToolMessage 생성
#
# `ToolMessage`를 생성할 때 중요한 점은 `tool_call_id`가 해당 도구를 호출한 `AIMessage`의 `tool_calls` 내 `id`와 일치해야 한다는 것입니다. 이 ID를 통해 모델은 어떤 도구 호출에 대한 결과인지 정확히 파악할 수 있습니다.
#
# 아래 코드는 도구 실행 결과를 담은 ToolMessage를 생성하는 예시입니다.

# %%
# 도구 실행 및 결과 메시지 생성
weather_result = "날씨 맑음. 섭씨 10도"
tool_message = ToolMessage(
    content=weather_result, tool_call_id="call_123"  # 호출 ID와 일치해야 함
)
tool_message

# %% [markdown]
# ### 도구 호출 대화 흐름
#
# 도구 호출이 포함된 완전한 대화 흐름은 다음과 같이 구성됩니다:
# 1. `HumanMessage`: 사용자의 원래 질문
# 2. `AIMessage`: 모델의 도구 호출 결정 (tool_calls 포함)
# 3. `ToolMessage`: 도구 실행 결과
#
# 이 세 메시지를 모델에 전달하면, 모델은 도구 실행 결과를 바탕으로 사용자에게 최종 응답을 생성합니다.
#
# 아래 코드는 도구 호출이 포함된 완전한 대화 흐름의 예시입니다.

# %%
# 대화 계속
messages = [
    HumanMessage("서울의 날씨는 어때?"),
    ai_message,  # 모델의 도구 호출
    tool_message,  # 도구 실행 결과
]

response = model.invoke(messages)  # 모델이 결과 처리
print(response.content)

# %% [markdown]
# ## 메시지 콘텐츠
#
# 메시지의 `content` 속성은 모델로 전송되는 실제 데이터를 담습니다. 단순 문자열부터 멀티모달 콘텐츠(이미지, 오디오 등)까지 다양한 형태를 지원합니다.
#
# LangChain은 두 가지 콘텐츠 형식을 지원합니다:
# - **문자열**: 단순 텍스트 메시지
# - **콘텐츠 블록 리스트**: 텍스트, 이미지, 오디오 등을 조합한 멀티모달 메시지
#
# 아래 코드는 다양한 콘텐츠 형식을 사용하는 예시입니다.

# %%
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1-mini")

# 문자열 콘텐츠
human_message = HumanMessage("Hello, how are you?")
print("String content:", human_message.content)

# Provider 네이티브 형식 (예: OpenAI)
human_message = HumanMessage(
    content=[
        {"type": "text", "text": "다음은 어떤 내용인지 설명해줘"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://blog.langchain.com/content/images/2023/09/image.png"
            },
        },
    ]
)
print("Provider 네이티브 형식:", human_message.content)

response = model.invoke([human_message])
print(response.content)

# %%
# 표준 콘텐츠 블록 목록
human_message = HumanMessage(
    content_blocks=[
        {"type": "text", "text": "다음은 어떤 내용인지 설명해줘"},
        {
            "type": "image",
            "url": "https://blog.langchain.com/content/images/2023/09/image.png",
        },
    ]
)
print("표준 콘텐츠 블록:", human_message.content_blocks)

response = model.invoke([human_message])
print(response.content)
