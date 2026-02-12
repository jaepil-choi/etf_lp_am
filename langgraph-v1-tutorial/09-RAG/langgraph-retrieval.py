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
# # Retrieval (검색)
#
# 대형 언어 모델(LLM)은 강력하지만 두 가지 주요 한계가 있습니다:
#
# - **제한된 컨텍스트**: 전체 코퍼스를 한 번에 처리할 수 없음
# - **정적 지식**: 훈련 데이터가 특정 시점에 고정됨
#
# **Retrieval(검색)**은 쿼리 시점에 관련 외부 지식을 가져와서 이러한 문제를 해결합니다. 이것이 **Retrieval-Augmented Generation (RAG)**의 기초입니다: 컨텍스트별 정보로 LLM의 답변을 향상시킵니다.
#
# ## RAG의 핵심 개념
#
# RAG는 검색과 생성을 결합하여 근거 있고 컨텍스트를 인식하는 답변을 생성합니다.

# %% [markdown]
# ## 사전 준비
#
# 환경 변수를 설정합니다.

# %%
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv(override=True)

# %%
# LangSmith 추적을 설정합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LangGraph-RAG")

# %% [markdown]
# ## 필요한 라이브러리 설치
#
# ```bash
# pip install langchain-openai faiss-cpu langchain-community
# ```

# %% [markdown]
# ## 지식 베이스 구축
#
# **지식 베이스**는 검색 중에 사용되는 문서 또는 구조화된 데이터의 저장소입니다.
#
# ### 검색 파이프라인
#
# 일반적인 검색 워크플로우:
#
# 1. **문서 로드**: 외부 소스에서 데이터 수집
# 2. **청크로 분할**: 큰 문서를 작은 조각으로 분할
# 3. **임베딩 생성**: 텍스트를 벡터로 변환
# 4. **벡터 저장소에 저장**: 벡터를 검색 가능한 데이터베이스에 저장
# 5. **쿼리 임베딩**: 사용자 질문을 벡터로 변환
# 6. **검색**: 유사한 벡터 찾기
# 7. **LLM에 전달**: 검색된 정보로 답변 생성

# %% [markdown]
# ### 간단한 지식 베이스 생성

# %%
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 샘플 문서
documents = [
    Document(
        page_content="LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다. 도구, 에이전트, 메모리 등을 제공합니다.",
        metadata={"source": "docs", "topic": "langchain"}
    ),
    Document(
        page_content="LangGraph는 상태 기반 에이전트를 구축하기 위한 라이브러리입니다. LangChain 위에 구축되었습니다.",
        metadata={"source": "docs", "topic": "langgraph"}
    ),
    Document(
        page_content="RAG는 Retrieval-Augmented Generation의 약자로, 외부 지식을 검색하여 LLM 응답을 향상시킵니다.",
        metadata={"source": "docs", "topic": "rag"}
    ),
    Document(
        page_content="벡터 데이터베이스는 임베딩을 저장하고 검색하는 데 특화된 데이터베이스입니다. FAISS, Pinecone 등이 있습니다.",
        metadata={"source": "docs", "topic": "vectordb"}
    ),
]

# 텍스트 분할기 (큰 문서의 경우)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 임베딩 및 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

print(f"Created vector store with {len(splits)} documents")

# %% [markdown]
# ### 벡터 저장소에서 검색

# %%
# 검색 테스트
query = "LangGraph란 무엇인가?"
docs = vectorstore.similarity_search(query, k=2)

print(f"Query: {query}\n")
for i, doc in enumerate(docs, 1):
    print(f"Result {i}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# %% [markdown]
# ## RAG 아키텍처
#
# RAG는 시스템의 필요에 따라 여러 방식으로 구현할 수 있습니다.
#
# | 아키텍처 | 설명 | 제어 | 유연성 | 지연 시간 | 사용 사례 |
# |---------|------|------|-------|----------|----------|
# | **2-Step RAG** | 검색이 항상 생성 전에 발생. 간단하고 예측 가능 | ✅ 높음 | ❌ 낮음 | ⚡ 빠름 | FAQ, 문서 봇 |
# | **Agentic RAG** | LLM 기반 에이전트가 추론 중 검색 시점과 방법을 결정 | ❌ 낮음 | ✅ 높음 | ⏳ 가변적 | 다중 도구를 사용하는 연구 어시스턴트 |
# | **Hybrid** | 검증 단계를 포함하여 두 접근 방식의 특성을 결합 | ⚖️ 중간 | ⚖️ 중간 | ⏳ 가변적 | 품질 검증이 필요한 도메인별 Q&A |

# %% [markdown]
# ## 2-Step RAG
#
# **2-Step RAG**에서는 검색 단계가 항상 생성 단계 전에 실행됩니다. 이 아키텍처는 간단하고 예측 가능합니다.

# %%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 프롬프트 템플릿
template = """다음 컨텍스트를 사용하여 질문에 답하세요:

컨텍스트:
{context}

질문: {question}

답변:"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 문서를 문자열로 포맷팅하는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 구성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 테스트
question = "LangGraph는 무엇이고 어떻게 사용하나요?"
answer = rag_chain.invoke(question)

print(f"Question: {question}\n")
print(f"Answer: {answer}")

# %% [markdown]
# ### 2-Step RAG의 장단점
#
# **장점**:
# - 간단하고 예측 가능
# - 빠른 응답 시간
# - 구현이 쉬움
#
# **단점**:
# - 유연성 부족
# - 항상 검색을 수행 (불필요한 경우에도)
# - 다중 단계 추론 제한적

# %% [markdown]
# ## Agentic RAG
#
# **Agentic RAG**는 에이전트가 추론 과정에서 **언제** 그리고 **어떻게** 정보를 검색할지 결정합니다.

# %%
from langchain.agents import create_agent
from langchain.tools import tool

# 검색 도구 생성
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = vectorstore.similarity_search(query, k=2)
    return "\n\n".join([doc.page_content for doc in docs])

# Agentic RAG 에이전트
agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],
    system_prompt="""당신은 도움이 되는 AI 어시스턴트입니다.
    질문에 답하기 위해 외부 정보가 필요하면 search_knowledge_base 도구를 사용하세요.
    검색 결과를 기반으로 명확하고 정확한 답변을 제공하세요."""
)

# 테스트
result = agent.invoke({
    "messages": [{"role": "user", "content": "RAG에 대해 설명해주세요."}]
})

print(result["messages"][-1].content)


# %% [markdown]
# ### Agentic RAG의 장단점
#
# **장점**:
# - 높은 유연성
# - 필요할 때만 검색
# - 다중 단계 추론 가능
# - 여러 도구 조합 가능
#
# **단점**:
# - 예측 불가능한 지연 시간
# - 더 많은 LLM 호출
# - 복잡한 구현

# %% [markdown]
# ### 다중 도구를 사용하는 Agentic RAG

# %%
@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_date() -> str:
    """Get the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# 여러 도구를 가진 에이전트
multi_tool_agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, calculate, get_current_date],
    system_prompt="""당신은 다재다능한 AI 어시스턴트입니다.
    - 지식 기반 질문: search_knowledge_base 사용
    - 계산: calculate 사용
    - 날짜 정보: get_current_date 사용
    
    적절한 도구를 선택하여 사용자 질문에 답하세요."""
)

# 복합 질문 테스트
questions = [
    "LangChain이 무엇인지 설명하고, 오늘 날짜를 알려주세요.",
    "벡터 데이터베이스에 대해 설명하고, 10 * 25를 계산해주세요.",
]

for question in questions:
    print(f"\n=== Question: {question} ===")
    result = multi_tool_agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    print(f"Answer: {result['messages'][-1].content}")

# %% [markdown]
# ## Hybrid RAG
#
# Hybrid RAG는 2-Step과 Agentic RAG의 특성을 결합합니다. 쿼리 전처리, 검색 검증, 생성 후 검사 등의 중간 단계를 도입합니다.

# %%
from langchain_core.prompts import ChatPromptTemplate

# 1. 쿼리 향상
query_enhancement_prompt = ChatPromptTemplate.from_template(
    """사용자 질문을 더 명확하고 구체적으로 재작성하세요.
    검색에 최적화된 형태로 만드세요.
    
    원본 질문: {question}
    
    향상된 질문:"""
)

# 2. 검색 검증
validation_prompt = ChatPromptTemplate.from_template(
    """다음 검색 결과가 질문에 답하기에 충분한지 평가하세요.
    
    질문: {question}
    
    검색 결과:
    {results}
    
    충분하면 'YES', 불충분하면 'NO'라고만 답하세요."""
)

# 3. 답변 생성
answer_prompt = ChatPromptTemplate.from_template(
    """검색된 정보를 바탕으로 질문에 답하세요.
    
    질문: {question}
    
    컨텍스트:
    {context}
    
    답변:"""
)

def hybrid_rag(question: str, max_iterations: int = 2) -> str:
    """Hybrid RAG with query enhancement and validation."""
    
    # 1. 쿼리 향상
    enhanced_query = (query_enhancement_prompt | llm | StrOutputParser()).invoke(
        {"question": question}
    )
    print(f"Enhanced query: {enhanced_query}\n")
    
    for i in range(max_iterations):
        # 2. 검색
        docs = vectorstore.similarity_search(enhanced_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print(f"Iteration {i+1}: Retrieved {len(docs)} documents")
        
        # 3. 검색 검증
        validation = (validation_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "results": context
        })
        
        if "YES" in validation.upper():
            print("Validation: PASSED\n")
            break
        else:
            print("Validation: FAILED - Refining query\n")
            # 쿼리 재작성 로직 (단순화)
            enhanced_query = f"{enhanced_query} (more specific)"
    
    # 4. 답변 생성
    answer = (answer_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "context": context
    })
    
    return answer

# 테스트
question = "LangChain과 관련된 기술들은?"
answer = hybrid_rag(question)

print(f"\nFinal Answer:\n{answer}")

# %% [markdown]
# ## 실용적인 예제: 문서 Q&A 시스템
#
# 실제 문서를 로드하고 검색하는 완전한 RAG 시스템을 구축해봅시다.

# %%
# 더 많은 문서 추가
extended_documents = [
    Document(
        page_content="""Python은 1991년 Guido van Rossum이 개발한 고수준 프로그래밍 언어입니다.
        간결하고 읽기 쉬운 문법으로 유명하며, 데이터 과학, 웹 개발, 자동화 등 다양한 분야에서 사용됩니다.""",
        metadata={"source": "programming", "topic": "python"}
    ),
    Document(
        page_content="""머신러닝은 데이터로부터 학습하는 알고리즘을 연구하는 인공지능의 한 분야입니다.
        지도 학습, 비지도 학습, 강화 학습 등의 방법이 있으며, 이미지 인식, 자연어 처리 등에 활용됩니다.""",
        metadata={"source": "ai", "topic": "machine-learning"}
    ),
    Document(
        page_content="""Transformer는 2017년 Google이 발표한 딥러닝 아키텍처입니다.
        Self-attention 메커니즘을 사용하며, GPT, BERT 등 현대 LLM의 기초가 되었습니다.""",
        metadata={"source": "ai", "topic": "transformer"}
    ),
    Document(
        page_content="""임베딩(Embedding)은 텍스트를 고차원 벡터 공간의 점으로 표현하는 기법입니다.
        의미적으로 유사한 텍스트는 벡터 공간에서 가까이 위치합니다. Word2Vec, BERT embeddings 등이 있습니다.""",
        metadata={"source": "nlp", "topic": "embeddings"}
    ),
]

# 확장된 벡터 저장소 생성
all_documents = documents + extended_documents
extended_vectorstore = FAISS.from_documents(all_documents, embeddings)

print(f"Extended knowledge base with {len(all_documents)} documents")

# %% [markdown]
# ### 대화형 Q&A 시스템

# %%
from langchain.agents import create_agent

@tool
def search_documents(query: str) -> str:
    """Search documents in the knowledge base."""
    docs = extended_vectorstore.similarity_search(query, k=3)
    results = []
    for doc in docs:
        results.append(f"[{doc.metadata['topic']}] {doc.page_content}")
    return "\n\n".join(results)

# 문서 Q&A 에이전트
qa_agent = create_agent(
    model=llm,
    tools=[search_documents],
    system_prompt="""당신은 지식 기반 Q&A 어시스턴트입니다.
    
    사용자 질문에 답할 때:
    1. search_documents 도구를 사용하여 관련 정보를 찾으세요
    2. 검색된 문서를 바탕으로 명확하고 정확한 답변을 제공하세요
    3. 답변의 출처(토픽)를 언급하세요
    4. 검색 결과가 불충분하면 이를 사용자에게 알리세요
    
    한국어로 답변하세요."""
)

# 테스트 질문들
test_questions = [
    "Python에 대해 설명해주세요.",
    "Transformer 아키텍처는 무엇인가요?",
    "임베딩이 무엇이고 어떻게 사용되나요?",
    "RAG와 머신러닝의 관계는?",
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")
    
    result = qa_agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    
    print(f"A: {result['messages'][-1].content}")


# %% [markdown]
# ## 고급 기능: 메타데이터 필터링
#
# 메타데이터를 사용하여 검색을 더 정확하게 만들 수 있습니다.

# %%
@tool
def search_by_topic(query: str, topic: str) -> str:
    """Search documents filtered by topic.
    
    Available topics: python, machine-learning, transformer, embeddings, langchain, langgraph, rag, vectordb
    """
    # 메타데이터 필터링
    docs = extended_vectorstore.similarity_search(
        query,
        k=3,
        filter={"topic": topic}
    )
    
    if not docs:
        return f"No documents found for topic: {topic}"
    
    results = []
    for doc in docs:
        results.append(doc.page_content)
    return "\n\n".join(results)

# 토픽 필터링 에이전트
filtered_agent = create_agent(
    model=llm,
    tools=[search_by_topic],
    system_prompt="""당신은 토픽별 문서 검색 어시스턴트입니다.
    
    사용자 질문에서 관련 토픽을 파악하고 search_by_topic을 사용하세요.
    가능한 토픽: python, machine-learning, transformer, embeddings, langchain, langgraph, rag, vectordb
    """
)

# 테스트
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": "Python에 대한 정보를 찾아주세요."}]
})

print(result["messages"][-1].content)

# %% [markdown]
# ## 모범 사례
#
# ### 1. 적절한 청크 크기 선택
#
# - 너무 작으면: 컨텍스트 부족
# - 너무 크면: 노이즈 증가, 검색 정확도 감소
# - 일반적으로 500-1000 토큰이 적절

# %%
# 좋은 예
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 적절한 크기
    chunk_overlap=50     # 컨텍스트 연속성 유지
)

# %% [markdown]
# ### 2. 검색 결과 수 조정
#
# - k 값이 너무 작으면: 중요한 정보 누락
# - k 값이 너무 크면: 노이즈 증가, 비용 증가

# %%
# 질문 유형에 따라 k 조정
simple_query_k = 2  # 간단한 질문
complex_query_k = 5  # 복잡한 질문

# %% [markdown]
# ### 3. 프롬프트 최적화
#
# 명확한 지시사항을 제공하세요.

# %%
# 좋은 프롬프트
good_prompt = """다음 컨텍스트를 사용하여 질문에 답하세요.
컨텍스트에 답이 없으면 '정보가 부족합니다'라고 답하세요.
추측하지 마세요.

컨텍스트: {context}
질문: {question}
답변:"""

# %% [markdown]
# ### 4. 메타데이터 활용
#
# 메타데이터로 검색 품질을 향상시키세요.

# %%
# 유용한 메타데이터
metadata = {
    "source": "document_name.pdf",
    "page": 5,
    "topic": "machine-learning",
    "date": "2024-01-01",
    "author": "John Doe"
}

# %% [markdown]
# ## 요약
#
# ### RAG 아키텍처 선택 가이드
#
# **2-Step RAG를 선택하는 경우**:
# - 간단한 FAQ 시스템
# - 예측 가능한 지연 시간이 중요
# - 구현 복잡도를 최소화하고 싶을 때
#
# **Agentic RAG를 선택하는 경우**:
# - 복잡한 다단계 추론 필요
# - 여러 도구와 데이터 소스 활용
# - 유연성이 중요할 때
#
# **Hybrid RAG를 선택하는 경우**:
# - 품질 검증이 중요
# - 쿼리 향상이 필요
# - 중간 수준의 복잡도와 제어가 필요할 때
#
# ### 핵심 요점
#
# 1. **지식 베이스 구축**: 문서 로드 → 청크 분할 → 임베딩 → 벡터 저장소
# 2. **검색**: 쿼리 임베딩 → 유사도 검색 → 관련 문서 반환
# 3. **생성**: 검색된 컨텍스트 + 질문 → LLM → 답변
# 4. **최적화**: 청크 크기, k 값, 프롬프트, 메타데이터 활용
#
# RAG는 LLM의 지식을 확장하고 최신 정보를 제공하는 강력한 패턴입니다.
