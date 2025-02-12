from typing import Any, Dict, List, Union
import time
from functools import wraps
import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.memory import ChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from arxivTool.arxivTool import get_embeddings

load_dotenv("arxivTool/.env")
 
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE")

QWEN_API_BASE = os.getenv("QWEN_API_BASE")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
                         
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4j_USER")
NEO4J_PASSWORD = os.getenv("NEO4j_PASSWORD")

# Connection to Neo4j
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, database="neo4j"
)

history_graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, database="history"
)
print("Connected to Neo4j MaterialsKG and History databases")

# Cypher validation tool for relationship directions
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)

cypher_llm = ChatOpenAI(
    model="deepseek-v3",
    openai_api_base=QWEN_API_BASE,
    openai_api_key=QWEN_API_KEY,
)

qa_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    openai_api_base=SILICONFLOW_API_BASE,
    openai_api_key=SILICONFLOW_API_KEY,
    temperature=0.7,
)


def convert_messages(input: List[Dict[str, Any]]) -> ChatMessageHistory:
    history = ChatMessageHistory()
    for item in input:
        history.add_user_message(item["result"]["question"])
        history.add_ai_message(item["result"]["answer"])
    return history


def get_history(input: Dict[str, Any]) -> ChatMessageHistory:
    input1 = input
    input1.pop("question")
    # Lookback conversation window
    window = 3
    data = history_graph.query(
        """
    MATCH (u:User {id:$user_id})-[:HAS_SESSION]->(s:Session {id:$session_id}),
                       (s)-[:LAST_MESSAGE]->(last_message)
    MATCH p=(last_message)<-[:NEXT*0.."""
        + str(window)
        + """]-()
    WITH p, length(p) AS length
    ORDER BY length DESC LIMIT 1
    UNWIND reverse(nodes(p)) AS node
    MATCH (node)-[:HAS_ANSWER]->(answer)
    RETURN {question:node.text, answer:answer.text} AS result
 """,
        params=input1,
    )
    history = convert_messages(data)
    return history.messages


def save_history(input):
    print(input)
    # 只保留output字段
    if input.get("function_response"):
        input.pop("function_response")
    if input.get("arxiv_function_response"):
        input.pop("arxiv_function_response")

    # store history to database
    history_graph.query(
        """MERGE (u:User {id: $user_id})
WITH u                
OPTIONAL MATCH (u)-[:HAS_SESSION]->(s:Session{id: $session_id}),
                (s)-[l:LAST_MESSAGE]->(last_message)
FOREACH (_ IN CASE WHEN last_message IS NULL THEN [1] ELSE [] END |
CREATE (u)-[:HAS_SESSION]->(s1:Session {id:$session_id}),
    (s1)-[:LAST_MESSAGE]->(q:Question {text:$question, cypher:$query, date:datetime()}),
        (q)-[:HAS_ANSWER]->(:Answer {text:$output}))                                
FOREACH (_ IN CASE WHEN last_message IS NOT NULL THEN [1] ELSE [] END |
CREATE (last_message)-[:NEXT]->(q:Question 
                {text:$question, cypher:$query, date:datetime()}),
                (q)-[:HAS_ANSWER]->(:Answer {text:$output}),
                (s)-[:LAST_MESSAGE]->(q)
DELETE l)                """,
        params=input,
    )

    # Return LLM response to the chain
    return input["output"]

def retry_with_backoff(max_retries=3, initial_wait=1, backoff_factor=2):
    """
    装饰器：实现指数退避重试机制
    max_retries: 最大重试次数
    initial_wait: 初始等待时间（秒）
    backoff_factor: 退避因子，下次等待时间 = 当前等待时间 * backoff_factor
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 表示包含第一次尝试
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:  # 如果是最后一次尝试
                        print(f"Failed after {max_retries} retries. Final error: {str(e)}")
                        raise last_exception
                    
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    wait_time *= backoff_factor  # 增加等待时间
            
            raise last_exception
        return wrapper
    return decorator

# 使用装饰器为get_embeddings添加重试机制
@retry_with_backoff(max_retries=3, initial_wait=2, backoff_factor=2)
def get_embeddings_with_retry(texts: List[str], token: str) -> List[List[float]]:
    return get_embeddings(texts, token=token)


def get_cypher_examples(input: Dict[str, Any]):
    question = input["question"]
    try:
        user_question_embedding = get_embeddings_with_retry([question], token=SILICONFLOW_API_KEY)[0]
    except Exception as e:
        print(f"Failed to get embeddings after all retries: {str(e)}")
        # 这里可以添加失败后的处理逻辑
        raise

    # Read the cypher_templates_with_embeddings.json file
    with open(
        "templates/cypher_templates_with_embeddings.json", "r", encoding="utf-8"
    ) as file:
        data = json.load(file)

    cypher_templates_embeddings = []
    # Convert embeddings to numpy arrays
    for record in data["queries"]:
        if "embedding" in record:
            cypher_templates_embeddings.append(record["embedding"])
        else:
            print(
                f"Warning: record with id {record.get('id', 'unknown')} does not have an embedding."
            )
    cypher_templates_embeddings = np.array(cypher_templates_embeddings)

    # 点乘计算相似度
    similarities = np.dot(cypher_templates_embeddings, user_question_embedding)
    # print(similarities)

    # 选择相似度最高的cypher语句
    cypher = data["queries"][np.argmax(similarities)]["cypher"]
    # print(f"Most similar cypher template: {cypher}")
    return cypher


# Generate Cypher statement based on natural language input
cypher_template = """Based on the Neo4j graph schema and Examples below, write a Cypher query that would answer the user's question.
Please ensure your response consists only of the Cypher query without any additional text or explanation.

{schema}

Examples:
{cypher_examples}

Question: {question}

Cypher query:
""" 

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        ("human", cypher_template),
    ]
)

cypher_response = (
    RunnablePassthrough.assign(
        schema=lambda _: graph.get_schema,
        cypher_examples=get_cypher_examples,
    )
    | cypher_prompt
    | cypher_llm.bind(stop=["\nCypherResult:"])
    | StrOutputParser()
)

# Generate natural language response based on database results
response_system = """You are a topological materials AI assistant. Follow these response priorities:

1. **Database-Driven Response**:
   When exact matches found in database:
   - Show material identifiers, topological indices, and experimental data
   - Highlight key parameters in **bold**
   - Present verified band structure and symmetry data

2. **Literature & Reasoning Based Response**:
   When database results are insufficient:
   - Start with any available verified data
   - Apply topological band theory principles
   - Consider symmetry indicators and similar materials
   - Support reasoning with recent literature
   - Clearly mark theoretical predictions vs. experimental results

Format responses as: material basics → topology → evidence → applications
Ask for clarification if material conditions or investigation context is unclear.
"""

response_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=response_system),
        HumanMessagePromptTemplate.from_template("{question}"),
        MessagesPlaceholder(variable_name="function_response"),
        MessagesPlaceholder(variable_name="arxiv_function_response"),
    ]
)


def get_function_response(
    query: str, question: str
) -> List[Union[AIMessage, ToolMessage]]:
    # 如果生成的cypher语句异常，则赋值为空字符串
    try:
        context = graph.query(cypher_validation(query))
        print(f"Cypher: {query}")
    except Exception as e:
        context = ""
        print(f"Error occurred: {e}")
        
    DATABASE_TOOL_ID = "call_H7fABDuzEau48T10Qn0Lsh0D"
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": DATABASE_TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetDBInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(context), tool_call_id=DATABASE_TOOL_ID),
    ]
    return messages

def get_arxiv_function_response(
    question: str, arxiv_context: str
) -> List[Union[AIMessage, ToolMessage]]:
    ARXIV_TOOL_ID = "arxiv_H7fABDuzEau48T10Qn0Lsh0E"  # 为 arXiv 工具定义一个唯一的 ID

    # 构建返回的消息列表
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": ARXIV_TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetArxivInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(arxiv_context), tool_call_id=ARXIV_TOOL_ID),
    ]

    return messages


chain = (
    RunnablePassthrough.assign(query=cypher_response)
    | RunnablePassthrough.assign(
        function_response=lambda x: get_function_response(x["query"], x["question"]),
    )
    | RunnablePassthrough.assign(
        arxiv_function_response=lambda x: get_arxiv_function_response(
            x["question"], x["arxiv_context"]
        ),
    )
    | RunnablePassthrough.assign(
        output=response_prompt | qa_llm | StrOutputParser(),
    )
    | save_history
)

# Add typing for input


class Question(BaseModel):
    question: str
    user_id: str
    session_id: str
    arxiv_context: str

chain = chain.with_types(input_type=Question)
