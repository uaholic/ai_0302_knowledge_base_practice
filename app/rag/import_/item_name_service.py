from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_state(state: ImportGraphState) -> tuple[list[dict], str]:
    chunks = state.get("chunks")
    file_title = state.get('file_title')
    if not chunks:
        logger.error("No chunks in state,failed")
        raise ValueError("No chunks in state,failed")
    if not file_title:
        file_title = chunks[0]['file_title']
        state['file_title'] = file_title
    return chunks, file_title


def build_context(chunks: list[dict]) -> str:
    top_chunks = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    context = ""
    for i, chunk in enumerate(top_chunks, start=1):
        context += f"切片：{i} 标题：{chunk['title']} 父标题：{chunk['parent_title']} 内容：{chunk['content']}\n"

    final_context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]

    return final_context


def recognize_item_name(context: str, file_title: str) -> str:
    chat_model = llm_provider.chat()
    system_prompt_str = load_prompt("product_recognition_system")
    human_prompt_str = load_prompt("item_name_recognition", file_title=file_title, context=context)
    message = [
        SystemMessage(content=system_prompt_str),
        HumanMessage(content=human_prompt_str),
    ]

    chains = chat_model|StrOutputParser()
    item_name = chains.invoke(input=message)
    logger.warning("识别主体名称：{}".format(item_name))

    if not item_name:
        item_name=file_title

    return item_name


def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    chunks, file_title = validate_state(state)
    context = build_context(chunks)
    item_name = recognize_item_name(context, file_title)

    logger.warning("识别主体名称：{}".format(item_name))
    return state
