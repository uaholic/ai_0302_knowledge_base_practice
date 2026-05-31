
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger

load_dotenv()

builder = StateGraph(ImportGraphState)

builder.add_node("node_entry", node_entry)
builder.add_node("node_pdf_to_md", node_pdf_to_md)
builder.add_node("node_md_img", node_md_img)
builder.add_node("node_document_split", node_document_split)
builder.add_node("node_item_name_recognition", node_item_name_recognition)
builder.add_node("node_bge_embedding", node_bge_embedding)
builder.add_node("node_import_milvus", node_import_milvus)

builder.set_entry_point("node_entry")


def node_entry_after(state: ImportGraphState) -> str:
    if state["is_md_read_enable"]:
        logger.info(f"node_entry节点判断的文件{state['local_file_path']}类型 md,跳转到:node_md_img 节点")
        return "node_md_img"
    elif state['is_pdf_read_enabled']:
        # true 是pdf
        logger.info(f"node_entry节点判断的文件{state['local_file_path']}类型 pdf,跳转到:node_pdf_to_md 节点")
        return "node_pdf_to_md"
    else:
        logger.warning(f"node_entry节点获取的文件: {state['local_file_path']} 无法处理对应的类型,直接跳转到END节点!")
        return END


builder.add_conditional_edges("node_entry",
                              node_entry_after,
                              {
                                     "node_md_img":"node_md_img",
                                     "node_pdf_to_md":"node_pdf_to_md",
                                     END:END
                                 })


builder.add_edge("node_pdf_to_md", "node_md_img")
builder.add_edge("node_md_img", "node_document_split")
builder.add_edge("node_document_split", "node_item_name_recognition")
builder.add_edge("node_item_name_recognition", "node_bge_embedding")
builder.add_edge("node_bge_embedding", "node_import_milvus")

kb_import_app = builder.compile()