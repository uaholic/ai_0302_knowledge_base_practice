import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger


def load_md_content(state: ImportGraphState)->tuple[str, str,Path]:
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")

    if not md_content:
        logger.warning("md_content is empty")
        md_content = Path(md_path).read_text(encoding="utf-8")
        state["md_content"] = md_content
        if not md_content:
            logger.error("md_content is empty and load from md_path failed")
            raise ValueError("md_content is empty and load from md_path failed")

    if not file_title:
        logger.warning("file_title is empty")
        file_title = Path(md_path).stem
        state["file_title"] = file_title

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content, file_title, Path(md_path)

def split_by_title(md_content: str, file_title: str) -> list[dict]:
    reg = re.compile(r"^\s*#{1,6}\s.+")

    lines = md_content.split("\n")

    chunks: list[dict] = []

    cur_title=None

    cur_content_lines: list[str] = []

    is_code_block = False

    for line in lines:
        line=line.strip()

        if not line:
            logger.warning(f"当前行为空行，跳过")
            continue

        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            cur_content_lines.append(line)
            continue

        if reg.match(line) and not is_code_block:
            if cur_title and len(cur_content_lines)>1:
                chunks.append({
                    "title": cur_title,
                    "content": "\n".join(cur_content_lines),
                    "file_title": file_title
                })

            cur_title = line
            cur_content_lines = [line]
        else:
            cur_content_lines.append(line)

    if cur_title and len(cur_content_lines)>1:
        chunks.append({
            "title": cur_title,
            "content": "\n".join(cur_content_lines),
            "file_title": file_title
        })

    if len(chunks)==0:
        chunks.append({
            "title": file_title,
            "content": md_content,
            "file_title": file_title
        })

    logger.info(f"split process done. chunks_size:{len(chunks)}")

    return chunks


def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    md_content, file_title, md_path = load_md_content(state)
    chunks = split_by_title(md_content, file_title)
    return state