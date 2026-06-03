import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_MAX_LENGTH,CHUNK_MIN_LENGTH,CHUNK_OVERLAP
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


def _split_long_chunk(chunk, max_size)->list[dict]:
    content = chunk.get("content", "") or ""
    title = chunk.get("title")
    body = content
    if content.startswith(title):
        body = content[len(title):].lstrip()

    prefix =  title+"\n"
    available_length = max_size - len(prefix)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_length,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？"],
    )

    sub_chunks = []
    for i,split_chunk in enumerate(splitter.split_text(body),start=1):
        text = split_chunk.strip()
        if not text:
            continue
        full_text = prefix+text
        sub_chunks.append({
            "title": f"{title}_{i}" if title else f"chunk_{i}",
            "content": full_text,
            "file_title": chunk.get("file_title"),
            "parent_title": title,
            "part": i
        })

    logger.info(f"chunk split done. chunk_size:{len(text)}")

    return sub_chunks


def _merge_short_chunks(final_chunks, max_size, min_size)->list[dict]:
    final_merge_chunks = []
    start_chunk = None

    for next_chunk in final_chunks:
        if not start_chunk:
            start_chunk=next_chunk
            continue

        is_short = len(start_chunk["content"]) < min_size
        is_same_parent = start_chunk["parent_title"] == next_chunk["parent_title"]
        if is_short and is_same_parent:
            next_content_remove_title = start_chunk["content"][len(start_chunk["parent_title"])+2:]
            start_content = start_chunk.get("content")
            merge_content = start_content+"\n"+next_content_remove_title
            if len(merge_content)<=max_size:
                start_chunk["content"] = merge_content
                logger.info(f"merge chunk done. chunk_size:{len(merge_content)}")
            else:
                final_merge_chunks.append(start_chunk)
                start_chunk = next_chunk
        else:
            final_merge_chunks.append(start_chunk)
            start_chunk = next_chunk
    if start_chunk:
        final_merge_chunks.append(start_chunk)
        logger.info(f"chunk split done. chunk_size:{len(start_chunk['content'])}")
    return final_merge_chunks


def refine_chunk(chunks, max_size=CHUNK_MAX_LENGTH,min_size=CHUNK_MIN_LENGTH)->list[dict]:

    final_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > max_size:
            final_chunks.extend(_split_long_chunk(chunk, max_size))
        else:
            final_chunks.append(chunk)

    for chunk in final_chunks:
        if "parent_title" not in chunk:
            chunk['parent_title']=chunk["title"]
        if "part" not in chunk:
            chunk['part']=1


    return _merge_short_chunks(final_chunks,max_size,min_size)


def backup_chunks_json(final_chunks, md_path):
    json_path_obj = md_path.parent / f"{md_path.stem}.json"
    json_path_obj.write_text(json.dumps(final_chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(f"backup chunks json done. json_path:{json_path_obj}")

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
    final_chunks = refine_chunk(chunks)
    backup_chunks_json(final_chunks, md_path)
    state["chunks"] = final_chunks
    return state