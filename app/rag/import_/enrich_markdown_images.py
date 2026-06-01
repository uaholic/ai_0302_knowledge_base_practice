import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger


def load_md_img_dir(state: ImportGraphState) -> tuple[str, Path, Path]:
    md_path = state['md_path']
    if not md_path:
        logger.error("md_path is empty")
        raise ValueError("md_path is empty")

    md_path_obj = Path(md_path)
    md_content = state['md_content']
    if not md_content:
        logger.warning(f"md_content is empty,load from md_path: {md_path}")
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"md_content is empty and load from md_path: {md_path} failed")
            raise ValueError(f"md_content is empty and load from md_path: {md_path} failed")

    md_img_dir = md_path_obj.parent / "images"

    return md_content, md_path_obj, md_img_dir


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def scan_md_imgs(md_content: str, img_dir: Path, context_limit: int = 100) -> list[tuple[str, str, tuple[str, str]]]:
    result = []
    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warning(f"unsupported image file: {img_file}")
            continue

        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(img_file.name)+".*?\)")
        match = reg.search(md_content)
        if not match:
            logger.warning(f"image file: {img_file} not found in md_content")
            continue
        start,end = match.span()
        pre_context = md_content[max(0, start-context_limit):start]
        post_context = md_content[end:min(len(md_content), end+context_limit)]
        result.append((img_file.name, str(img_file), (pre_context, post_context)))
    return result


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    md_content, md_path_obj, md_img_dir =load_md_img_dir(state)
    if not any(md_img_dir.iterdir()):
        # 空文件夹
        logger.warning(f"当前{md_content}没有图片,无需图片处理!正常进入下一个节点!!")
        return state

    scan_md_imgs(md_content, md_img_dir)
    return state
