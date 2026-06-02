import base64
import mimetypes
import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.infra.config.providers import infra_config
from app.infra.llm.providers import llm_provider
from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


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
        state["md_content"] = md_content
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

        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(img_file.name)+r".*?\)")
        match = reg.search(md_content)
        if not match:
            logger.warning(f"image file: {img_file.name} not found in md_content")
            continue
        start,end = match.span()
        pre_context = md_content[max(0, start-context_limit):start]
        post_context = md_content[end:min(len(md_content), end+context_limit)]
        result.append((img_file.name, str(img_file), (pre_context, post_context)))
    return result

def img_summary(img_ctx_list: list[tuple[str, str, tuple[str, str]]],stem: str) -> dict[str, str]:
    """
    图片总结服务：
    1. 调用多模态模型生成图片总结
    """
    vision_model = llm_provider.vision_chat()
    results : dict[str, str]= {}
    for img_file_name, img_file_path, context in img_ctx_list:
        apply_api_rate_limit()
        image_summary_prompt = load_prompt("image_summary" , root_folder=stem,image_content=context)
        img_path_obj = Path(img_file_path)
        img_base64 = base64.b64encode(img_path_obj.read_bytes()).decode(encoding="utf-8")
        human_msg = HumanMessage(
            content=[
                {
                    # 图片的内容
                    "type": "image_url",
                    # 图片具体内容
                    # http地址
                    # base64     data:图片类型;base64,base64字符串
                    # import mimetypes  . guess_type (文件名 带后缀名)
                    "image_url": {"url": f"data:{mimetypes.guess_type(img_file_name)[0]};base64,{img_base64}"},
                },
                # 图片对应的辅助描述
                {"type": "text", "text": f"{image_summary_prompt}"},
            ]
        )

        model_chain = vision_model|StrOutputParser()
        img_desc = model_chain.invoke([human_msg])
        results[img_file_name] = img_desc

    logger.info(f"图片总结结果: {results}")
    return results

def upload_img_and_replace(image_ctx_list: list[tuple[str, str, tuple[str, str]]],
                           image_summaries_dict: dict[str, str], md_content: str, stem: str)-> str:

    client = minio_gateway.client()
    object_list = client.list_objects(
        bucket_name=infra_config.minio.bucket_name,
        prefix=f"{minio_gateway.image_dir}/{stem}",
        recursive=True,
    )

    delete_objs = [DeleteObject(obj.object_name) for obj in object_list]

    errors = client.remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_objs,
    )

    for error in errors:
        logger.error(f"删除图片失败: {error}")

    logger.info(f"删除图片成功: {len(delete_objs)}")

    img_url_dict: dict[str, str] = {}
    for img_file_name, img_file_path, context in image_ctx_list:
        # 上传图片
        try:
            object_name = f"{minio_gateway.image_dir}/{stem}/{img_file_name}"
            client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=img_file_path,
                content_type=mimetypes.guess_type(img_file_name)[0]
            )
            img_url_dict[img_file_name] = minio_gateway.build_img_url(stem, img_file_name)
        except Exception as e:
            logger.warning(f"{img_file_name}的图片上传失败!跳过继续上传!!")

    for img_name,img_url in img_url_dict.items():
        img_desc = image_summaries_dict[img_name]

        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(img_name)+r".*?\)")

        md_content = reg.sub(lambda _:f"![{img_desc}]({img_url})", md_content)

    return md_content

def backup_new_md(md_content: str, md_path_obj: Path)->str:
    md_backup_path = md_path_obj.parent / f"{md_path_obj.stem}_new.md"

    md_backup_path.write_text(md_content, encoding="utf-8")

    return str(md_backup_path)

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

    img_ctx_list : List[tuple[str,str,tuple[str,str]]] = scan_md_imgs(md_content, md_img_dir)

    img_desc = img_summary(img_ctx_list,md_path_obj.stem)

    md_content = upload_img_and_replace(img_ctx_list, img_desc, md_content, md_path_obj.stem)

    md_path_obj = backup_new_md(md_content, md_path_obj)

    return state
