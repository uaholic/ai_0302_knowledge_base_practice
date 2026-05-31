import copy
import json
from typing import TypedDict
from app.shared.runtime.logger import logger


class ImportGraphState(TypedDict):

    task_id:str

    is_md_read_enable:bool
    is_pdf_read_enable:bool

    local_file_path:str
    local_dir:str
    md_path:str
    pdf_path:str
    file_title:str

    md_content:str
    item_name:str
    trunks:list
    embedding_content:list


default_state: ImportGraphState = {
    "task_id": "",
    "is_md_read_enable": True,
    "is_pdf_read_enable": True,
    "local_file_path": "",
    "local_dir": "",
    "md_path": "",
    "pdf_path": "",
    "file_title": "",
    "md_content": "",
    "item_name": "",
    "trunks": [],
    "embedding_content": [],
}

def create_default_state(**overrides) -> ImportGraphState:
    copy_state = copy.deepcopy(default_state)

    copy_state.update(overrides)

    return copy_state

def get_default_state() -> ImportGraphState:
    return copy.deepcopy(default_state)




if __name__ == '__main__':
    state = create_default_state(task_id="task_007")
    logger.info(f"测试复制方法: \n {json.dumps(state, ensure_ascii=False, indent=4)}")

    state1 = get_default_state()
    logger.info(f"测试复制方法: \n {json.dumps(state1, ensure_ascii=False, indent=4)}")
