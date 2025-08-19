from pathlib import Path
from loguru import logger


def get_directory_metadata(directory: str):
    metadata = []
    dir_path = Path(directory)
    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                md = {
                    "path": str(file_path.relative_to(dir_path)),
                    "lastAccessTime": stat.st_atime * 1000,  # in milliseconds
                    "readCount": 0,  # Default, as no tracking
                    "writeCount": 0,
                    "editCount": 0,
                    "operationsInLastHour": 0,
                    "lastOperation": "unknown",
                    "estimatedTokens": stat.st_size // 4  # Rough estimate
                }
                metadata.append(md)
            except OSError:
                continue
    return metadata[:10]


def file_recover():
    logger.debug(f"Current working directory: {Path.cwd()}")
    directory = Path.cwd()
    logger.debug(f"Recovering files from {directory}")
    metadatas = get_directory_metadata(directory)
    logger.debug(f"Metadatas: {metadatas}")
    # ranked_files = []
    # for md in metadatas:
    #     md_copy = md.copy()
    #     score = self.file_restorer.calculate_importance_score(md_copy)
    #     md_copy["score"] = score
    #     ranked_files.append(md_copy)
    # selected = self.file_restorer.select_optimal_file_set(ranked_files)
    # # Sort selected files by score descending
    # sorted_selected = sorted(selected["files"], key=lambda f: f["score"], reverse=True)
    # # Read contents
    # dir_path = Path(directory).resolve()
    # contents = []
    # for file in sorted_selected:
    #     full_path = dir_path / file["path"]
    #     try:
    #         content = full_path.read_text(encoding="utf-8")
    #         contents.append(f"File: {file['path']}\nScore: {file['score']}\nContent:\n{content}\n\n")
    #     except Exception as e:
    #         contents.append(f"File: {file['path']}\nError reading: {str(e)}\n\n")
    # return "".join(contents)

# file_recover()

from transformers.utils import default_cache_path
print(default_cache_path)