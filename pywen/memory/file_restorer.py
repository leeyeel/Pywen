import time
import math
from pathlib import Path
from typing import List, Dict, Any


class IntelligentFileRestorer:

    def __init__(self):
        self.max_files = 20
        self.max_tokens_per_file = 8192
        self.total_token_limit = 32768
        self.scoring_weights = {
            "temporal": 0.35,
            "frequency": 0.25,
            "operation": 0.20,
            "fileType": 0.20,
        }


    def calculate_importance_score(self, metadata):
        total_score = 0.0
        temporal_score = self.calculate_temporal_score(metadata)
        total_score += temporal_score * self.scoring_weights["temporal"]
        frequency_score = self.calculate_frequency_score(metadata)
        total_score += frequency_score * self.scoring_weights["frequency"]
        operation_score = self.calculate_operation_score(metadata)
        total_score += operation_score * self.scoring_weights["operation"]
        file_type_score = self.calculate_file_type_score(metadata)
        total_score += file_type_score * self.scoring_weights["fileType"]

        return round(total_score)


    def calculate_temporal_score(self, metadata):
        now = time.time() * 1000  # in milliseconds
        hours_since_last_access = (now - metadata["lastAccessTime"]) / (1000 * 60 * 60)
        if hours_since_last_access <= 1:
            return 100
        elif hours_since_last_access <= 6:
            return 90
        elif hours_since_last_access <= 24:
            return 75
        else:
            return max(10, 75 * math.exp(-0.1 * (hours_since_last_access - 24)))


    def calculate_frequency_score(self, metadata):
        total_operations = metadata["readCount"] + metadata["writeCount"] + metadata["editCount"]
        score = min(80, total_operations * 5)
        recent_operations = metadata.get("operationsInLastHour", 0)
        score += min(20, recent_operations * 10)
        return min(100, score)


    def calculate_operation_score(self, metadata):
        score = 0
        score += metadata["writeCount"] * 15
        score += metadata["editCount"] * 10
        score += metadata["readCount"] * 3
        if metadata.get("lastOperation") == "write":
            score += 25
        elif metadata.get("lastOperation") == "edit":
            score += 15
        return min(100, score)


    def calculate_file_type_score(self, metadata):
        extension = metadata["path"].split(".")[-1].lower()
        code_extensions = {
            "js": 100, "ts": 100, "jsx": 95, "tsx": 95,
            "py": 90, "java": 85, "cpp": 85, "c": 85,
            "go": 80, "rs": 80, "php": 75, "rb": 75
        }
        
        config_extensions = {
            "json": 70, "yaml": 65, "yml": 65, "toml": 60,
            "xml": 55, "ini": 50, "env": 50, "config": 50

        }

        doc_extensions = {
            "md": 40, "txt": 30, "doc": 25, "docx": 25,
            "pdf": 20, "html": 35, "css": 45

        }

        if extension in code_extensions:
            return code_extensions[extension]
        elif extension in config_extensions:
            return config_extensions[extension]
        elif extension in doc_extensions:
            return doc_extensions[extension]
        return 30


    def find_best_fit_file(self, files, remaining_tokens):
        sorted_files = sorted(files, key=lambda f: f["score"], reverse=True)
        for f in sorted_files:
            if f["estimatedTokens"] <= remaining_tokens:
                return f
        return None


    def select_optimal_file_set(self, ranked_files):
        selected_files = []
        total_tokens = 0
        file_count = 0
        sorted_files = sorted(ranked_files, key=lambda f: f["score"], reverse=True)
        i = 0
        while i < len(sorted_files):
            file = sorted_files[i]
            if file_count >= self.max_files:
                print(f"📊 达到文件数量限制: {self.max_files}")
                break
            if file["estimatedTokens"] > self.max_tokens_per_file:
                print(f"⚠️ 文件 {file['path']} 超出单文件限制，跳过")
                i += 1
                continue
            if total_tokens + file["estimatedTokens"] > self.total_token_limit:
                print(f"📊 添加 {file['path']} 将超出总Token限制")
                remaining_tokens = self.total_token_limit - total_tokens
                alternative_file = self.find_best_fit_file(sorted_files[i+1:], remaining_tokens)
                if alternative_file:
                    selected_files.append(alternative_file)
                    total_tokens += alternative_file["estimatedTokens"]
                    file_count += 1
                    sorted_files.remove(alternative_file)
                i += 1
                continue
            selected_files.append(file)
            total_tokens += file["estimatedTokens"]
            file_count += 1
            i += 1
        return {
            "files": selected_files,
            "totalFiles": file_count,
            "totalTokens": total_tokens,
            "efficiency": (total_tokens / self.total_token_limit) * 100 if self.total_token_limit > 0 else 0
        }

 
    def file_recover(self, file_counter) -> str:
        if not file_counter:
            print("⚠️ 暂无文件记录，无法恢复。")
            return None

        # 1. 计算每条记录的 importance score
        ranked_files = []
        for meta in file_counter.values():
            meta_copy = meta.copy()              # 防止打分函数内部修改
            meta_copy["score"] = self.calculate_importance_score(meta_copy)
            ranked_files.append(meta_copy)

        # 2. 按分数+约束选最优文件集
        selected = self.select_optimal_file_set(ranked_files)

        # 3. 按分数倒序读取内容
        contents = []
        base_path = Path.cwd().resolve()
        for file_info in sorted(selected["files"], key=lambda f: f["score"], reverse=True):
            full_path = base_path / file_info["path"]
            try:
                text = full_path.read_text(encoding="utf-8")
                contents.append(
                    f"File: {file_info['path']}\n"
                    f"Score: {file_info['score']}\n"
                    f"Content:\n{text}\n\n"
                )
            except Exception as e:
                contents.append(
                    f"File: {file_info['path']}\n"
                    f"Error reading: {e}\n\n"
                )

        return "".join(contents)