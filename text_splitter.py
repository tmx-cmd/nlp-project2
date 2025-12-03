from typing import List, Dict
from tqdm import tqdm


class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """将文本切分为块

        要求：
        1. 将文本按照chunk_size切分为多个块
        2. 相邻块之间要有chunk_overlap的重叠（用于保持上下文连续性）
        3. 尽量在句子边界处切分（查找句子结束符：。！？.!?\n\n）
        4. 返回切分后的文本块列表
        """
        if not text:
            return []

        # 定义句子结束符
        sentence_endings = "。！？.!?\n\n"

        # 按句子结束符分割文本
        sentences = []
        current_sentence = ""
        i = 0

        while i < len(text):
            char = text[i]
            current_sentence += char

            # 检查是否是句子结束符
            if char in sentence_endings:
                # 查找连续的结束符（如\n\n）
                j = i + 1
                while j < len(text) and text[j] in sentence_endings and text[j] == char:
                    current_sentence += text[j]
                    j += 1
                i = j - 1

                # 添加句子（去除首尾空白）
                sentence = current_sentence.strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = ""
            i += 1

        # 处理最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # 如果没有找到句子边界，按字符分割
        if not sentences:
            sentences = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # 将句子组合成块
        chunks = []
        current_chunk = ""
        overlap_text = ""

        for sentence in sentences:
            # 如果当前句子加上现有块会超过chunk_size，开始新块
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # 保留overlap_text用于下一个块
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence

            # 更新overlap_text（取当前块的后chunk_overlap个字符）
            if len(current_chunk) >= self.chunk_overlap:
                overlap_text = current_chunk[-self.chunk_overlap:]
            else:
                overlap_text = current_chunk

        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 如果块太小，尝试合并
        if len(chunks) > 1:
            merged_chunks = []
            temp_chunk = ""

            for chunk in chunks:
                if len(temp_chunk) + len(chunk) <= self.chunk_size:
                    temp_chunk += chunk
                else:
                    if temp_chunk:
                        merged_chunks.append(temp_chunk)
                    temp_chunk = chunk

            if temp_chunk:
                merged_chunks.append(temp_chunk)

            chunks = merged_chunks

        return chunks

    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """切分多个文档。
        对于PDF和PPT，已经按页/幻灯片分割，不再进行二次切分
        对于DOCX和TXT，进行文本切分
        """
        chunks_with_metadata = []

        for doc in tqdm(documents, desc="处理文档", unit="文档"):
            content = doc.get("content", "")
            filetype = doc.get("filetype", "")

            if filetype in [".pdf", ".pptx"]:
                chunk_data = {
                    "content": content,
                    "filename": doc.get("filename", "unknown"),
                    "filepath": doc.get("filepath", ""),
                    "filetype": filetype,
                    "page_number": doc.get("page_number", 0),
                    "chunk_id": 0,
                    "images": doc.get("images", []),
                }
                chunks_with_metadata.append(chunk_data)

            elif filetype in [".docx", ".txt"]:
                chunks = self.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "content": chunk,
                        "filename": doc.get("filename", "unknown"),
                        "filepath": doc.get("filepath", ""),
                        "filetype": filetype,
                        "page_number": 0,
                        "chunk_id": i,
                        "images": [],
                    }
                    chunks_with_metadata.append(chunk_data)

        print(f"\n文档处理完成，共 {len(chunks_with_metadata)} 个块")
        return chunks_with_metadata
