import os
import uuid
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
)


class VectorStore:

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        # 初始化ChromaDB
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"description": "课程材料向量数据库"}
        )

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示

        TODO: 使用OpenAI API获取文本的embedding向量



        """
        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """添加文档块到向量数据库

        要求：
        1. 遍历文档块
        2. 获取文档块内容
        3. 获取文档块元数据
        4. 生成唯一ID并添加到向量数据库
        5. 打印添加进度
        """
        for chunk in tqdm(chunks, desc="添加文档块", unit="块"):
            # 生成唯一ID
            doc_id = str(uuid.uuid4())

            # 构建metadata
            metadata = {
                "filename": chunk.get("filename", "unknown"),
                "filepath": chunk.get("filepath", ""),
                "filetype": chunk.get("filetype", ""),
                "page_number": chunk.get("page_number", 0),
                "chunk_id": chunk.get("chunk_id", 0),
                "has_images": len(chunk.get("images", [])) > 0,  # 布尔值表示是否包含图片
            }

            # 获取embedding（使用我们自定义的Embedding模型）
            embedding = self.get_embedding(chunk["content"])

            # 添加到向量数据库，显式传入embedding，保证维度一致
            self.collection.add(
                documents=[chunk["content"]],
                metadatas=[metadata],
                ids=[doc_id],
                embeddings=[embedding],
            )

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """搜索相关文档

        要求：
        1. 首先获取查询文本的embedding向量（调用self.get_embedding）
        2. 使用self.collection进行向量搜索, 得到top_k个结果
        3. 格式化返回结果，每个结果包含：
           - id: 文档ID
           - content: 文档内容
           - metadata: 元数据（文件名、页码等）
        4. 返回格式化的结果列表
        """

        embedding = self.get_embedding(query)
        query_results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        # ChromaDB返回的是嵌套列表，我们只需要第一个查询的结果
        ids = query_results["ids"][0]
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]

        results = []
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            results.append({
                "id": doc_id,
                "content": document,
                "metadata": metadata,
            })
        return results

    def clear_collection(self) -> None:
        """清空collection"""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name, metadata={"description": "课程向量数据库"}
        )
        print("向量数据库已清空")

    def get_collection_count(self) -> int:
        """获取collection中的文档数量"""
        return self.collection.count()
