from typing import List, Dict, Optional, Tuple

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    MODEL_NAME,
    TOP_K,
)
from vector_store import VectorStore


class RAGAgent:
    def __init__(
        self,
        model: str = MODEL_NAME,
    ):
        self.model = model

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        self.vector_store = VectorStore()

        # 系统提示词：限定角色、语气和回答边界
        self.system_prompt = """
你现在扮演一名**认真、负责的大学课程助教**，课程主题是：自然语言处理 / 统计语言模型 / 预训练语言模型 / 大语言模型等相关内容。

回答要求：
1. **语言**：始终使用简体中文回答。
2. **身份**：以“助教”的身份和学生对话，语气友好、有耐心，但不要过度口语化。
3. **依据材料回答**：
   - 优先使用系统提供的“课程材料片段（context）”来回答问题。
   - 如果材料中有明确结论，先按材料解释，再做适当补充和举例。
   - 如果材料没有涉及相关内容，要明确说出“材料中没有直接说明”，然后给出一般性的、标注为“额外背景知识”的解释。
4. **不要编造来源**：
   - 只能使用检索到的材料内容，不要虚构书名、论文、作者或页码。
   - 如果不确定，请直接说不确定，并建议学生回看课件或询问老师。
5. **解释风格**：
   - 先给出**直观解释**，再给出**稍微正式一点的技术表述**。
   - 避免一次性给出大段公式推导，有需要时可以先说明核心结论，再补充细节。
6. **与考试/作业相关的问题**：
   - 可以帮助学生理解概念、方法和解题思路。
   - 不要直接给出完整作业答案，如果学生明显在抄作业，请引导其自己思考。

当你生成回答时，请尽量：
1. 结构清晰，可以使用小标题或序号（1、2、3）分点说明。
2. 在合适的地方提醒学生“这是根据第X页课件/某节内容总结的”（如果在context中能看出文件名和页码）。
3. 回答结束时，可以简短给出**进一步复习建议**，例如推荐复习哪一讲的内容。
""".strip()

    def retrieve_context(
        self, query: str, top_k: int = TOP_K
    ) -> Tuple[str, List[Dict]]:
        """检索相关上下文
        """
        # 1. 使用向量数据库检索相关文档
        results = self.vector_store.search(query, top_k=top_k)

        # 2. 构建上下文字符串，并在每个片段中加入来源信息
        context_blocks = []
        for i, item in enumerate(results, start=1):
            content = item.get("content", "")
            metadata = item.get("metadata", {}) or {}

            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number", 0)
            filetype = metadata.get("filetype", "")

            source_tag = f"{filename}"
            if page_number:
                source_tag += f" - 第 {page_number} 页/页码"
            if filetype:
                source_tag += f" ({filetype})"

            block = (
                f"【检索结果 {i}】\n"
                f"来源：{source_tag}\n"
                f"内容：\n{content}\n"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks).strip()
        return context_text, results

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """生成回答
        
        参数:
            query: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        # 构造用户提示词：包含检索上下文、学生问题和对回答方式的提醒
        user_text = f"""
下面是与学生问题相关的课程材料片段（可能来自不同的课件/页码）：

{context}

请根据上述“课程材料片段”为学生回答下面的问题。如果材料中没有直接涉及，也请明确说明。

学生问题：{query}

回答时请：
1. 优先引用和解释上面的课程材料内容；
2. 在合适的位置提到相关片段的来源（文件名、页码等，如果在材料中能看出来的话）；
3. 用适合本科生的方式讲解，可以适当类比或举简单例子；
4. 如果需要补充课外背景知识，请标注为“【额外背景】”。
""".strip()

        messages.append({"role": "user", "content": user_text})
        
        # 多模态接口示意（如需添加图片支持，可参考以下格式）：
        # content_parts = [{"type": "text", "text": user_text}]
        # content_parts.append({
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        # })
        # messages.append({"role": "user", "content": content_parts})

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, max_tokens=1500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def answer_question(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> Dict[str, any]:
        """回答问题
        
        参数:
            query: 用户问题
            chat_history: 对话历史
            top_k: 检索文档数量
            
        返回:
            生成的回答
        """
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)

        if not context:
            context = "（未检索到特别相关的课程材料）"

        answer = self.generate_response(query, context, chat_history)

        return answer

    def chat(self) -> None:
        """交互式对话"""
        print("=" * 60)
        print("欢迎使用智能课程助教系统！")
        print("=" * 60)

        chat_history = []

        while True:
            try:
                query = input("\n学生: ").strip()

                if not query:
                    continue

                answer = self.answer_question(query, chat_history=chat_history)

                print(f"\n助教: {answer}")

                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                print(f"\n错误: {str(e)}")
