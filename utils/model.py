from openai import OpenAI
from utils.RAG import basic_rag
from utils.vector_store import VectorStore
from utils.graph_rag import GraphRAG
import os
import logging
from typing import List, Dict, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """你是一个名叫"周易智能助手"的 AI 助手。

基本原则：
1. 对话风格：
   - 保持友好、专业且自然的对话风格
   - 根据用户的语气和问题类型调整回应方式
   - 避免过于生硬或过于随意的表达

2. 知识运用：
   - 优先以通用知识和常识回答日常问题
   - 在涉及专业领域时，结合易经等传统智慧提供建议
   - 确保建议实用且符合现代生活情境

3. 回答策略：
   - 先理解用户核心诉求
   - 给出清晰、有条理的回应
   - 必要时提供具体的行动建议
   - 保持开放态度，引导用户思考

4. 专业解读：
   - 结合检索到的权威资料
   - 将深奥概念转化为易懂表达
   - 在必要时结合时空因素分析
   - 注重实用性和可操作性

5. 安全边界：
   - 不提供医疗、法律等专业建议
   - 对敏感话题保持中立
   - 遇到不确定内容主动说明"""

USER_PROMPT_TEMPLATE = """
背景信息：
- 当前时间：{current_time}
- 用户问题：{question}

请基于以上信息，结合相关知识，给出专业、实用且易于理解的回应。"""

class AIModel:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, temperature: float = 0.7, max_tokens: int = 2000):
        """
        初始化AI模型
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
            temperature: 温度参数
            max_tokens: 最大token数
        """
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API密钥必须通过参数提供或设置OPENAI_API_KEY环境变量")
                
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化本地RAG系统
        try:
            self.graph_rag = GraphRAG.load("data/graph_rag")
            logger.info("GraphRAG 加载成功")
        except Exception as e:
            logger.warning(f"GraphRAG 加载失败: {str(e)}")
            self.graph_rag = None
        
        try:
            self.vector_store = VectorStore.load("data/vector_store")
            logger.info("向量存储加载成功")
        except Exception as e:
            logger.warning(f"向量存储加载失败: {str(e)}")
            self.vector_store = None

    def get_context(self, query: str) -> str:
        """获取本地知识库的相关上下文"""
        contexts = []
        
        # 1. 尝试使用GraphRAG检索
        if self.graph_rag:
            try:
                results = self.graph_rag.search(query, top_k=3)
                if results:
                    contexts.extend([r['content'] for r in results])
            except Exception as e:
                logger.warning(f"GraphRAG检索失败: {str(e)}")
        
        # 2. 尝试使用向量存储检索
        if self.vector_store and not contexts:
            try:
                results = self.vector_store.search(query, top_k=3)
                if results:
                    contexts.extend([r['content'] for r in results])
            except Exception as e:
                logger.warning(f"向量存储检索失败: {str(e)}")
        
        # 3. 如果都失败了，尝试使用基础RAG
        if not contexts:
            try:
                context, _ = basic_rag(query)
                if context:
                    contexts.append(context)
            except Exception as e:
                logger.warning(f"基础RAG检索失败: {str(e)}")
        
        return "\n\n".join(contexts) if contexts else ""

    def generate_with_rag(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Union[str, List]]:
        """生成回复"""
        try:
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            context = self.get_context(query)
            
            # 优化系统提示词
            system_prompt = SYSTEM_PROMPT + """

作为周易智能助手，请注意：
1. 保持对话的连贯性和上下文关联
2. 回答要自然流畅，不要提及"参考资料"、"根据文献"等提示词
3. 将易经知识自然地融入回答中
4. 使用清晰的标题和列表格式组织回答
5. 保持专业性的同时确保易于理解"""

            # 优化用户提示词
            if context:
                user_prompt = f"""基于当前情况：
- 时间：{current_time}
- 问题：{query}

请结合易经智慧和专业知识，给出系统性的分析和建议。回答要：
1. 保持与之前对话的连贯性
2. 给出清晰的分析和具体建议
3. 使用易于理解的语言
4. 适当运用易经智慧
5. 注重实用性和可操作性"""
            else:
                user_prompt = f"""当前时间：{current_time}
问题：{query}

请给出系统性的分析和建议，注意与之前对话保持连贯。"""
            
            # 初始化历史记录
            if history is None:
                history = []
            
            # 构建消息
            messages = [
                {'role': 'system', 'content': system_prompt}
            ] + history + [
                {'role': 'user', 'content': user_prompt}
            ]
            
            # 调用API
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                if not response or not response.choices:
                    raise ValueError("API 返回的响应为空或无效")
                    
                reply = response.choices[0].message.content
                
                # 更新历史记录
                new_history = history + [
                    {'role': 'user', 'content': query},
                    {'role': 'assistant', 'content': reply}
                ]
                
                return {
                    'response': reply,
                    'history': new_history
                }
                
            except Exception as api_error:
                logger.error(f"API调用失败: {str(api_error)}")
                raise
            
        except Exception as e:
            logger.error(f"生成回复失败: {str(e)}")
            return {
                'response': f"抱歉，生成回复时出现错误: {str(e)}",
                'history': history or []
            }

    def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Union[str, List]]:
        """
        生成回复
        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示词
        Returns:
            包含response和history的字典
        """
        if not self.client:
            raise ValueError("未初始化OpenAI客户端，请确保提供了有效的API密钥")
            
        try:
            messages = []
            
            # 添加系统提示
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # 添加历史对话
            if history:
                for msg in history:
                    messages.append(msg)
                    
            # 添加当前用户输入
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # 更新历史
            if history is None:
                history = []
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": assistant_message})
            
            return {
                "response": assistant_message,
                "history": history
            }
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            raise


