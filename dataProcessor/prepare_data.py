import os
import sys
import logging
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dataProcessor.data_process import ZhouYiDataProcessor
from utils.graph_rag import GraphRAG
from utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

# 配置项
CONFIG = {
    # OpenAI 配置
    "api_key": os.getenv("OPENAI_API_KEY"),  # 从环境变量获取 API 密钥
    "base_url": "https://www.gptapi.us/v1",  # 设置 API 基础地址
    
    # PDF处理配置
    "output_dir": "output",  # 处理结果输出目录
    
    # GPT模型配置
    "model": "gpt-4-vision-preview",  # 使用的模型
    "gpt_worker": 2,  # GPT处理的工作线程数
    "verbose": True,  # 是否显示详细日志
    
    # 自定义提示词（针对易经处理优化）
    "prompt": {
        "prompt": """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。对于易经内容：
1. 保持原文的章节结构和层次
2. 正确识别和处理卦象符号（如：☰、☷、䷀等）
3. 保持注释与正文的对应关系
4. 对于图表：
   - 准确描述卦象图示
   - 保留图表中的文字说明
5. 格式要求：
   - 使用标准Markdown语法
   - 章节使用适当的标题级别（#、##等）
   - 保持段落的自然分隔
   - 对重要内容进行适当标记
""",
        "rect_prompt": """遇到特殊区域时：
1. 表格：转换为Markdown表格格式
2. 图片：添加适当的图片描述
3. 卦象图：详细描述卦象的结构和含义
""",
        "role_prompt": "你是一个专业的易经文档解析器，使用markdown和latex语法输出图片的内容。特别注意保持易经中的卦象符号和专业术语的准确性。"
    }
}

def process_single_pdf(pdf_path: str, api_key: str = None):
    """
    处理单个PDF文件
    Args:
        pdf_path: PDF文件路径
        api_key: OpenAI API 密钥
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return
            
        logger.info(f"开始处理PDF文件: {pdf_path}")
        
        # 创建处理器实例
        processor = ZhouYiDataProcessor(
            api_key=api_key or CONFIG["api_key"],
            base_url=CONFIG["base_url"]
        )
        
        # 处理PDF文件
        content = processor.read_pdf(pdf_path)
        if not content:
            logger.error("PDF处理失败")
            return
            
        logger.info("PDF处理完成")
        
        # 构建知识图谱和向量存储
        processed_data = [{
            "id": Path(pdf_path).stem,
            "content": content,
            "source": pdf_path
        }]
        
        # 构建GraphRAG
        graph_rag = GraphRAG()
        graph_rag.build_knowledge_graph(processed_data)
        graph_rag.save("data/graph_rag")
        
        # 构建向量存储
        vector_store = VectorStore()
        vector_store.create_index(processed_data)
        vector_store.save("data/vector_store")
        
        logger.info("数据准备完成")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 检查环境变量中是否设置了 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("请设置 OPENAI_API_KEY 环境变量")
        exit(1)
    
    # 指定要处理的PDF文件路径
    pdf_path = "data/yi_ching.pdf"  # 这里可以改成你想处理的PDF文件路径
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    process_single_pdf(pdf_path) 