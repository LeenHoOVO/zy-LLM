import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
import json

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.graph_rag import GraphRAG
from utils.vector_store import VectorStore

logger = logging.getLogger(__name__)

def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # 如果 UTF-8 失败，尝试 GBK 编码
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {str(e)}")
        return ""

def process_files(data_dir: str = "data") -> List[Dict]:
    """处理所有支持的文件格式"""
    processed_data = []
    data_path = Path(data_dir)
    
    # 支持的文件类型
    supported_extensions = {'.txt', '.md'}
    
    # 遍历所有文件
    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() in supported_extensions:
            logger.info(f"处理文件: {file_path}")
            content = read_file(str(file_path))
            
            if content:
                processed_data.append({
                    "id": file_path.stem,
                    "content": content,
                    "source": str(file_path),
                    "file_type": file_path.suffix.lower()[1:]  # 不包含点号的文件类型
                })
                logger.info(f"成功处理: {file_path.name}")
            else:
                logger.warning(f"文件处理失败: {file_path.name}")
    
    return processed_data

def build_knowledge_base(data_dir: str = "data", output_dir: str = "data"):
    """构建知识库"""
    try:
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理所有文件
        logger.info("开始处理文件...")
        processed_data = process_files(data_dir)
        
        if not processed_data:
            logger.error("没有找到可处理的文件")
            return
            
        # 记录处理统计信息
        stats = {
            "total_files": len(processed_data),
            "by_type": {}
        }
        
        # 统计各类型文件数量
        for item in processed_data:
            file_type = item["file_type"]
            if file_type not in stats["by_type"]:
                stats["by_type"][file_type] = 0
            stats["by_type"][file_type] += 1
        
        # 构建 GraphRAG
        logger.info("开始构建 GraphRAG...")
        graph_rag = GraphRAG()
        graph_rag.build_knowledge_graph(processed_data)
        graph_rag.save(str(output_path / "graph_rag"))
        logger.info("GraphRAG 构建完成")
        
        # 构建向量存储
        logger.info("开始构建向量存储...")
        vector_store = VectorStore()
        vector_store.create_index(processed_data)
        vector_store.save(str(output_path / "vector_store"))
        logger.info("向量存储构建完成")
        
        # 保存处理后的数据和统计信息
        with open(output_path / "processed_data.json", 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        with open(output_path / "processing_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        logger.info("\n=== 处理统计信息 ===")
        logger.info(f"总文件数: {stats['total_files']}")
        logger.info("\n按文件类型统计:")
        for file_type, count in stats["by_type"].items():
            logger.info(f"{file_type.upper()}: {count} 个文件")
        
        logger.info("\n知识库构建完成！")
        
    except Exception as e:
        logger.error(f"构建知识库失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置参数
    data_dir = "data"  # 默认数据目录
    output_dir = "data"  # 默认输出目录
    
    # 从命令行参数获取目录（如果提供）
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 构建知识库
    build_knowledge_base(data_dir, output_dir) 