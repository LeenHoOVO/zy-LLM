import os
import sys
import json
import re
from typing import List, Dict
from pathlib import Path
import logging
from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class ZhouYiDataProcessor:
    def __init__(self, data_dir: str = "data", api_key: str = None, base_url: str = "https://www.gptapi.us/v1"):
        """
        初始化数据处理器
        Args:
            data_dir: 数据目录的路径
            api_key: OpenAI API 密钥
            base_url: OpenAI API 基础地址
        """
        self.data_dir = Path(data_dir)
        self.pdf_processor = PDFProcessor(api_key=api_key, base_url=base_url)
        self.processed_data = []
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'by_type': {
                'pdf': {'total': 0, 'success': 0, 'failed': 0},
                'txt': {'total': 0, 'success': 0, 'failed': 0}
            }
        }
        
    def read_pdf(self, file_path: str) -> str:
        """读取PDF文件"""
        try:
            return self.pdf_processor.process_pdf(file_path)
        except Exception as e:
            logger.error(f"PDF读取失败 {file_path}: {str(e)}")
            return ""

    def read_txt(self, file_path: str) -> str:
        """读取TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"TXT读取失败 {file_path}: {str(e)}")
                return ""
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
            return ""

    def read_file(self, file_path: Path) -> str:
        """根据文件类型选择适当的读取方法"""
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return self.read_pdf(str(file_path))
        elif suffix in ['.txt', '.md']:
            return self.read_txt(str(file_path))
        else:
            logger.warning(f"不支持的文件格式: {file_path}")
            return ""

    def split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        将文本分割成小块，保持文本完整性
        Args:
            text: 输入文本
            chunk_size: 每块的最大字符数
        """
        # 清理文本，保留有意义的换行
        text = re.sub(r' +', ' ', text)  # 合并多个空格
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # 统一段落分隔符
        
        # 定义重要的分隔标记
        chapter_pattern = r'(【.*?】|《.*?》)'  # 章节标记
        hexagram_pattern = r'([䷀-䷿]|[☰-☷])'  # 卦象标记
        
        # 1. 首先按章节分割
        chapter_splits = re.split(f'(?=({chapter_pattern}))', text)
        chunks = []
        current_chunk = ""
        
        for split in chapter_splits:
            if not split.strip():
                continue
                
            # 检查是否是新章节的开始
            is_new_chapter = bool(re.match(chapter_pattern, split.strip()))
            
            # 如果当前内容加上新内容超过大小限制，且不是章节标题
            if len(current_chunk) + len(split) > chunk_size and not is_new_chapter:
                # 进一步按段落分割
                paragraphs = current_chunk.split('\n\n')
                temp_chunk = ""
                
                for para in paragraphs:
                    if len(temp_chunk) + len(para) <= chunk_size:
                        temp_chunk += (para + '\n\n')
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = para + '\n\n'
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                current_chunk = split
                
            else:
                # 如果是章节标题或者内容还不够大，继续累积
                current_chunk += split
                
            # 检查是否包含完整的卦象和注释
            if current_chunk:
                # 确保卦象和注释的完整性
                if (current_chunk.count('【') == current_chunk.count('】') and 
                    current_chunk.count('《') == current_chunk.count('》') and
                    not re.search(hexagram_pattern + r'.*$', current_chunk)):
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        # 处理最后剩余的内容
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def process_file(self, file_path: Path) -> List[Dict]:
        """
        处理单个文件
        Args:
            file_path: 文件路径
        """
        file_type = file_path.suffix.lower()[1:]  # 去掉点号
        self.stats['total_files'] += 1
        self.stats['by_type'].setdefault(file_type, {'total': 0, 'success': 0, 'failed': 0})
        self.stats['by_type'][file_type]['total'] += 1

        logger.info(f"开始处理文件: {file_path}")
        logger.info(f"文件类型: {file_type}")

        try:
            text = self.read_file(file_path)
            if not text.strip():
                logger.warning(f"文件内容为空: {file_path}")
                self.stats['by_type'][file_type]['failed'] += 1
                self.stats['failed_files'] += 1
                return []

            chunks = self.split_text_into_chunks(text)
            logger.info(f"文件 {file_path.name} 已分割为 {len(chunks)} 个块")
            self.stats['total_chunks'] += len(chunks)
            
            # 从文件路径提取分类信息
            relative_path = file_path.relative_to(self.data_dir)
            category = relative_path.parts[0] if len(relative_path.parts) > 1 else "未分类"
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    "id": f"{category}_{file_path.stem}_{i}",
                    "category": category,
                    "file_name": file_path.name,
                    "content": chunk,
                    "source": str(file_path),
                    "chunk_index": i,
                    "processing_info": {
                        "file_type": file_type,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    }
                }
                processed_chunks.append(chunk_info)

            self.stats['by_type'][file_type]['success'] += 1
            self.stats['processed_files'] += 1
            return processed_chunks

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
            self.stats['by_type'][file_type]['failed'] += 1
            self.stats['failed_files'] += 1
            return []

    def process_all_files(self) -> List[Dict]:
        """处理data目录下的所有文件"""
        all_documents = []
        
        # 遍历所有文件
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.md']:
                print(f"正在处理文件: {file_path}")
                documents = self.process_file(file_path)
                all_documents.extend(documents)
                
        return all_documents

    def save_processed_data(self, output_file: str = "processed_data.json"):
        """保存处理后的数据"""
        processed_data = self.process_all_files()
        output_path = self.data_dir / "processed" / output_file
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存处理统计信息
        stats_path = output_path.parent / "processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 打印详细的处理统计信息
        logger.info("\n=== 处理统计信息 ===")
        logger.info(f"总文件数: {self.stats['total_files']}")
        logger.info(f"成功处理文件数: {self.stats['processed_files']}")
        logger.info(f"处理失败文件数: {self.stats['failed_files']}")
        logger.info(f"总文本块数: {self.stats['total_chunks']}")
        logger.info("\n按文件类型统计:")
        for file_type, stats in self.stats['by_type'].items():
            logger.info(f"{file_type.upper()}:")
            logger.info(f"  - 总数: {stats['total']}")
            logger.info(f"  - 成功: {stats['success']}")
            logger.info(f"  - 失败: {stats['failed']}")
            if stats['total'] > 0:
                success_rate = (stats['success'] / stats['total']) * 100
                logger.info(f"  - 成功率: {success_rate:.2f}%")
        
        # 保存处理后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"\n处理完成！")
        logger.info(f"处理后的数据已保存到: {output_path}")
        logger.info(f"处理统计信息已保存到: {stats_path}")
        
        return processed_data 