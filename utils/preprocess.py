import os
import re
import jieba
import pandas as pd
from typing import List, Dict
from pypdf import PdfReader
import torch
from tqdm import tqdm
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer

class DocumentProcessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        """
        初始化文档处理器
        
        Args:
            raw_dir: 原始文档目录
            processed_dir: 处理后文档保存目录
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunks_dir = os.path.join(processed_dir, "chunks")
        self.embeddings_dir = os.path.join(processed_dir, "embeddings")
        
        # 创建必要的目录
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # 初始化embedding模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = AutoModel.from_pretrained("BAAI/bge-large-zh").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")
        self.embedding_model.eval()
        
    def read_txt(self, file_path: str) -> str:
        """读取txt文件，尝试不同的编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
                
        raise UnicodeDecodeError(f"无法用常见编码（{', '.join(encodings)}）读取文件: {file_path}")
            
    def read_pdf(self, file_path: str) -> str:
        """读取pdf文件"""
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    # 提取页面文本
                    page_text = page.extract_text()
                    if page_text.strip():  # 如果页面有内容
                        # 添加页码标记
                        text_parts.append(f"[第{page_num}页]\n{page_text}")
                except Exception as e:
                    print(f"警告: 处理PDF文件 {file_path} 第{page_num}页时出错: {str(e)}")
                    continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"错误: 无法读取PDF文件 {file_path}: {str(e)}")
            return ""
        
    def clean_text(self, text: str) -> str:
        """
        清理文本
        - 移除多余空白
        - 统一标点符号
        - 基础文本规范化
        """
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = re.sub(r'["""]', '"', text)  # 统一引号
        
        text = text.strip()
        return text
        
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        将文本分割成小段
        
        Args:
            text: 输入文本
            chunk_size: 每段的大约字符数
        """
        # 按页分割
        pages = text.split('[第')
        chunks = []
        
        for page in pages:
            if not page.strip():
                continue
            
            # 保留页码信息
            if '页]' in page:
                page_info = '[第' + page.split('页]')[0] + '页]\n'
                content = '页]'.join(page.split('页]')[1:])
            else:
                page_info = ''
                content = page
            
            # 按句子分割内容
            sentences = re.split('([。！？])', content)
            current_chunk = page_info if page_info else ''
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = page_info + sentence if page_info else sentence
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
        
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """获取文本向量"""
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="生成向量"):
                # 添加特殊前缀以提高检索效果
                text = f"为这段内容生成向量：{text}"
                
                # 编码文本
                encoded = self.tokenizer(
                    text,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.embedding_model.device)
                
                # 获取向量
                output = self.embedding_model(**encoded)
                embedding = output.last_hidden_state[:, 0].cpu().numpy()  # 使用[CLS]标记的输出
                embeddings.append(embedding[0])
                
        return embeddings
        
    def process_documents(self):
        """处理所有文档"""
        if os.path.exists(self.raw_dir):
            for file_name in tqdm(os.listdir(self.raw_dir), desc="处理文件"):
                file_path = os.path.join(self.raw_dir, file_name)
                if file_name.endswith('.txt'):
                    text = self.read_txt(file_path)
                elif file_name.endswith('.pdf'):
                    text = self.read_pdf(file_path)
                else:
                    continue
                    
                text = self.clean_text(text)
                chunks = self.split_text_into_chunks(text)
                
                # 保存分段结果
                chunks_path = os.path.join(self.chunks_dir, f"{file_name}_chunks.txt")
                with open(chunks_path, 'w', encoding='utf-8') as f:
                    for chunk in chunks:
                        f.write(chunk + "\n\n")
                
                # 生成并保存向量
                embeddings = self.get_embeddings(chunks)
                embeddings_path = os.path.join(self.embeddings_dir, f"{file_name}_embeddings.npy")
                np.save(embeddings_path, embeddings)
                
                # 保存chunks和embeddings的映射关系
                mapping_path = os.path.join(self.embeddings_dir, f"{file_name}_mapping.json")
                mapping = {
                    "chunks": chunks,
                    "embeddings_file": f"{file_name}_embeddings.npy"
                }
                with open(mapping_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2) 