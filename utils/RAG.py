import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple
from transformers import AutoModel, AutoTokenizer

class RAGRetriever:
    def __init__(self, processed_dir: str):
        """
        初始化检索器
        
        Args:
            processed_dir: 处理后的数据目录
        """
        self.processed_dir = processed_dir
        self.chunks_dir = os.path.join(processed_dir, "chunks")
        self.embeddings_dir = os.path.join(processed_dir, "embeddings")
        
        # 加载embedding模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = AutoModel.from_pretrained("BAAI/bge-large-zh").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")
        self.embedding_model.eval()
        
        # 加载所有文档的映射和向量
        self.load_all_documents()
        
    def load_all_documents(self):
        """加载所有文档的映射和向量"""
        self.all_chunks = []  # 存储所有文本块
        self.all_embeddings = []  # 存储所有向量
        self.chunk_sources = []  # 存储每个chunk的来源文件
        
        # 遍历embeddings目录下的所有mapping文件
        for file_name in os.listdir(self.embeddings_dir):
            if file_name.endswith('_mapping.json'):
                mapping_path = os.path.join(self.embeddings_dir, file_name)
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                
                # 加载文本块
                chunks = mapping['chunks']
                
                # 加载对应的向量
                embeddings_file = mapping['embeddings_file']
                embeddings_path = os.path.join(self.embeddings_dir, embeddings_file)
                embeddings = np.load(embeddings_path)
                
                # 添加到总列表中
                self.all_chunks.extend(chunks)
                self.all_embeddings.extend(embeddings)
                self.chunk_sources.extend([file_name.replace('_mapping.json', '')] * len(chunks))
        
        # 将所有向量转换为numpy数组
        self.all_embeddings = np.array(self.all_embeddings)
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """获取查询文本的向量"""
        with torch.no_grad():
            # 添加特殊前缀
            query = f"为这段内容生成向量：{query}"
            
            # 编码文本
            encoded = self.tokenizer(
                query,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 获取向量
            output = self.embedding_model(**encoded)
            embedding = output.last_hidden_state[:, 0].cpu().numpy()
            
            return embedding[0]
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """
        检索相关文本
        
        Args:
            query: 查询文本
            top_k: 返回的相关文本数量
            
        Returns:
            包含相关文本和来源信息的列表
        """
        # 获取查询向量
        query_embedding = self.get_query_embedding(query)
        
        # 计算相似度
        similarities = np.dot(self.all_embeddings, query_embedding)
        
        # 获取top_k个最相似的索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 构建返回结果
        results = []
        for idx in top_indices:
            results.append({
                'text': self.all_chunks[idx],
                'source': self.chunk_sources[idx],
                'similarity': float(similarities[idx])
            })
            
        return results

def format_context(retrieved_docs: List[Dict[str, str]]) -> str:
    """格式化检索到的上下文"""
    context = "根据以下参考资料：\n\n"
    for i, doc in enumerate(retrieved_docs, 1):
        context += f"[{i}] 来自《{doc['source']}》:\n{doc['text']}\n\n"
    return context

def generate_prompt(query: str, retrieved_docs: List[Dict[str, str]]) -> str:
    """生成完整的提示词"""
    context = format_context(retrieved_docs)
    prompt = f"{context}\n问题：{query}\n\n请根据上述参考资料回答问题，如果参考资料中没有相关信息，请明确指出。回答时注明使用了哪些参考资料。"
    return prompt
