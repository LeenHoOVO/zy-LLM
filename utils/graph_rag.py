import os
import torch
import json
import logging
from typing import List, Dict, Set, Tuple
import numpy as np
from pathlib import Path
import networkx as nx
import re
import pickle
from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, model_name: str = "Jerry0/text2vec-base-chinese"):
        """
        初始化GraphRAG
        Args:
            model_name: 文本向量模型名称
        """
        try:
            # 使用modelscope下载模型
            model_dir = snapshot_download(model_name)
            
            # 使用transformers加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir)
            
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
            self.model.eval()  # 设置为评估模式
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
        
        self.documents = []
        self.graph = nx.Graph()
        self.key_concepts = set()
        self.small_chunks = {}
        self.parent_chunks = {}
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        """使用transformers获取文本嵌入"""
        try:
            # 对文本进行编码
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # 获取文本嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的输出作为文本表示
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]  # 返回第一个样本的嵌入向量
            
        except Exception as e:
            logger.error(f"文本嵌入计算失败: {str(e)}")
            raise

    def build_knowledge_graph(self, documents: List[Dict]):
        """构建分层知识图谱"""
        try:
            self.documents = documents
            
            # 1. 创建小块文本
            for doc in documents:
                small_chunks = self._create_small_chunks(doc['content'])
                for chunk_id, chunk in small_chunks.items():
                    self.small_chunks[chunk_id] = chunk
                    # 添加小块节点
                    self.graph.add_node(chunk_id, 
                                    type='small_chunk',
                                    content=chunk['content'])
            
            logger.info(f"创建了 {len(self.small_chunks)} 个小块")
            
            # 2. 创建父块文本（更大的上下文）
            for doc in documents:
                parent_chunks = self._create_parent_chunks(doc['content'])
                for chunk_id, chunk in parent_chunks.items():
                    self.parent_chunks[chunk_id] = chunk
                    # 添加父块节点
                    self.graph.add_node(chunk_id, 
                                    type='parent_chunk',
                                    content=chunk['content'])
            
            logger.info(f"创建了 {len(self.parent_chunks)} 个父块")
            
            # 3. 建立小块到父块的映射
            self._build_chunk_relations()
            
            # 4. 提取和构建概念关系
            self._extract_key_concepts(documents)
            
            # 5. 添加概念节点
            for concept in self.key_concepts:
                self.graph.add_node(concept, type='concept')
            
            # 6. 建立概念关系
            self._build_concept_relations(documents)
            
            # 7. 创建向量索引
            logger.info("开始构建向量索引...")
            small_chunks_text = [chunk['content'] for chunk in self.small_chunks.values()]
            if small_chunks_text:
                # 批量处理以提高效率
                batch_size = 32
                small_vectors = []
                
                for i in range(0, len(small_chunks_text), batch_size):
                    batch_texts = small_chunks_text[i:i + batch_size]
                    batch_vectors = [self.get_text_embedding(text) for text in batch_texts]
                    small_vectors.extend(batch_vectors)
                
                small_vectors = np.array(small_vectors)
                self.vectors = small_vectors
                
            logger.info(f"知识图谱构建完成: {len(self.graph.nodes)} 个节点")
            
        except Exception as e:
            logger.error(f"构建知识图谱失败: {str(e)}")
            raise
        
    def _create_small_chunks(self, text: str, size: int = 200) -> Dict[str, Dict]:
        """创建小块文本（句子级别）"""
        chunks = {}
        sentences = re.split(r'([。！？])', text)
        current_chunk = ""
        chunk_id = 0
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 加回标点
                
            if len(current_chunk) + len(sentence) <= size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks[f"small_{chunk_id}"] = {
                        'content': current_chunk,
                        'start_pos': text.find(current_chunk),
                        'end_pos': text.find(current_chunk) + len(current_chunk)
                    }
                    chunk_id += 1
                current_chunk = sentence
                
        if current_chunk:
            chunks[f"small_{chunk_id}"] = {
                'content': current_chunk,
                'start_pos': text.find(current_chunk),
                'end_pos': text.find(current_chunk) + len(current_chunk)
            }
            
        return chunks
        
    def _create_parent_chunks(self, text: str, size: int = 1000) -> Dict[str, Dict]:
        """创建父块文本（段落级别）"""
        chunks = {}
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= size:
                current_chunk += (para + '\n\n')
            else:
                if current_chunk:
                    chunks[f"parent_{chunk_id}"] = {
                        'content': current_chunk,
                        'start_pos': text.find(current_chunk),
                        'end_pos': text.find(current_chunk) + len(current_chunk)
                    }
                    chunk_id += 1
                current_chunk = para + '\n\n'
                
        if current_chunk:
            chunks[f"parent_{chunk_id}"] = {
                'content': current_chunk,
                'start_pos': text.find(current_chunk),
                'end_pos': text.find(current_chunk) + len(current_chunk)
            }
            
        return chunks
        
    def _build_chunk_relations(self):
        """建立小块到父块的映射关系"""
        for small_id, small_chunk in self.small_chunks.items():
            # 找到包含这个小块的父块
            for parent_id, parent_chunk in self.parent_chunks.items():
                if (small_chunk['start_pos'] >= parent_chunk['start_pos'] and 
                    small_chunk['end_pos'] <= parent_chunk['end_pos']):
                    self.graph.add_edge(small_id, parent_id, type='parent')
                    
    def _extract_key_concepts(self, documents: List[Dict]):
        """提取关键概念"""
        # 扩展周易特定概念
        concept_patterns = [
            r'[☰☱☲☳☴☵☶☷]',  # 八卦符号
            r'[䷀-䷿]',  # 六十四卦符号
            r'(乾|坤|震|巽|坎|离|艮|兑)卦',  # 八卦名称
            r'[初二三四五上]爻',  # 爻位
            r'([六九])(爻|阳|阴)',  # 爻性质
            r'(彖|象|文言)传',  # 传名
            r'周易|易经',  # 典籍名称
            r'卦辞|爻辞',  # 辞类
            r'大象|小象',  # 象类
            r'(乾|坤|震|巽|坎|离|艮|兑)(为|曰|象)',  # 卦象解释
            r'(君子|小人)(以|则)',  # 取象
        ]
        
        for doc in documents:
            content = doc['content']
            # 提取所有匹配的概念
            for pattern in concept_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    self.key_concepts.add(match.group())
    
    def _build_concept_relations(self, documents: List[Dict] = None):
        """建立文档和概念之间的关系"""
        # 遍历所有块（包括小块和父块）
        for chunk_id, chunk in {**self.small_chunks, **self.parent_chunks}.items():
            chunk_concepts = set()
            content = chunk['content']
            
            # 查找块中包含的概念
            for concept in self.key_concepts:
                if concept in content:
                    chunk_concepts.add(concept)
                    self.graph.add_edge(chunk_id, concept, type='contains')
            
            # 建立概念之间的关系
            for c1 in chunk_concepts:
                for c2 in chunk_concepts:
                    if c1 < c2:  # 避免重复边
                        self.graph.add_edge(c1, c2, 
                                        type='co-occurrence',
                                        chunk=chunk_id)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """实现small2big检索策略"""
        try:
            # 1. 获取查询的向量表示
            query_vector = self.get_text_embedding(query)
            
            # 2. 计算与所有小块的相似度
            similarities = []
            for i, (chunk_id, chunk) in enumerate(self.small_chunks.items()):
                chunk_vector = self.vectors[i]
                similarity = np.dot(query_vector, chunk_vector)
                similarities.append((chunk_id, similarity))
            
            # 3. 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 4. 获取top_k个小块及其父块
            results = []
            seen_chunks = set()
            
            for chunk_id, similarity in similarities[:top_k]:
                if chunk_id not in seen_chunks:
                    # 添加小块
                    chunk = self.small_chunks[chunk_id]
                    results.append({
                        'content': chunk['content'],
                        'similarity': similarity,
                        'type': 'small_chunk',
                        'chunk_id': chunk_id,
                        'related_concepts': [n for n in self.graph.neighbors(chunk_id) 
                                          if self.graph.nodes[n]['type'] == 'concept']
                    })
                    seen_chunks.add(chunk_id)
                    
                    # 添加父块
                    for parent_id in self.graph.neighbors(chunk_id):
                        if (parent_id.startswith('parent_') and 
                            parent_id not in seen_chunks):
                            parent_chunk = self.parent_chunks[parent_id]
                            parent_similarity = similarity * 0.8  # 降低父块的相似度权重
                            results.append({
                                'content': parent_chunk['content'],
                                'similarity': parent_similarity,
                                'type': 'parent_chunk',
                                'chunk_id': parent_id,
                                'related_concepts': [n for n in self.graph.neighbors(parent_id) 
                                                  if self.graph.nodes[n]['type'] == 'concept']
                            })
                            seen_chunks.add(parent_id)
            
            # 5. 按相似度重新排序
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []

    def save(self, save_dir: str):
        """保存GraphRAG数据"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图结构和chunks数据
        with open(save_dir / "knowledge_graph.pkl", 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'small_chunks': self.small_chunks,
                'parent_chunks': self.parent_chunks,
                'key_concepts': self.key_concepts,
                'vectors': self.vectors
            }, f)
        
        # 保存文档数据
        with open(save_dir / "data.json", 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'concepts': list(self.key_concepts)
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"GraphRAG数据已保存到: {save_dir}")

    @classmethod
    def load(cls, save_dir: str):
        """加载保存的GraphRAG数据"""
        save_dir = Path(save_dir)
        instance = cls()
        
        try:
            # 加载图结构和chunks数据
            with open(save_dir / "knowledge_graph.pkl", 'rb') as f:
                data = pickle.load(f)
                instance.graph = data['graph']
                instance.small_chunks = data['small_chunks']
                instance.parent_chunks = data['parent_chunks']
                instance.key_concepts = data['key_concepts']
                instance.vectors = data['vectors']
            
            # 加载文档数据
            with open(save_dir / "data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                instance.documents = data['documents']
            
            logger.info(f"GraphRAG加载成功，包含 {len(instance.small_chunks)} 个小块和 {len(instance.parent_chunks)} 个父块")
            return instance
            
        except Exception as e:
            logger.error(f"加载GraphRAG失败: {str(e)}")
            raise

def main():
    """主函数：构建和测试GraphRAG"""
    # 加载处理好的数据
    with open("data/processed/processed_data.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # 创建GraphRAG实例
    graph_rag = GraphRAG()
    
    # 构建知识图谱
    graph_rag.build_knowledge_graph(documents)
    
    # 保存数据
    graph_rag.save("data/graph_rag")
    
    # 测试搜索
    query = "解释乾卦九三爻的含义"
    results = graph_rag.search(query)
    
    print(f"\n查询: {query}")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"块类型: {result['type']}")
        print(f"块ID: {result['chunk_id']}")
        print(f"相关概念: {', '.join(result['related_concepts'])}")
        print(f"内容: {result['content'][:200]}...")

if __name__ == "__main__":
    main() 