import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
import pickle
import torch
from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

class VectorStore:
    def __init__(self, model_name: str = "Jerry0/text2vec-base-chinese"):
        """
        初始化向量存储
        Args:
            model_name: 使用的文本向量模型名称
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
            print(f"模型加载失败: {str(e)}")
            raise
            
        self.index = None
        self.documents = []
        
    def encode_text(self, text: str) -> np.ndarray:
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
            print(f"文本嵌入计算失败: {str(e)}")
            raise

    def create_index(self, documents: List[Dict]):
        """
        为文档创建向量索引
        Args:
            documents: 文档列表
        """
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        
        # 计算文本向量
        print("正在计算文本向量...")
        embeddings = []
        for text in texts:
            embedding = self.encode_text(text)
            # 归一化向量
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        self.index.add(embeddings.astype('float32'))
        
        print(f"索引创建完成，共包含 {len(documents)} 条文档")
        
    def load_processed_data(self, json_path: str) -> List[Dict]:
        """加载处理好的文档数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 从 graph_rag 的数据格式转换为向量存储需要的格式
                if isinstance(data, dict) and 'documents' in data:
                    documents = data['documents']
                    # 确保每个文档都有必要的字段
                    processed_docs = []
                    for doc in documents:
                        if isinstance(doc, dict) and 'content' in doc:
                            processed_docs.append({
                                'content': doc['content'],
                                'source': doc.get('source', '未知'),
                                'category': doc.get('category', '未分类')
                            })
                    return processed_docs
                else:
                    print("数据格式不正确，需要包含 'documents' 字段")
                    return []
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return []
            
    def save(self, save_dir: str):
        """
        保存向量索引和文档
        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))
        
        # 保存文档
        with open(save_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
            
        print(f"向量存储已保存到: {save_dir}")
        
    @classmethod
    def load(cls, save_dir: str, model_name: str = "Jerry0/text2vec-base-chinese"):
        """
        加载已保存的向量存储
        Args:
            save_dir: 保存目录
            model_name: 模型名称
        """
        save_dir = Path(save_dir)
        instance = cls(model_name)
        
        # 加载索引
        instance.index = faiss.read_index(str(save_dir / "faiss_index.bin"))
        
        # 加载文档
        with open(save_dir / "documents.pkl", 'rb') as f:
            instance.documents = pickle.load(f)
            
        return instance
        
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相似文档
        Args:
            query: 查询文本
            top_k: 返回最相似的文档数量
        """
        # 计算查询文本的向量
        query_vector = self.encode_text(query)
        # 归一化查询向量
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 搜索最相似的文档
        scores, indices = self.index.search(query_vector.reshape(1, -1).astype('float32'), top_k)
        
        # 返回结果
        results = []
        for i, idx in enumerate(indices[0]):
            doc = self.documents[idx]
            results.append({
                'content': doc['content'],
                'source': doc['source'],
                'category': doc['category'],
                'similarity': float(scores[0][i])  # 直接使用内积分数作为相似度
            })
            
        return results

def main():
    """主函数：创建和保存向量存储"""
    try:
        # 创建向量存储
        vector_store = VectorStore()
        
        # 加载处理好的数据
        print("正在加载数据...")
        documents = vector_store.load_processed_data(r"G:\code\zy-LLM\data\graph_rag\data.json")
        if not documents:
            print("没有找到有效的文档数据")
            return
            
        print(f"成功加载 {len(documents)} 个文档")
        
        # 创建索引
        vector_store.create_index(documents)
        
        # 保存向量存储
        vector_store.save("data/vector_store")
        
        # 测试搜索
        print("\n测试搜索示例：")
        queries = [
            "周易八卦的含义是什么？",
            "乾卦的含义",
            "六爻的解释"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            results = vector_store.search(query, top_k=3)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 相似度: {result['similarity']:.4f}")
                print(f"来源: {result['source']}")
                print(f"内容: {result['content'][:200]}...")
            
    except Exception as e:
        print(f"程序运行失败: {str(e)}")

if __name__ == "__main__":
    main() 