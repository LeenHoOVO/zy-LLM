from .graph_rag import GraphRAG
import logging
import jieba.posseg as pseg
import re
import json
from collections import Counter
import math

logger = logging.getLogger(__name__)

def extract_key_terms(query: str) -> list:
    """从查询中提取关键词"""
    # 添加停用词过滤
    stopwords = set(['的', '了', '和', '与', '及', '或', '而', '把'])
    
    key_terms = []
    words = pseg.cut(query)
    
    # 改进词性过滤
    valid_flags = {'n', 'v', 'a', 'nr', 'ns', 'nt', 'nz'}  # 扩展词性范围
    
    for word, flag in words:
        if (
            len(word) > 1 and
            word not in stopwords and
            any(flag.startswith(vf) for vf in valid_flags)
        ):
            key_terms.append(word)
    
    # 2. 识别特殊模式
    patterns = [
        (r'[乾坤震巽坎离艮兑]卦', '卦象'),  # 八卦
        (r'[初二三四五上]爻', '爻位'),      # 爻位
        (r'[六九][爻阳阴]', '爻性'),        # 爻性
        (r'易经|周易', '典籍'),            # 典籍名称
        (r'卦辞|爻辞|彖辞|象辞', '解释'),   # 解释类型
    ]
    
    for pattern, type_ in patterns:
        matches = re.finditer(pattern, query)
        for match in matches:
            term = match.group()
            if term not in key_terms:  # 避免重复
                key_terms.append(term)
    
    # 3. 添加完整的查询词
    if len(query) <= 4 and query not in key_terms:
        key_terms.append(query)
    
    logger.info(f"从查询 '{query}' 中提取的关键词: {key_terms}")
    return key_terms

def keyword_search(graph_rag, key_terms: list, top_k: int = 5) -> list:
    """基于关键词的检索"""
    # 添加TF-IDF加权
    def calculate_tfidf(term, content, all_chunks):
        tf = content.count(term)
        df = sum(1 for chunk in all_chunks.values() if term in chunk['content'])
        idf = math.log(len(all_chunks) / (df + 1))
        return tf * idf
    
    chunk_scores = []
    all_chunks = {**graph_rag.small_chunks, **graph_rag.parent_chunks}
    
    for chunk_id, chunk in all_chunks.items():
        content = chunk['content']
        # 使用TF-IDF计算分数
        score = sum(calculate_tfidf(term, content, all_chunks) for term in key_terms)
        if score > 0:
            chunk_scores.append((chunk_id, score))
    
    # 按分数排序
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    return chunk_scores[:top_k]

def basic_rag(query: str, model=None) -> str:
    """基本的RAG查询"""
    try:
        # 加载处理后的数据
        with open("data/processed/processed_data.json", 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
            
        # 提取关键词
        key_terms = extract_key_terms(query)
        logger.info(f"提取的关键词: {key_terms}")
        
        # 计算每个文档的相关性分数
        scored_texts = []
        for item in processed_data:
            content = item.get('content', '')
            score = 0
            matches = []
            
            # 关键词匹配
            for term in key_terms:
                count = content.count(term)
                if count > 0:
                    score += count
                    matches.append(term)
            
            # 如果有匹配，添加到结果中
            if score > 0:
                scored_texts.append({
                    'content': content[:200] + "..." if len(content) > 200 else content,  # 只保留前200字符
                    'score': score,
                    'matches': matches,
                    'source': item.get('source', '未知来源')
                })
        
        # 按分数排序
        scored_texts.sort(key=lambda x: x['score'], reverse=True)
        
        if not scored_texts:
            logger.warning(f"未找到与关键词 {key_terms} 相关的内容")
            return None, f"未找到与查询 '{query}' 相关的内容。"
        
        # 记录匹配情况（简化输出）
        logger.info("\n=== 匹配结果摘要 ===")
        logger.info(f"找到相关文档数: {len(scored_texts)}")
        logger.info(f"最高匹配分数: {scored_texts[0]['score']}")
        logger.info(f"匹配的关键词: {list(set([term for text in scored_texts[:3] for term in text['matches']]))}")
        
        # 合并最相关的文本
        context = "\n\n".join([
            f"来源: {text['source']}\n{text['content']}"
            for text in scored_texts[:3]
        ])
        
        # 如果提供了模型实例，直接生成回答
        if model:
            result = model.generate_with_rag(query, context)
            return context, result["response"]
            
        # 否则返回上下文
        return context, None
        
    except Exception as e:
        logger.error(f"RAG查询失败: {str(e)}")
        return None, f"查询处理失败: {str(e)}"
