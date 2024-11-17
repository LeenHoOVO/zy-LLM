from llama_index import VectorStoreIndex, SimpleDirectoryReader
from utils.model import load_chatglm_model, generate_response_from_model

def basic_rag(query, docs_dir="data"):
    """
    基础的RAG实现
    Args:
        query: 用户查询
        docs_dir: 文档目录路径
    """
    # 加载文档
    documents = SimpleDirectoryReader(docs_dir).load_data()
    
    # 创建索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 检索相关文档
    retriever = index.as_retriever()
    retrieved_nodes = retriever.retrieve(query)
    
    # 构建提示词
    context = "\n".join([node.text for node in retrieved_nodes])
    prompt = f"""作为易经专家，根据以下信息回答问题：

    易经信息：{context}
    问题：{query}
    请给出答案："""
    
    # 加载模型并生成回答
    model, tokenizer = load_chatglm_model()
    response = generate_response_from_model(model, tokenizer, prompt)
    
    return response
