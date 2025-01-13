import os
import torch
from transformers import AutoModel, AutoTokenizer
import warnings
from utils.RAG import RAGRetriever, generate_prompt

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)

def load_model(model_path: str):
    """加载模型和分词器"""
    print(f"正在从路径加载模型: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        revision="main"
    )
    
    # 根据GPU显存选择加载方式
    if torch.cuda.is_available():  # 如果GPU可用
        device = "cuda"
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
            revision="main"
        ).to(device)
  
        
    print(f"模型已加载到设备: {device}")
    model.eval()
    return model, tokenizer

def main():
    try:
        # 设置模型路径
        model_path = os.path.abspath(r"G:\code\zy-LLM\models\ZhipuAI\ChatGLM-6B")
        
        # 加载模型和tokenizer
        print("正在加载ChatGLM模型...")
        model, tokenizer = load_model(model_path)
        
        # 初始化RAG检索器
        retriever = RAGRetriever("data/processed")
        print("RAG检索器初始化完成！")
        
        # 开始对话循环
        history = []
        while True:
            user_input = input("\n请输入您的问题 (输入 'quit' 退出): ").strip()
            
            if user_input.lower() == 'quit':
                print("感谢使用，再见！")
                break
                
            if not user_input:
                continue
                
            try:
                # 检索相关文档
                retrieved_docs = retriever.retrieve(user_input)
                
                # 生成带有上下文的提示词
                prompt = generate_prompt(user_input, retrieved_docs)
                
                # 生成回答（使用流式输出）
                with torch.inference_mode():
                    response = ""
                    last_response = ""
                    for response_chunk in model.stream_chat(
                        tokenizer,
                        prompt,  # 使用带有上下文的提示词
                        history=history,
                        temperature=0.7,
                        top_p=0.9,
                        max_length=2048,
                        repetition_penalty=1.1,
                        num_beams=1,
                        do_sample=True
                    ):
                        if isinstance(response_chunk, tuple):
                            chunk = response_chunk[0]
                        else:
                            chunk = response_chunk
                            
                        if chunk:
                            new_text = chunk[len(last_response):]
                            print(new_text, end="", flush=True)
                            last_response = chunk
                            response = chunk
                    print()
                    
                    if response:
                        history.append((user_input, response))
                        
            except Exception as e:
                print(f"生成回答时出错: {str(e)}")
                continue
                
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    # 设置环境变量以优化性能
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # 如果有CUDA，设置一些优化选项
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    main()

